//! Area selection: what a rectangle dragged across the view selects.

use duck_engine_common::{
    array_to_point3, centroid_of_slice, transform_point, ConvexPolyhedron, InnerSpace, Point3,
    RgbaColor,
};

use crate::event::EventContext;
use crate::geom_query::{is_visible_along, pick_all_from_volume, Containment, VolumePickResult};
use crate::input::MouseButton;
use crate::scene::resource::{Mesh, SubGeometryElement, SubGeometryKind};
use crate::selection::SelectionItem;
use crate::OverlayRect;

use super::{SelectionAction, SelectionKinds, SelectionMode, OCCLUSION_SLACK_PX};

/// How many points of an item are tested for visibility at most.
const MAX_VISIBILITY_SAMPLES: usize = 8;

const WINDOW_COLOR: RgbaColor = RgbaColor { r: 0.2, g: 0.5, b: 1.0, a: 1.0 };
const CROSSING_COLOR: RgbaColor = RgbaColor { r: 0.25, g: 0.8, b: 0.35, a: 1.0 };
const FILL_ALPHA: f32 = 0.15;

/// Which items an area takes, by the direction it was dragged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AreaKind {
    /// Dragged rightward: the items wholly inside.
    Window,
    /// Dragged leftward: the items it touches.
    Crossing,
}

impl AreaKind {
    fn accepts(self, containment: Containment) -> bool {
        match self {
            AreaKind::Window => containment == Containment::Inside,
            AreaKind::Crossing => containment != Containment::Outside,
        }
    }
}

/// An area being dragged out, in view pixels.
pub(super) struct AreaDrag {
    pub button: MouseButton,
    pub action: SelectionAction,
    /// Where the drag started.
    pub anchor: (f32, f32),
    pub cursor: (f32, f32),
}

impl AreaDrag {
    fn kind(&self) -> AreaKind {
        if self.cursor.0 >= self.anchor.0 { AreaKind::Window } else { AreaKind::Crossing }
    }

    /// The top-left and bottom-right corners.
    fn corners(&self) -> ((f32, f32), (f32, f32)) {
        let (a, b) = (self.anchor, self.cursor);
        ((a.0.min(b.0), a.1.min(b.1)), (a.0.max(b.0), a.1.max(b.1)))
    }

    /// The rectangle drawn while dragging, colored by kind.
    pub fn overlay_rect(&self) -> OverlayRect {
        let (min, max) = self.corners();
        let color = match self.kind() {
            AreaKind::Window => WINDOW_COLOR,
            AreaKind::Crossing => CROSSING_COLOR,
        };
        OverlayRect {
            min,
            max,
            fill_color: color.with_alpha(FILL_ALPHA),
            stroke_color: color,
            stroke_width: 1.0,
        }
    }
}

/// An item the area takes if it shows.
struct Candidate {
    item: SelectionItem,
    /// World points on the item inside the area, any one of which showing
    /// shows the item; `None` when it needn't show.
    samples: Option<Vec<Point3>>,
}

/// The items `drag` selects at the resolution `mode` gives, in scene order.
/// Items of a kind in `hidden` are taken where faces hide them; any other must
/// show somewhere in the area.
pub(super) fn select_area(
    drag: &AreaDrag,
    mode: SelectionMode,
    hidden: SelectionKinds,
    ctx: &EventContext,
) -> Vec<SelectionItem> {
    let (min, max) = drag.corners();
    if max.0 - min.0 < 1.0 || max.1 - min.1 < 1.0 {
        return Vec::new();
    }
    let kind = drag.kind();
    let camera = &*ctx.camera;
    let (width, height) = ctx.size;
    let volume = camera.volume_from_screen_rect(min, max, width, height);
    let hits = pick_all_from_volume(&volume, &ctx.scene, kind == AreaKind::Crossing);

    let candidates: Vec<Candidate> = {
        let scene = ctx.scene.lock();
        hits.iter()
            .filter_map(|hit| {
                let mesh = scene.get_instance(hit.instance_id).and_then(|i| scene.get_mesh(i.mesh()))?;
                Some(candidates(hit, mesh, mode, kind, hidden, &volume))
            })
            .flatten()
            .collect()
    };

    // The scene lock is released: each visibility test picks through the scene again.
    let pixel_size = camera.world_size_per_pixel((camera.eye - camera.target).magnitude(), height);
    let slack = OCCLUSION_SLACK_PX * pixel_size;
    let shows = |point: &Point3| {
        let screen = camera.project_point_screen(*point, width, height);
        let ray = camera.ray_from_screen_point(screen.x as f32, screen.y as f32, width, height);
        is_visible_along(&ctx.scene, &ray, *point, slack)
    };
    candidates
        .into_iter()
        .filter(|c| c.samples.as_ref().is_none_or(|samples| samples.iter().any(shows)))
        .map(|c| c.item)
        .collect()
}

/// The items one instance offers the area: the node as a whole, or the
/// topology elements of the kinds `mode` allows.
fn candidates(
    hit: &VolumePickResult,
    mesh: &Mesh,
    mode: SelectionMode,
    kind: AreaKind,
    hidden: SelectionKinds,
    volume: &ConvexPolyhedron,
) -> Vec<Candidate> {
    let candidate = |item, item_kind: SelectionKinds, primitives: Vec<(SubGeometryKind, usize)>| {
        let samples = (!hidden.intersects(item_kind)).then(|| samples(hit, mesh, &primitives, volume));
        Candidate { item, samples }
    };
    let node = || {
        let primitives = KINDS.into_iter().flat_map(|k| touching(hit, k, 0..usize::MAX)).collect();
        kind.accepts(hit.hit.containment())
            .then(|| candidate(SelectionItem::Node(hit.node_id), SelectionKinds::NODE, primitives))
    };

    let kinds = match mode {
        SelectionMode::Node | SelectionMode::Progressive(_) => return node().into_iter().collect(),
        SelectionMode::SubGeometry(kinds) => kinds,
    };
    let sub = kinds & SelectionKinds::sub_geometry();
    let topology = mesh.topology().filter(|t| !t.is_empty() && !sub.is_empty());
    let Some(topology) = topology else {
        // Without elements to resolve to, the node stands in if allowed.
        return kinds.contains(SelectionKinds::NODE).then(node).flatten().into_iter().collect();
    };

    let mut found = Vec::new();
    for element_kind in KINDS.into_iter().filter(|&k| sub.contains(SelectionKinds::from(k))) {
        let containments = hit.hit.of_kind(element_kind);
        for (index, range) in topology.ranges(element_kind).iter().enumerate() {
            let primitives = range.start as usize..(range.start + range.count) as usize;
            let containment =
                Containment::of(containments.get(primitives.clone()).into_iter().flatten().copied());
            if !kind.accepts(containment) {
                continue;
            }
            let element = SubGeometryElement::new(element_kind, index as u32);
            found.push(candidate(
                SelectionItem::SubGeometry { node_id: hit.node_id, element },
                SelectionKinds::from(element_kind),
                touching(hit, element_kind, primitives).collect(),
            ));
        }
    }
    found
}

const KINDS: [SubGeometryKind; 3] =
    [SubGeometryKind::Face, SubGeometryKind::Edge, SubGeometryKind::Pointset];

/// The primitives of `kind` among `range` that touch the volume.
fn touching(
    hit: &VolumePickResult,
    kind: SubGeometryKind,
    range: std::ops::Range<usize>,
) -> impl Iterator<Item = (SubGeometryKind, usize)> + '_ {
    let containments = hit.hit.of_kind(kind);
    let range = range.start.min(containments.len())..range.end.min(containments.len());
    range
        .filter(move |&i| containments[i] != Containment::Outside)
        .map(move |i| (kind, i))
}

/// Up to [`MAX_VISIBILITY_SAMPLES`] world points, spread over `primitives`:
/// the center of each one's part inside the volume.
fn samples(
    hit: &VolumePickResult,
    mesh: &Mesh,
    primitives: &[(SubGeometryKind, usize)],
    volume: &ConvexPolyhedron,
) -> Vec<Point3> {
    let stride = primitives.len().div_ceil(MAX_VISIBILITY_SAMPLES).max(1);
    primitives
        .iter()
        .step_by(stride)
        .filter_map(|&(kind, index)| {
            let positions: Vec<[f32; 3]> = match kind {
                SubGeometryKind::Face => mesh.triangle(index)?.map(|v| v.position).to_vec(),
                SubGeometryKind::Edge => mesh.segment(index)?.map(|v| v.position).to_vec(),
                SubGeometryKind::Pointset => vec![mesh.point(index)?.position],
            };
            let world: Vec<Point3> = positions
                .into_iter()
                .map(|p| transform_point(&hit.transform, array_to_point3(p)))
                .collect();
            centroid_of_slice(&volume.clip_polygon(&world))
        })
        .collect()
}
