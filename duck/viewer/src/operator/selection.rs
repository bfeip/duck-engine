use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::bindings::{InputBinding, InputMap};
use crate::event::{AppEvent, DeviceEvent, Event, EventContext};
use crate::geom_query::{is_visible_along, pick_all_from_ray, RayPickQuery, RayPickResult};
use crate::input::{Modifiers, MouseButton};
use crate::operator::Operator;
use crate::scene::common::{Ray, Real};
use crate::scene::resource::{NodeId, SubGeometryElement, SubGeometryKind};
use crate::scene::Scene;
use crate::selection::SelectionItem;
use duck_engine_common::InnerSpace;

/// How close the cursor must come to an edge or point to pick it, in pixels.
const PICK_RADIUS_PX: Real = 6.0;
/// How far a face may sit in front of an edge or point that still counts as
/// visible, in pixels: room for error where the two meet.
const OCCLUSION_SLACK_PX: Real = 2.0;
/// Edges or points this close to each other in pixels tie, and a visible one
/// wins the tie.
const TIE_PX: Real = 1.0;

/// Semantic actions for the selection operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SelectionAction {
    /// Cast a ray and replace the selection with the hit item (or clear if nothing hit).
    Select,
    /// Cast a ray and toggle the hit item in/out of the current selection.
    AddToSelection,
}

bitflags! {
    /// The kinds of geometry a selection may resolve to. Used as a mask on the
    /// non-`Node` [`SelectionMode`] variants to restrict what a click can select.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct SelectionKinds: u8 {
        /// Whole nodes (also acts as the fallback when a hit has no topology mapping).
        const NODE = 1 << 0;
        /// Faces.
        const FACE = 1 << 1;
        /// Edges.
        const EDGE = 1 << 2;
        /// Points.
        const POINT = 1 << 3;
    }
}

impl SelectionKinds {
    /// All sub-geometry kinds (everything except [`NODE`](Self::NODE)).
    pub fn sub_geometry() -> Self {
        Self::FACE | Self::EDGE | Self::POINT
    }
}

impl From<SubGeometryKind> for SelectionKinds {
    fn from(kind: SubGeometryKind) -> Self {
        match kind {
            SubGeometryKind::Face => SelectionKinds::FACE,
            SubGeometryKind::Edge => SelectionKinds::EDGE,
            SubGeometryKind::Pointset => SelectionKinds::POINT,
        }
    }
}

/// Controls what granularity a click resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionMode {
    /// Whole-node only, ignoring face/edge/point topology.
    Node,
    /// Resolve to the closest hit among the allowed kinds. If the [`NODE`] flag is
    /// set it acts as a fallback when the hit primitive has no topology mapping (or
    /// when no sub-geometry kinds are allowed).
    ///
    /// e.g. `SubGeometry(FACE)` = faces only; `SubGeometry(FACE | EDGE)` = either.
    SubGeometry(SelectionKinds),
    /// Node first; a second click on an already-selected node drills into the
    /// closest allowed sub-geometry.
    Progressive(SelectionKinds),
}

impl Default for SelectionMode {
    fn default() -> Self {
        SelectionMode::Progressive(SelectionKinds::all())
    }
}

impl SelectionMode {
    /// Which primitive types the ray query should test, as `(faces, lines, points)`.
    fn query_primitives(&self) -> (bool, bool, bool) {
        let kinds = match self {
            // Node and node-fallback both need *some* hit to obtain a node id, so
            // query everything.
            SelectionMode::Node => return (true, true, true),
            SelectionMode::SubGeometry(k) | SelectionMode::Progressive(k) => *k,
        };
        let sub = kinds & SelectionKinds::sub_geometry();
        if sub.is_empty() {
            // Only a node fallback is requested — query everything to get a node hit.
            (true, true, true)
        } else {
            (
                sub.contains(SelectionKinds::FACE),
                sub.contains(SelectionKinds::EDGE),
                sub.contains(SelectionKinds::POINT),
            )
        }
    }
}

/// Operator for selecting objects in the scene via mouse click.
///
/// A click resolves to the highest kind within reach: a point, then an edge
/// (each within a few pixels of the cursor, nearest first), then the face under
/// the cursor. Edges and points a face hides are out of reach unless
/// [`hidden`](Self::hidden) allows their kind.
pub struct SelectionOperator {
    pub bindings: InputMap<SelectionAction>,
    pub mode: SelectionMode,
    /// Sub-geometry kinds that stay selectable where faces hide them, for views
    /// that draw them (e.g. hidden edges in
    /// [`workflow::hidden_line`](crate::renderer::workflow::hidden_line)). Only
    /// [`EDGE`](SelectionKinds::EDGE) and [`POINT`](SelectionKinds::POINT) apply.
    /// Empty by default.
    pub hidden: SelectionKinds,
}

impl SelectionOperator {
    /// Creates a new selection operator in the default mode.
    pub fn new() -> Self {
        Self::with_mode(SelectionMode::default())
    }

    /// Creates a new selection operator with the given [`SelectionMode`].
    pub fn with_mode(mode: SelectionMode) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                SelectionAction::Select,
            )
            .bind(
                InputBinding::MouseClick {
                    button: MouseButton::Left,
                    modifiers: Modifiers { shift: true, ..Default::default() },
                },
                SelectionAction::AddToSelection,
            );
        Self { bindings, mode, hidden: SelectionKinds::empty() }
    }

    fn pick_closest(&self, cursor_x: f32, cursor_y: f32, ctx: &mut EventContext) -> Option<SelectionItem> {
        let camera = ctx.camera.clone();
        let ray = camera.ray_from_screen_point(cursor_x, cursor_y, ctx.size.0, ctx.size.1);
        let camera_distance = (camera.eye - camera.target).magnitude();
        // World size of a pixel, calibrated to the camera target depth.
        // Approximation: uses a single depth reference, so the effective pixel budget
        // varies with geometry depth (near objects get more pixels, far objects fewer).
        let pixel_size = camera.world_size_per_pixel(camera_distance, ctx.size.1);
        let tie = TIE_PX * pixel_size;

        let (pick_faces, pick_lines, pick_points) = self.mode.query_primitives();
        let query = RayPickQuery::for_kinds(
            ray,
            PICK_RADIUS_PX * pixel_size,
            pick_faces,
            pick_lines,
            pick_points,
        );
        let results = pick_all_from_ray(&query, &ctx.scene);

        match self.mode {
            SelectionMode::Node => results.first().map(|hit| SelectionItem::Node(hit.node_id)),

            SelectionMode::SubGeometry(kinds) => {
                let candidates = candidates(&results, &ctx.scene, &ray, pixel_size, self.hidden);
                let sub = kinds & SelectionKinds::sub_geometry();
                resolve(&candidates, sub, None, tie).or_else(|| {
                    kinds
                        .contains(SelectionKinds::NODE)
                        .then(|| best(&candidates, tie))
                        .flatten()
                        .map(|c| SelectionItem::Node(c.hit.node_id))
                })
            }

            SelectionMode::Progressive(kinds) => {
                let candidates = candidates(&results, &ctx.scene, &ray, pixel_size, self.hidden);
                // The node the click means is the one owning its best candidate, so a
                // second click lands on that same candidate.
                let node = best(&candidates, tie)?.hit.node_id;
                // Drill into sub-geometry once the node is selected — whether the whole
                // node or any of its sub-geometry is currently the active selection.
                if ctx.selection.is_node_selected(node) {
                    let sub = kinds & SelectionKinds::sub_geometry();
                    resolve(&candidates, sub, Some(node), tie).or(Some(SelectionItem::Node(node)))
                } else {
                    Some(SelectionItem::Node(node))
                }
            }
        }
    }

    fn perform_selection(&self, cursor_x: f32, cursor_y: f32, ctx: &mut EventContext) {
        match self.pick_closest(cursor_x, cursor_y, ctx) {
            Some(item) => ctx.selection.set(item),
            None => ctx.selection.clear(),
        }
    }

    fn perform_add_to_selection(&self, cursor_x: f32, cursor_y: f32, ctx: &mut EventContext) {
        if let Some(item) = self.pick_closest(cursor_x, cursor_y, ctx) {
            ctx.selection.toggle(item);
        }
    }
}

impl Operator for SelectionOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(DeviceEvent::MouseClick { button, position, .. }) = event else {
            return false;
        };
        let actions = self.bindings.actions_for_click(*button, ctx.modifiers);
        let select = actions.contains(&SelectionAction::Select);
        let add = actions.contains(&SelectionAction::AddToSelection);
        if !select && !add {
            return false;
        }

        let previous = ctx.selection.as_slice().to_vec();
        if select {
            self.perform_selection(position.0, position.1, ctx);
        } else {
            self.perform_add_to_selection(position.0, position.1, ctx);
        }
        let current = ctx.selection.as_slice().to_vec();
        ctx.emit(AppEvent::Selection { previous, current });
        true
    }

    fn name(&self) -> &str {
        "Selection"
    }
}

/// A ray hit a click may resolve to.
struct Candidate<'a> {
    hit: &'a RayPickResult,
    /// The topology element the hit resolves to, if its mesh has topology.
    element: Option<SubGeometryElement>,
    /// False for an edge or point a face hides.
    visible: bool,
}

/// Gathers the hits a click may resolve to from `results` (sorted closest-first):
/// faces as they are, and each edge or point element once, by its hit nearest the
/// ray. Edges and points a face hides are dropped unless `hidden` allows their kind.
fn candidates<'a>(
    results: &'a [RayPickResult],
    scene: &Scene,
    ray: &Ray,
    pixel_size: Real,
    hidden: SelectionKinds,
) -> Vec<Candidate<'a>> {
    let elements: Vec<Option<SubGeometryElement>> = {
        let scene = scene.lock();
        results
            .iter()
            .map(|hit| {
                let kind = hit.hit.kind();
                let mesh = scene
                    .get_instance(hit.instance_id)
                    .and_then(|inst| scene.get_mesh(inst.mesh()));
                mesh.and_then(|m| m.sub_geometry_for_primitive(kind, hit.hit.primitive_index()))
                    .map(|index| SubGeometryElement::new(kind, index))
            })
            .collect()
    };

    let mut candidates: Vec<Candidate> = Vec::new();
    for (hit, element) in results.iter().zip(elements) {
        let kind = hit.hit.kind();
        if kind != SubGeometryKind::Face {
            // Hits without topology only stand for their node, so they collapse per node too.
            let same = candidates.iter_mut().find(|c| {
                c.hit.node_id == hit.node_id && c.hit.hit.kind() == kind && c.element == element
            });
            if let Some(same) = same {
                if hit.hit.distance_to_ray() < same.hit.hit.distance_to_ray() {
                    same.hit = hit;
                }
                continue;
            }
        }
        candidates.push(Candidate { hit, element, visible: true });
    }

    // The scene lock is released: each visibility test picks through the scene again.
    let slack = OCCLUSION_SLACK_PX * pixel_size;
    candidates.retain_mut(|c| {
        let kind = c.hit.hit.kind();
        if kind == SubGeometryKind::Face {
            return true;
        }
        c.visible = is_visible_along(scene, ray, c.hit.hit_point, slack);
        c.visible || hidden.contains(SelectionKinds::from(kind))
    });
    candidates
}

/// The candidate a click means: one of the highest kind present (point, then
/// edge, then face). Faces rank by depth. Points and edges rank by distance from
/// the cursor ray, a visible one winning a tie (within `tie`) with a hidden one.
fn best<'c, 'a>(
    candidates: impl IntoIterator<Item = &'c Candidate<'a>>,
    tie: Real,
) -> Option<&'c Candidate<'a>> {
    let candidates: Vec<&Candidate> = candidates.into_iter().collect();
    let kind = [SubGeometryKind::Pointset, SubGeometryKind::Edge, SubGeometryKind::Face]
        .into_iter()
        .find(|&kind| candidates.iter().any(|c| c.hit.hit.kind() == kind))?;
    let of_kind = candidates.into_iter().filter(|c| c.hit.hit.kind() == kind);

    if kind == SubGeometryKind::Face {
        return of_kind.min_by(|a, b| a.hit.distance.total_cmp(&b.hit.distance));
    }
    let of_kind: Vec<&Candidate> = of_kind.collect();
    let nearest = of_kind
        .iter()
        .map(|c| c.hit.hit.distance_to_ray())
        .fold(Real::INFINITY, Real::min);
    of_kind
        .into_iter()
        .filter(|c| c.hit.hit.distance_to_ray() <= nearest + tie)
        .min_by(|a, b| {
            b.visible
                .cmp(&a.visible)
                .then(a.hit.hit.distance_to_ray().total_cmp(&b.hit.hit.distance_to_ray()))
        })
}

/// The [`best`] candidate that resolves to a topology element of a kind in
/// `kinds`, considering only `node`'s candidates if given.
fn resolve(
    candidates: &[Candidate],
    kinds: SelectionKinds,
    node: Option<NodeId>,
    tie: Real,
) -> Option<SelectionItem> {
    let allowed = candidates.iter().filter(|c| {
        c.element.is_some()
            && kinds.contains(SelectionKinds::from(c.hit.hit.kind()))
            && node.is_none_or(|node| c.hit.node_id == node)
    });
    let winner = best(allowed, tie)?;
    Some(SelectionItem::SubGeometry { node_id: winner.hit.node_id, element: winner.element? })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::common::{Point3, Transform, Vector3};
    use crate::scene::resource::{
        FaceMaterial, Instance, Mesh, MeshPrimitive, NodeFlags, PrimitiveType, SubMeshRange,
        Topology, Vertex,
    };
    use crate::scene::{PositionedCamera, Projection, SceneData};
    use crate::selection::SelectionManager;

    const VIEWPORT: (u32, u32) = (800, 600);
    /// World size of one pixel under [`camera`]: a 6-unit-tall view over 600 px.
    const PIXEL: Real = 0.01;
    /// The cursor row, a world height of 0.5 under [`camera`]: clear of the
    /// fixtures' quad diagonals.
    const ROW: f32 = 250.0;

    type Corner = [f32; 3];

    /// Orthographic camera on +Z looking at the origin, so the screen pixel
    /// `(400 + x / PIXEL, 300 - y / PIXEL)` sees the world line through `(x, y)`.
    fn camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 10.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::unit_y(),
            aspect: VIEWPORT.0 as Real / VIEWPORT.1 as Real,
            projection: Projection::Orthographic { half_height: 3.0, half_depth: 100.0 },
        }
    }

    /// The cursor column `pixels` right of the screen centre.
    fn column(pixels: f32) -> f32 {
        VIEWPORT.0 as f32 / 2.0 + pixels
    }

    /// Adds a part whose faces are `quads` and whose edges are `segments`, each
    /// one topology element, in order.
    fn add_part(scene: &Scene, quads: &[[Corner; 4]], segments: &[[Corner; 2]]) -> NodeId {
        let vertex = |position: Corner| Vertex {
            position,
            tex_coords: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
        };
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for quad in quads {
            let base = vertices.len() as u32;
            vertices.extend(quad.iter().copied().map(vertex));
            triangles.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let mut lines = Vec::new();
        for segment in segments {
            lines.extend([vertices.len() as u32, vertices.len() as u32 + 1]);
            vertices.extend(segment.iter().copied().map(vertex));
        }
        let mut mesh = Mesh::from_raw(
            vertices,
            vec![
                MeshPrimitive { primitive_type: PrimitiveType::TriangleList, indices: triangles },
                MeshPrimitive { primitive_type: PrimitiveType::LineList, indices: lines },
            ],
        );
        mesh.set_topology(Topology {
            face_ranges: (0..quads.len() as u32)
                .map(|i| SubMeshRange { start: 2 * i, count: 2 })
                .collect(),
            edge_ranges: (0..segments.len() as u32)
                .map(|i| SubMeshRange { start: i, count: 1 })
                .collect(),
            pointset_ranges: Vec::new(),
        });
        let mesh = scene.add_mesh(mesh);
        let material = scene.add_face_material(FaceMaterial::new());
        scene
            .add_instance_node(
                None,
                Instance::new(mesh).with_face_material(material),
                None,
                Transform::IDENTITY,
                NodeFlags::NONE,
            )
            .unwrap()
            .id()
    }

    /// A unit-square face at depth `z`.
    fn square(z: f32) -> [Corner; 4] {
        [[-1.0, -1.0, z], [1.0, -1.0, z], [1.0, 1.0, z], [-1.0, 1.0, z]]
    }

    /// A vertical edge at `x`, depth `z`.
    fn vertical(x: f32, z: f32) -> [Corner; 2] {
        [[x, -1.0, z], [x, 1.0, z]]
    }

    fn face(node_id: NodeId, index: u32) -> SelectionItem {
        SelectionItem::SubGeometry {
            node_id,
            element: SubGeometryElement::new(SubGeometryKind::Face, index),
        }
    }

    fn edge(node_id: NodeId, index: u32) -> SelectionItem {
        SelectionItem::SubGeometry {
            node_id,
            element: SubGeometryElement::new(SubGeometryKind::Edge, index),
        }
    }

    /// Clicks with `op` at `(x, ROW)` and returns what it picks, given the current
    /// `selection`.
    fn click(
        op: &SelectionOperator,
        scene: &Scene,
        selection: &mut SelectionManager,
        x: f32,
    ) -> Option<SelectionItem> {
        let mut cursor = Some((x, ROW));
        let mut camera = camera();
        let mut ctx = EventContext {
            size: VIEWPORT,
            cursor_position: &mut cursor,
            scene: scene.clone(),
            camera: &mut camera,
            selection,
            modifiers: Modifiers::default(),
            emit_queue: Vec::new(),
        };
        op.pick_closest(x, ROW, &mut ctx)
    }

    fn pick(op: &SelectionOperator, scene: &Scene, x: f32) -> Option<SelectionItem> {
        click(op, scene, &mut SelectionManager::new(), x)
    }

    fn sub_geometry_op() -> SelectionOperator {
        SelectionOperator::with_mode(SelectionMode::SubGeometry(SelectionKinds::sub_geometry()))
    }

    /// Two faces meeting in a valley that opens toward the camera, with an edge
    /// along the crease at x = 0.
    fn concave_corner(scene: &Scene) -> NodeId {
        add_part(
            scene,
            &[
                [[-1.0, -1.0, 1.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 1.0]],
                [[0.0, -1.0, 0.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
            ],
            &[vertical(0.0, 0.0)],
        )
    }

    #[test]
    fn concave_edge_is_picked_beside_the_crease() {
        let scene = Scene::new(SceneData::new());
        let part = concave_corner(&scene);
        // The ray strikes the right face in front of the crease, but the crease
        // is in view and within reach.
        assert_eq!(pick(&sub_geometry_op(), &scene, column(3.0)), Some(edge(part, 0)));
    }

    #[test]
    fn face_is_picked_beyond_reach_of_the_crease() {
        let scene = Scene::new(SceneData::new());
        let part = concave_corner(&scene);
        assert_eq!(pick(&sub_geometry_op(), &scene, column(20.0)), Some(face(part, 1)));
    }

    #[test]
    fn hidden_edge_is_picked_only_when_allowed() {
        let scene = Scene::new(SceneData::new());
        let part = add_part(&scene, &[square(0.0)], &[vertical(0.0, -1.0)]);
        let mut op = sub_geometry_op();
        assert_eq!(pick(&op, &scene, column(3.0)), Some(face(part, 0)));

        op.hidden = SelectionKinds::EDGE;
        assert_eq!(pick(&op, &scene, column(3.0)), Some(edge(part, 0)));
    }

    /// Front and back edges that overlap on screen: the visible one wins.
    #[test]
    fn visible_edge_wins_a_tie_with_a_hidden_one() {
        let scene = Scene::new(SceneData::new());
        let part = add_part(&scene, &[square(0.0)], &[vertical(0.0, -2.0), vertical(0.0, 0.0)]);
        let mut op = sub_geometry_op();
        op.hidden = SelectionKinds::EDGE;
        assert_eq!(pick(&op, &scene, column(3.0)), Some(edge(part, 1)));
    }

    /// Out of a tie, the edge nearer the cursor wins even when hidden.
    #[test]
    fn hidden_edge_nearer_the_cursor_wins() {
        let scene = Scene::new(SceneData::new());
        let hidden_x = 0.0;
        let visible_x = 5.0 * PIXEL as f32;
        let part = add_part(
            &scene,
            &[square(0.0)],
            &[vertical(hidden_x, -2.0), vertical(visible_x, 0.0)],
        );
        let mut op = sub_geometry_op();
        assert_eq!(pick(&op, &scene, column(1.0)), Some(edge(part, 1)));

        op.hidden = SelectionKinds::EDGE;
        assert_eq!(pick(&op, &scene, column(1.0)), Some(edge(part, 0)));
    }

    /// A hidden edge of a part behind another: the first click selects its part,
    /// the second the edge.
    #[test]
    fn progressive_reaches_a_hidden_edge_through_another_part() {
        let scene = Scene::new(SceneData::new());
        let front = add_part(&scene, &[square(0.0)], &[]);
        let back = add_part(&scene, &[square(-2.0)], &[vertical(0.0, -2.0)]);
        let mut op = SelectionOperator::new();
        let mut selection = SelectionManager::new();
        assert_eq!(
            click(&op, &scene, &mut selection, column(3.0)),
            Some(SelectionItem::Node(front)),
        );

        op.hidden = SelectionKinds::EDGE;
        let first = click(&op, &scene, &mut selection, column(3.0));
        assert_eq!(first, Some(SelectionItem::Node(back)));
        selection.set(first.unwrap());
        assert_eq!(click(&op, &scene, &mut selection, column(3.0)), Some(edge(back, 0)));
    }
}
