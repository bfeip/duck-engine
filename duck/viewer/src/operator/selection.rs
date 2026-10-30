use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::bindings::{InputBinding, InputMap};
use crate::event::{AppEvent, DeviceEvent, Event, EventContext};
use crate::geom_query::{is_visible_along, pick_all_from_ray, RayPickQuery, RayPickResult};
use crate::input::{ElementState, Key, Modifiers, MouseButton, NamedKey};
use crate::operator::Operator;
use crate::scene::common::{Ray, Real};
use crate::scene::resource::{NodeId, SubGeometryElement, SubGeometryKind};
use crate::scene::Scene;
use crate::selection::{SelectionItem, SelectionManager};
use crate::ScreenOverlay;
use duck_engine_common::InnerSpace;

mod area;

use area::AreaDrag;

/// How close the cursor must come to an edge or point to pick it, in pixels.
const PICK_RADIUS_PX: Real = 6.0;
/// How far a face may sit in front of an edge or point that still counts as
/// visible, in pixels: room for error where the two meet.
const OCCLUSION_SLACK_PX: Real = 2.0;
/// Edges or points this close to each other in pixels tie, and a visible one
/// wins the tie.
const TIE_PX: Real = 1.0;

/// Semantic actions for the selection operator, bound to clicks and drags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SelectionAction {
    /// Replace the selection with the clicked item or the area's items
    /// (clearing it if there are none).
    Select,
    /// Toggle the clicked item in or out of the selection, or add the area's
    /// items to it.
    AddToSelection,
}

bitflags! {
    /// The kinds of geometry a selection may resolve to. Used as a mask on the
    /// non-`Node` [`SelectionMode`] variants to restrict what a click or an
    /// area can select.
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
    /// closest allowed sub-geometry. An area selects whole nodes.
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

/// Operator for selecting objects in the scene by clicking, or by dragging out
/// an area.
///
/// A click resolves to the highest kind within reach: a point, then an edge
/// (each within a few pixels of the cursor, nearest first), then the face under
/// the cursor. Edges and points a face hides are out of reach unless
/// [`hidden`](Self::hidden) allows their kind.
///
/// An area dragged rightward takes the items wholly inside it, one dragged
/// leftward every item it touches. Only items that show somewhere in the area
/// count, unless [`hidden`](Self::hidden) allows their kind. The area is drawn
/// over the view while dragging, and Escape abandons it.
pub struct SelectionOperator {
    pub bindings: InputMap<SelectionAction>,
    pub mode: SelectionMode,
    /// Kinds that stay selectable where faces hide them, for views that draw
    /// them (e.g. hidden edges in
    /// [`workflow::hidden_line`](crate::renderer::workflow::hidden_line)), or
    /// draw through everything ([`SelectionKinds::all`]). A click can only reach
    /// hidden [`EDGE`](SelectionKinds::EDGE)s and [`POINT`](SelectionKinds::POINT)s;
    /// an area honors every kind. Empty by default.
    pub hidden: SelectionKinds,
    /// The area being dragged out, if any.
    area: Option<AreaDrag>,
}

impl SelectionOperator {
    /// Creates a new selection operator in the default mode.
    pub fn new() -> Self {
        Self::with_mode(SelectionMode::default())
    }

    /// Creates a new selection operator with the given [`SelectionMode`].
    pub fn with_mode(mode: SelectionMode) -> Self {
        let shift = Modifiers { shift: true, ..Default::default() };
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                SelectionAction::Select,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: shift },
                SelectionAction::AddToSelection,
            )
            .bind(
                InputBinding::MouseDragStart {
                    button: MouseButton::Left,
                    modifiers: Modifiers::default(),
                },
                SelectionAction::Select,
            )
            .bind(
                InputBinding::MouseDragStart { button: MouseButton::Left, modifiers: shift },
                SelectionAction::AddToSelection,
            );
        Self { bindings, mode, hidden: SelectionKinds::empty(), area: None }
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

    fn click(&self, button: MouseButton, position: (f32, f32), ctx: &mut EventContext) -> bool {
        let Some(action) = action(self.bindings.actions_for_click(button, ctx.modifiers)) else {
            return false;
        };
        let picked = self.pick_closest(position.0, position.1, ctx);
        change_selection(ctx, |selection| match (action, picked) {
            (SelectionAction::Select, Some(item)) => selection.set(item),
            (SelectionAction::Select, None) => selection.clear(),
            (SelectionAction::AddToSelection, Some(item)) => selection.toggle(item),
            (SelectionAction::AddToSelection, None) => {}
        });
        true
    }

    fn finish_area(&self, area: &AreaDrag, ctx: &mut EventContext) {
        let items = area::select_area(area, self.mode, self.hidden, ctx);
        change_selection(ctx, |selection| {
            if area.action == SelectionAction::Select {
                selection.clear();
            }
            selection.extend(items);
        });
    }
}

impl Operator for SelectionOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseClick { button, position, .. } => self.click(*button, *position, ctx),
            DeviceEvent::MouseDragStart { button, start_pos, current_pos } => {
                let actions = self.bindings.actions_for_drag_start(*button, ctx.modifiers);
                let Some(action) = action(actions) else { return false };
                self.area =
                    Some(AreaDrag { button: *button, action, anchor: *start_pos, cursor: *current_pos });
                true
            }
            DeviceEvent::MouseDrag { button, current_pos, .. } => match &mut self.area {
                Some(area) if area.button == *button => {
                    area.cursor = *current_pos;
                    true
                }
                _ => false,
            },
            DeviceEvent::MouseDragEnd { button, end_pos, .. } => {
                let Some(mut area) = self.area.take_if(|area| area.button == *button) else {
                    return false;
                };
                area.cursor = *end_pos;
                self.finish_area(&area, ctx);
                true
            }
            DeviceEvent::KeyboardInput { event, .. }
                if self.area.is_some()
                    && event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                self.area = None;
                true
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Selection"
    }

    fn overlay(&self, overlay: &mut ScreenOverlay) {
        if let Some(area) = &self.area {
            overlay.add_rect(area.overlay_rect());
        }
    }
}

/// The action bound input asks for, `Select` winning when it asks for both.
fn action(actions: &[SelectionAction]) -> Option<SelectionAction> {
    [SelectionAction::Select, SelectionAction::AddToSelection]
        .into_iter()
        .find(|action| actions.contains(action))
}

/// Applies `change` to the selection and reports it.
fn change_selection(ctx: &mut EventContext, change: impl FnOnce(&mut SelectionManager)) {
    let previous = ctx.selection.as_slice().to_vec();
    change(ctx.selection);
    let current = ctx.selection.as_slice().to_vec();
    ctx.emit(AppEvent::Selection { previous, current });
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
    use crate::input::{KeyEvent, PhysicalKey};
    use crate::scene::{PositionedCamera, Projection, SceneData};

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

    // ===== Areas =====

    const SHIFT: Modifiers = Modifiers { shift: true, control: false, alt: false, super_key: false };

    /// The pixel [`camera`] shows the world line through `(x, y)` at.
    fn pixel(x: Real, y: Real) -> (f32, f32) {
        let (width, height) = VIEWPORT;
        ((width as Real / 2.0 + x / PIXEL) as f32, (height as Real / 2.0 - y / PIXEL) as f32)
    }

    /// Dispatches `events` to `op` with `modifiers` held, returning whether
    /// each was consumed.
    fn send(
        op: &mut SelectionOperator,
        scene: &Scene,
        selection: &mut SelectionManager,
        modifiers: Modifiers,
        events: impl IntoIterator<Item = DeviceEvent>,
    ) -> Vec<bool> {
        let mut cursor = None;
        let mut camera = camera();
        let mut ctx =
            EventContext::new(VIEWPORT, &mut cursor, scene.clone(), &mut camera, selection, modifiers);
        events.into_iter().map(|e| op.dispatch(&Event::Device(e), &mut ctx)).collect()
    }

    fn drag_start(from: (f32, f32)) -> DeviceEvent {
        DeviceEvent::MouseDragStart { button: MouseButton::Left, start_pos: from, current_pos: from }
    }

    fn drag_to(from: (f32, f32), to: (f32, f32)) -> DeviceEvent {
        let delta = (to.0 - from.0, to.1 - from.1);
        DeviceEvent::MouseDrag { button: MouseButton::Left, start_pos: from, current_pos: to, delta }
    }

    fn drag_end(from: (f32, f32), to: (f32, f32)) -> DeviceEvent {
        DeviceEvent::MouseDragEnd { button: MouseButton::Left, start_pos: from, end_pos: to }
    }

    /// Drags an area with `op` from world `(x, y)` to world `(x, y)` with
    /// `modifiers` held, and returns the selection it leaves.
    fn area(
        op: &mut SelectionOperator,
        scene: &Scene,
        selection: &mut SelectionManager,
        modifiers: Modifiers,
        from: (Real, Real),
        to: (Real, Real),
    ) -> Vec<SelectionItem> {
        let (from, to) = (pixel(from.0, from.1), pixel(to.0, to.1));
        let consumed =
            send(op, scene, selection, modifiers, [drag_start(from), drag_to(from, to), drag_end(from, to)]);
        assert_eq!(consumed, [true; 3]);
        selection.as_slice().to_vec()
    }

    fn select_area(op: &mut SelectionOperator, scene: &Scene, from: (Real, Real), to: (Real, Real)) -> Vec<SelectionItem> {
        area(op, scene, &mut SelectionManager::new(), Modifiers::default(), from, to)
    }

    /// A unit square face at depth `z`, centred on `x`.
    fn square_at(x: f32, z: f32) -> [Corner; 4] {
        square(z).map(|[cx, cy, cz]| [cx + x, cy, cz])
    }

    /// Dragging rightward takes what lies wholly inside, leftward what the
    /// area touches. Areas take whole nodes in the default progressive mode.
    #[test]
    fn area_direction_decides_between_inside_and_touching() {
        let scene = Scene::new(SceneData::new());
        let left = add_part(&scene, &[square_at(0.0, 0.0)], &[]);
        let right = add_part(&scene, &[square_at(3.0, 0.0)], &[]);
        let mut op = SelectionOperator::new();

        let window = select_area(&mut op, &scene, (-1.5, 1.5), (2.5, -1.5));
        assert_eq!(window, [SelectionItem::Node(left)]);

        let crossing = select_area(&mut op, &scene, (2.5, -1.5), (-1.5, 1.5));
        assert_eq!(crossing, [SelectionItem::Node(left), SelectionItem::Node(right)]);
    }

    #[test]
    fn area_resolves_to_faces_and_edges() {
        let scene = Scene::new(SceneData::new());
        let part = add_part(&scene, &[square(0.0)], &[vertical(0.5, 0.0)]);
        let mut op = sub_geometry_op();

        // Around the edge, across the face.
        assert_eq!(select_area(&mut op, &scene, (0.4, 1.2), (0.6, -1.2)), [edge(part, 0)]);
        assert_eq!(
            select_area(&mut op, &scene, (0.6, -1.2), (0.4, 1.2)),
            [face(part, 0), edge(part, 0)],
        );
        // Across the edge.
        assert!(select_area(&mut op, &scene, (0.4, 0.5), (0.6, -0.5)).is_empty());
        assert_eq!(select_area(&mut op, &scene, (0.6, -0.5), (0.4, 0.5)), [face(part, 0), edge(part, 0)]);
    }

    /// An area clear of every vertex and edge of the face it lies on.
    #[test]
    fn crossing_area_inside_a_face_takes_it() {
        let scene = Scene::new(SceneData::new());
        let part = add_part(&scene, &[square(0.0)], &[]);
        let mut op = sub_geometry_op();
        assert_eq!(select_area(&mut op, &scene, (0.3, -0.2), (0.1, 0.2)), [face(part, 0)]);
    }

    #[test]
    fn area_takes_hidden_items_only_when_allowed() {
        let scene = Scene::new(SceneData::new());
        let front = add_part(&scene, &[square(0.0)], &[]);
        let back = add_part(&scene, &[square(-2.0)], &[vertical(0.0, -2.0)]);
        let around = ((-1.5, 1.5), (1.5, -1.5));

        let mut op = SelectionOperator::with_mode(SelectionMode::Node);
        assert_eq!(select_area(&mut op, &scene, around.0, around.1), [SelectionItem::Node(front)]);
        op.hidden = SelectionKinds::NODE;
        assert_eq!(
            select_area(&mut op, &scene, around.0, around.1),
            [SelectionItem::Node(front), SelectionItem::Node(back)],
        );

        let mut op = sub_geometry_op();
        assert_eq!(select_area(&mut op, &scene, around.0, around.1), [face(front, 0)]);
        op.hidden = SelectionKinds::all();
        assert_eq!(
            select_area(&mut op, &scene, around.0, around.1),
            [face(front, 0), face(back, 0), edge(back, 0)],
        );
    }

    #[test]
    fn shift_area_adds_and_plain_area_replaces() {
        let scene = Scene::new(SceneData::new());
        let left = add_part(&scene, &[square_at(0.0, 0.0)], &[]);
        let right = add_part(&scene, &[square_at(3.0, 0.0)], &[]);
        let mut op = SelectionOperator::new();
        let mut selection = SelectionManager::new();
        let (around_left, around_right) = (((-1.5, 1.5), (1.5, -1.5)), ((1.5, 1.5), (4.5, -1.5)));

        area(&mut op, &scene, &mut selection, Modifiers::default(), around_left.0, around_left.1);
        let both = area(&mut op, &scene, &mut selection, SHIFT, around_right.0, around_right.1);
        assert_eq!(both, [SelectionItem::Node(left), SelectionItem::Node(right)]);

        let replaced =
            area(&mut op, &scene, &mut selection, Modifiers::default(), around_right.0, around_right.1);
        assert_eq!(replaced, [SelectionItem::Node(right)]);
    }

    #[test]
    fn area_is_drawn_while_dragging_in_its_kind_color() {
        let scene = Scene::new(SceneData::new());
        let mut op = SelectionOperator::new();
        let mut selection = SelectionManager::new();
        let drawn = |op: &SelectionOperator| {
            let mut overlay = ScreenOverlay::default();
            op.overlay(&mut overlay);
            overlay.rects().to_vec()
        };
        let (from, to) = ((100.0, 100.0), (200.0, 150.0));

        send(&mut op, &scene, &mut selection, Modifiers::default(), [drag_start(from), drag_to(from, to)]);
        let window = drawn(&op);
        assert_eq!(window.len(), 1);
        assert_eq!((window[0].min, window[0].max), (from, to));

        send(&mut op, &scene, &mut selection, Modifiers::default(), [drag_to(to, (50.0, 50.0))]);
        let crossing = drawn(&op);
        assert_eq!((crossing[0].min, crossing[0].max), ((50.0, 50.0), from));
        assert_ne!(crossing[0].stroke_color, window[0].stroke_color);

        send(&mut op, &scene, &mut selection, Modifiers::default(), [drag_end(from, to)]);
        assert!(drawn(&op).is_empty());
    }

    #[test]
    fn escape_abandons_an_area() {
        let scene = Scene::new(SceneData::new());
        let part = add_part(&scene, &[square(0.0)], &[]);
        let mut op = SelectionOperator::new();
        let mut selection = SelectionManager::new();
        selection.set(SelectionItem::Node(part));
        let escape = DeviceEvent::KeyboardInput {
            event: KeyEvent {
                physical_key: PhysicalKey::Unidentified,
                logical_key: Key::Named(NamedKey::Escape),
                state: ElementState::Pressed,
                repeat: false,
            },
            is_synthetic: false,
        };
        let (from, to) = (pixel(0.5, 0.5), pixel(2.0, -2.0));

        let consumed = send(
            &mut op,
            &scene,
            &mut selection,
            Modifiers::default(),
            [drag_start(from), escape.clone(), drag_end(from, to), escape],
        );
        assert_eq!(consumed, [true, true, false, false]);
        assert_eq!(selection.as_slice(), [SelectionItem::Node(part)]);
    }
}
