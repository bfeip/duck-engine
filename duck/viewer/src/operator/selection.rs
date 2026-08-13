use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::bindings::{InputBinding, InputMap};
use crate::event::{AppEvent, DeviceEvent, Event, EventContext};
use crate::geom_query::{RayPickQuery, RayPickResult, pick_all_from_ray};
use crate::input::{Modifiers, MouseButton};
use crate::operator::Operator;
use crate::scene::resource::{SubGeometryElement, SubGeometryKind};
use crate::scene::Scene;
use crate::selection::SelectionItem;
use duck_engine_common::InnerSpace;

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
pub struct SelectionOperator {
    pub bindings: InputMap<SelectionAction>,
    pub mode: SelectionMode,
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
        Self { bindings, mode }
    }

    fn pick_closest(&self, cursor_x: f32, cursor_y: f32, ctx: &mut EventContext) -> Option<SelectionItem> {
        let camera = ctx.camera.clone();
        let ray = camera.ray_from_screen_point(cursor_x, cursor_y, ctx.size.0, ctx.size.1);
        let camera_distance = (camera.eye - camera.target).magnitude();
        // Line/point tolerance: 6 pixels in world space, calibrated to the camera target depth.
        // Approximation: uses a single depth reference, so the effective pixel budget
        // varies with geometry depth (near objects get more pixels, far objects fewer).
        let tolerance = camera.world_size_per_pixel(camera_distance, ctx.size.1) * 6.0;

        let (pick_faces, pick_lines, pick_points) = self.mode.query_primitives();
        let query = RayPickQuery::for_kinds(ray, tolerance, pick_faces, pick_lines, pick_points);
        let results = pick_all_from_ray(&query, &ctx.scene);
        let first_node = results.first().map(|hit| hit.node_id);

        match self.mode {
            SelectionMode::Node => first_node.map(SelectionItem::Node),

            SelectionMode::SubGeometry(kinds) => {
                let sub = kinds & SelectionKinds::sub_geometry();
                let allow_node = kinds.contains(SelectionKinds::NODE);
                if sub.is_empty() {
                    return allow_node.then_some(first_node.map(SelectionItem::Node)).flatten();
                }
                resolve_sub_geometry(&results, &ctx.scene, sub)
                    .or_else(|| allow_node.then_some(first_node.map(SelectionItem::Node)).flatten())
            }

            SelectionMode::Progressive(kinds) => {
                let node = first_node?;
                // Drill into sub-geometry once the node is selected — whether the whole
                // node or any of its sub-geometry is currently the active selection.
                if ctx.selection.is_node_selected(node) {
                    let sub = kinds & SelectionKinds::sub_geometry();
                    resolve_sub_geometry(&results, &ctx.scene, sub)
                        .or(Some(SelectionItem::Node(node)))
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

/// Scans ray pick results (sorted closest-first) for the first hit whose kind is
/// allowed by `sub_kinds` and which resolves to a topology element. Hits of a
/// disallowed kind, or allowed hits whose mesh lacks the relevant topology, are
/// skipped so a deeper but resolvable element can still win.
fn resolve_sub_geometry(
    results: &[RayPickResult],
    scene: &Scene,
    sub_kinds: SelectionKinds,
) -> Option<SelectionItem> {
    let scene = scene.lock();
    for hit in results {
        let kind = hit.hit.kind();
        if !sub_kinds.contains(SelectionKinds::from(kind)) {
            continue;
        }
        let mesh = scene
            .get_instance(hit.instance_id)
            .and_then(|inst| scene.get_mesh(inst.mesh()));

        if let Some(index) = mesh.and_then(|m| m.sub_geometry_for_primitive(kind, hit.hit.primitive_index())) {
            return Some(SelectionItem::SubGeometry {
                node_id: hit.node_id,
                element: SubGeometryElement::new(kind, index),
            });
        }
    }
    None
}
