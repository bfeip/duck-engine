//! Move, rotate and scale.
//!
//! [`Manipulation`] is the transform itself: its constraint, what drives it —
//! a gizmo grip, the cursor after a key, or values typed in — and the delta
//! that comes out. A tool holds one, feeds it events and grip drags, and acts on
//! the [`Outcome`]s it returns by previewing and committing the delta on
//! whatever it moves. [`values_ui`] shows its values in the tool's panel.

mod gizmo;
mod manipulation;
mod typed;

pub use manipulation::TransformMode;
pub(super) use manipulation::{Frame, Input, Manipulation, Outcome, Space};

use manipulation::{Constraint, Field, FieldKind};

use std::sync::{Arc, Mutex};

use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::resource::{NodeId, SubGeometryKind};
use duck_engine_scene::Scene;
use duck_engine_viewer::common::{
    centroid_of_slice, decompose_matrix, Axis, InnerSpace, LengthDisplay, Matrix4, Point3,
    Quaternion, Real, SquareMatrix, Transform, Vector3,
};
use duck_engine_viewer::event::{DeviceEvent, Event, EventContext};
use duck_engine_viewer::operator::{Handle, HandleEvent, SelectionKinds, SelectionMode};
use duck_engine_viewer::selection::{SelectionItem, SelectionManager};

use crate::document::{dvec3_to_point3, dvec3_to_vec3, Document, PartId};
use crate::notifications::Notifications;
use crate::preview::PreviewSession;
use crate::tools::{selected_parts, Gesture, ModelingTool, PanelContext, ToolInfo, Workspace};
use crate::ui::{icons, length_value};

/// CAD-aware transform tool for one operation (move, rotate, *or* scale). Each
/// [`TransformMode`] is registered as its own palette tool.
///
/// What it moves follows the selection:
///
/// - **Whole parts** preview by moving their nodes, and commit by baking the
///   delta into their B-Reps via [`Document::bake_transform`].
/// - **A face**, drilled into as the primary selection, previews as a ghost of
///   the face, and commits by re-solving the body around the moved face via
///   [`Document::tweak_faces`]. A face only scales uniformly.
///
/// Selecting the tool shows its gizmo; drag a grip to transform. The mode's key
/// (G/R/S) starts a free transform at the cursor and X/Y/Z start or change an
/// axis-constrained one (Shift+X/Y/Z a plane). Typing a number sets the amount
/// exactly — `R X 90 ↵` turns a quarter about X — with Tab stepping between the
/// values of a free or plane move; the panel shows the values and edits them
/// too. Left-click or Enter confirms, right-click or Escape cancels. The tool
/// stays for the next transform until Escape with none in progress, or another
/// tool.
///
/// The gizmo runs along the part's own frame, or the world's when the tool's
/// panel switches it; each activation starts on the part's.
pub struct TransformTool {
    manipulation: Manipulation,
    parts: Parts,
    face: FaceTweak,
    /// What a transform would start on now, and about which frame. Refreshed
    /// while idle, so the gizmo follows the selection.
    offer: Option<(Route, Frame)>,
    /// What the transform in progress locked onto.
    active: Option<Route>,
    workspace: Workspace,
    /// Set by Escape with no transform in progress, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}

/// What a transform moves.
#[derive(Clone, PartialEq)]
enum Route {
    /// These whole parts.
    Parts(Vec<NodeId>),
    /// One face, re-solving its body.
    Face(FaceTarget),
}

impl TransformTool {
    pub fn new(mode: TransformMode, workspace: &Workspace) -> Self {
        Self {
            manipulation: Manipulation::new(mode),
            parts: Parts::default(),
            face: FaceTweak::new(Arc::clone(&workspace.document)),
            offer: None,
            active: None,
            workspace: workspace.clone(),
            finished: false,
        }
    }

    /// What the selection offers a transform: a face drilled into, else the
    /// selected parts.
    fn resolve(&mut self, selection: &SelectionManager) -> Option<(Route, Frame)> {
        let document = self.workspace.document.lock().unwrap();
        if let Some(target) = selected_face(selection, &document) {
            // Resolving the face locks the document itself.
            drop(document);
            return self.face.frame_for(target).map(|frame| (Route::Face(target), frame));
        }
        let nodes = selected_parts(selection, &document);
        let frame = parts_frame(&nodes, selection.primary().map(|item| item.node_id()), &document)?;
        Some((Route::Parts(nodes), frame))
    }

    /// The frame a transform would start about now.
    fn frame(&self) -> Option<Frame> {
        self.offer.as_ref().map(|(_, frame)| *frame)
    }

    fn scene(&self) -> Scene {
        self.workspace.document.lock().unwrap().scene().clone()
    }

    /// Carries out what the transform asked. Returns whether the event is
    /// taken.
    fn act(&mut self, outcome: Outcome, selection: &mut SelectionManager) -> bool {
        match outcome {
            Outcome::Ignored => return false,
            Outcome::Consumed => {}
            // A transform started by typing has already moved.
            Outcome::Started => match self.begin() {
                true => self.preview(),
                false => self.manipulation.abort(),
            },
            Outcome::Changed => self.preview(),
            Outcome::Confirmed(delta) => self.commit(delta, selection),
            Outcome::Cancelled => self.cancel(),
        }
        true
    }

    /// Locks onto what was on offer when the transform started.
    fn begin(&mut self) -> bool {
        let Some((route, _)) = self.offer.clone() else { return false };
        let begun = match &route {
            Route::Parts(nodes) => self.parts.begin(nodes, &self.scene()),
            Route::Face(target) => self.face.begin(*target, &self.workspace.geometry_options()),
        };
        self.active = begun.then_some(route);
        begun
    }

    fn preview(&mut self) {
        let delta = self.manipulation.delta();
        match &self.active {
            Some(Route::Parts(_)) => self.parts.preview(&delta, &self.scene()),
            Some(Route::Face(_)) => self.face.preview(&delta),
            None => {}
        }
    }

    fn commit(&mut self, delta: Matrix4, selection: &mut SelectionManager) {
        match self.active.take() {
            Some(Route::Parts(_)) => self.parts.commit(delta, &self.workspace),
            Some(Route::Face(target)) => {
                self.face.commit(target, delta, selection, &self.workspace.notifications)
            }
            None => {}
        }
    }

    fn cancel(&mut self) {
        match self.active.take() {
            Some(Route::Parts(_)) => self.parts.restore(&self.scene()),
            Some(Route::Face(_)) => self.face.cancel(),
            None => {}
        }
    }
}

impl ModelingTool for TransformTool {
    fn info(&self) -> ToolInfo {
        let shortcut = Some(self.manipulation.mode().key());
        match self.manipulation.mode() {
            TransformMode::Translate => ToolInfo { id: "move", icon: icons::MOVE, shortcut },
            TransformMode::Rotate => ToolInfo { id: "rotate", icon: icons::ROTATE, shortcut },
            TransformMode::Scale => ToolInfo { id: "scale", icon: icons::SCALE, shortcut },
        }
    }

    /// Whole parts first; a second click drills into a face for tweaking.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::Progressive(SelectionKinds::FACE)
    }

    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        // Keep what a transform would start on current while idle: the gizmo
        // sits on it, and a key starts on it.
        let refresh = matches!(
            event,
            Event::Device(DeviceEvent::Update { .. } | DeviceEvent::KeyboardInput { .. })
        );
        if refresh && !self.manipulation.is_active() {
            self.offer = self.resolve(ctx.selection);
        }

        let units = self.workspace.display_units();
        let outcome = self.manipulation.dispatch(event, self.frame(), &Input::of(ctx, units));
        if self.act(outcome, ctx.selection) {
            return true;
        }
        // Escape cancels a transform in progress, above; with none, it leaves.
        let escape = Gesture::read(event, ctx.modifiers) == Some(Gesture::Cancel);
        if escape && !self.manipulation.is_active() {
            self.finished = true;
            return true;
        }
        false
    }

    fn handles(&self) -> Vec<Handle> {
        self.manipulation.handles(self.offer.as_ref().map(|(_, frame)| frame))
    }

    fn on_handle(&mut self, event: &HandleEvent, ctx: &mut EventContext) {
        let units = self.workspace.display_units();
        let outcome = self.manipulation.on_handle(event, self.frame(), &Input::of(ctx, units));
        self.act(outcome, ctx.selection);
    }

    fn panel_title(&self) -> Option<&str> {
        match self.manipulation.mode() {
            TransformMode::Translate => Some("Move"),
            TransformMode::Rotate => Some("Rotate"),
            TransformMode::Scale => Some("Scale"),
        }
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, _panel: &mut PanelContext) {
        let before = self.manipulation.orientation();
        let mut orientation = before;
        ui.label("Axes");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut orientation, Space::Local, "Local");
            ui.selectable_value(&mut orientation, Space::World, "Global");
        });
        self.manipulation.set_orientation(orientation);
        // A free move's values run along the axes, so new axes move it.
        let turned = orientation != before;
        if values_ui(ui, &mut self.manipulation, self.workspace.display_units()) || turned {
            self.preview();
        }
    }

    fn activate(&mut self) {
        self.manipulation.set_orientation(Space::Local);
    }

    fn deactivate(&mut self) {
        // Put back whatever a transform in progress previewed.
        self.parts.restore(&self.scene());
        self.face.abort();
        self.manipulation.abort();
        self.active = None;
        self.offer = None;
        self.finished = false;
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

/// The values of `manipulation`'s transform in progress, for its tool's panel:
/// what it is confined to, and a field for each value, which edits it — the
/// one keys are typing into outlined. Shows nothing while idle. Returns whether
/// a value was edited.
pub(super) fn values_ui(
    ui: &mut egui::Ui,
    manipulation: &mut Manipulation,
    units: LengthDisplay,
) -> bool {
    let fields = manipulation.fields();
    if fields.is_empty() {
        return false;
    }
    ui.separator();
    ui.label(describe(manipulation));
    let mut edited = false;
    egui::Grid::new("transform_values").num_columns(2).show(ui, |ui| {
        for (index, field) in fields.iter().enumerate() {
            match field.kind {
                FieldKind::Length => {
                    let axis = field.axis.expect("a length runs along an axis");
                    let rgb = axis.color();
                    let color = egui::Rgba::from_rgb(rgb.r, rgb.g, rgb.b);
                    ui.colored_label(color, axis_name(axis))
                }
                FieldKind::Angle => ui.label("Angle"),
                FieldKind::Factor => ui.label("Factor"),
            };
            if let Some(value) = value_field(ui, field, units) {
                manipulation.set_field(index, value);
                edited = true;
            }
            ui.end_row();
        }
    });
    ui.weak("Type a value; Tab for the next");
    edited
}

/// The field editing one value, a length shown in `units`. Returns the new
/// value, as applied, when it was edited.
fn value_field(ui: &mut egui::Ui, field: &Field, units: LengthDisplay) -> Option<Real> {
    let mut shown = match field.kind {
        FieldKind::Angle => field.value.to_degrees(),
        _ => field.value,
    };
    let drag = match field.kind {
        FieldKind::Length => length_value(&mut shown, units),
        FieldKind::Angle => egui::DragValue::new(&mut shown).speed(0.5).suffix("°"),
        FieldKind::Factor => egui::DragValue::new(&mut shown).speed(0.01).suffix("×"),
    };
    // Shown as typed, so a value part-way through — "-", "1." — reads as such.
    let drag = match field.typed.clone() {
        Some(text) => drag.custom_formatter(move |_, _| text.clone()),
        None => drag,
    };
    let stroke = match field.active {
        true => ui.visuals().selection.stroke,
        false => egui::Stroke::NONE,
    };
    let changed = egui::Frame::NONE.stroke(stroke).show(ui, |ui| ui.add(drag).changed()).inner;
    changed.then(|| match field.kind {
        FieldKind::Angle => shown.to_radians(),
        _ => shown,
    })
}

/// What the transform in progress is confined to, as its values are headed.
fn describe(manipulation: &Manipulation) -> String {
    let space = |space| match space {
        Space::Local => "local",
        Space::World => "global",
    };
    match (manipulation.mode(), manipulation.constraint()) {
        (TransformMode::Translate, Constraint::Free) => {
            format!("Free, along {} axes", space(manipulation.orientation()))
        }
        (TransformMode::Rotate, Constraint::Free) => "About the view".to_owned(),
        (TransformMode::Scale, Constraint::Free) => "Uniform".to_owned(),
        (TransformMode::Rotate, Constraint::Axis(axis, s) | Constraint::Plane(axis, s)) => {
            format!("About {} {}", space(s), axis_name(axis))
        }
        (_, Constraint::Axis(axis, s)) => format!("Along {} {}", space(s), axis_name(axis)),
        (_, Constraint::Plane(normal, s)) => {
            let names: String =
                Axis::ALL.into_iter().filter(|&axis| axis != normal).map(axis_name).collect();
            format!("In the {} {names} plane", space(s))
        }
    }
}

fn axis_name(axis: Axis) -> &'static str {
    match axis {
        Axis::X => "X",
        Axis::Y => "Y",
        Axis::Z => "Z",
    }
}

/// Where a transform of the parts at `nodes` acts: about the centre of their
/// bounds, along the `primary` part's own axes, or the first's when the
/// primary is none of them.
pub(super) fn parts_frame(nodes: &[NodeId], primary: Option<NodeId>, document: &Document) -> Option<Frame> {
    let pivot = {
        let scene = document.scene().lock();
        let centres: Vec<Point3> = nodes
            .iter()
            .filter_map(|&node| {
                scene.nodes_bounding(node).bounds.map(|aabb| aabb.center()).or_else(|| {
                    scene.nodes_transform(node).map(|m| decompose_matrix(&m).position)
                })
            })
            .collect();
        centroid_of_slice(&centres)?
    };
    let axes = primary.filter(|primary| nodes.contains(primary)).or(nodes.first().copied())?;
    let rotation = document.part_at(axes)?.frame().rotation;
    Some(Frame { pivot, rotation, uniform_scale: false })
}

/// The face drilled into as the primary selection, if it belongs to a part.
fn selected_face(selection: &SelectionManager, document: &Document) -> Option<FaceTarget> {
    let SelectionItem::SubGeometry { node_id, element } = selection.primary()? else {
        return None;
    };
    if element.kind != SubGeometryKind::Face {
        return None;
    }
    let part = document.part_for_node(node_id)?;
    Some(FaceTarget { node: node_id, part, face_index: element.index })
}

/// Whole parts: previewed on their nodes, and baked into their B-Reps on
/// commit.
#[derive(Default)]
struct Parts {
    rest: Vec<Rest>,
}

/// Where a moved node rested.
struct Rest {
    node: NodeId,
    local: Transform,
    world: Matrix4,
    /// Takes a world transform into the node's parent's space.
    to_parent: Matrix4,
}

impl Parts {
    /// Snapshots the nodes at `nodes`. Fails with none to move.
    fn begin(&mut self, nodes: &[NodeId], scene: &Scene) -> bool {
        let scene = scene.lock();
        self.rest = nodes
            .iter()
            .filter_map(|&node| {
                let parent = scene.get_node(node)?.parent().and_then(|p| scene.nodes_transform(p));
                Some(Rest {
                    node,
                    local: scene.get_node(node)?.transform(),
                    world: scene.nodes_transform(node)?,
                    to_parent: parent.and_then(|m| m.invert()).unwrap_or(Matrix4::identity()),
                })
            })
            .collect();
        !self.rest.is_empty()
    }

    /// Shows the nodes moved by `delta`.
    fn preview(&self, delta: &Matrix4, scene: &Scene) {
        let mut scene = scene.lock();
        for rest in &self.rest {
            if scene.has_node(rest.node) {
                let local = rest.to_parent * delta * rest.world;
                scene.set_node_transform(rest.node, decompose_matrix(&local));
            }
        }
    }

    /// Puts the nodes back where they rested.
    fn restore(&mut self, scene: &Scene) {
        let mut scene = scene.lock();
        for rest in self.rest.drain(..) {
            if scene.has_node(rest.node) {
                scene.set_node_transform(rest.node, rest.local);
            }
        }
    }

    /// Bakes `delta` into each part as one undo step. The delta itself, not
    /// what the preview could show of it: a stretch across a turned part is
    /// beyond a node's transform, but not the B-Rep's.
    fn commit(&mut self, delta: Matrix4, workspace: &Workspace) {
        let rest = std::mem::take(&mut self.rest);
        let mut document = workspace.document.lock().unwrap();
        let scene = document.scene().clone();
        let mut document = document.undo_scope("Transform");
        for rest in rest {
            let Some(part) = document.part_for_node(rest.node) else { continue };
            if let Err(e) = document.bake_transform(part, delta) {
                scene.set_node_transform(rest.node, rest.local);
                workspace.notifications.failure("Transform", &e);
            }
        }
    }
}

/// The picked face, identified the way the selection system reports it: by
/// tessellation order within a part's mesh.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceTarget {
    node: NodeId,
    part: PartId,
    face_index: u32,
}

/// One face of a solid, moved by re-solving the body around it.
///
/// The drag is previewed cheaply by rigidly moving a ghost tessellation of just
/// the face — no OCCT work per frame. Committing runs the actual tweak
/// ([`Document::tweak_faces`]), which re-solves the body around the moved face
/// and re-tessellates in place; a tweak the kernel cannot re-solve leaves the
/// part untouched and reports the error.
struct FaceTweak {
    /// Ghost of the picked face shown while dragging.
    preview: PreviewSession,
    /// Pivot and axes resolved from the last-targeted face (`None` when it has
    /// no usable pivot), cached because the gizmo follows it every update.
    resolved: Option<(FaceTarget, Option<(Point3, Quaternion)>)>,
    document: Arc<Mutex<Document>>,
}

impl FaceTweak {
    fn new(document: Arc<Mutex<Document>>) -> Self {
        Self { preview: PreviewSession::new(Arc::clone(&document)), resolved: None, document }
    }

    /// Pivot (face centroid) and axes (local Z = outward face normal) for
    /// `target`, resolved from the B-Rep and cached.
    fn resolve(&mut self, target: FaceTarget) -> Option<(Point3, Quaternion)> {
        if let Some((cached, resolved)) = self.resolved
            && cached == target
        {
            return resolved;
        }
        let resolved = (|| {
            let doc = self.document.lock().unwrap();
            let face = doc.part_at(target.node)?.face(target.face_index).ok()?;
            // `normal_at_center` goes through `BRepGProp_Face::Normal`, which
            // applies the face's orientation: it already points out.
            let n = match face.normal_at_center() {
                Ok(n) => n,
                Err(e) => {
                    log::warn!("Tweak target has no usable pivot: {e}");
                    return None;
                }
            };
            let normal = dvec3_to_vec3(n).normalize();
            let pivot = dvec3_to_point3(face.center_of_mass());
            Some((pivot, Quaternion::from_arc(Vector3::unit_z(), normal, None)))
        })();
        self.resolved = Some((target, resolved));
        resolved
    }

    /// The frame a tweak of `target` acts about, if the face resolves.
    fn frame_for(&mut self, target: FaceTarget) -> Option<Frame> {
        let (pivot, rotation) = self.resolve(target)?;
        // A non-uniform scale is no similarity, so the body can't be re-solved
        // around it.
        Some(Frame { pivot, rotation, uniform_scale: true })
    }

    /// Locks onto `target`: ghosts the face for the upcoming drag. Fails if the
    /// face can't be resolved or ghosted.
    fn begin(&mut self, target: FaceTarget, options: &CadTessellationOptions) -> bool {
        if self.resolve(target).is_none() {
            log::warn!("Tweak target could not be resolved to a CAD face");
            return false;
        }
        let ghost_shape = {
            let doc = self.document.lock().unwrap();
            doc.part_at(target.node)
                .and_then(|part| part.face(target.face_index).ok())
                .map(opencascade::primitives::Shape::from)
        };
        let Some(ghost_shape) = ghost_shape else { return false };
        if self.preview.add_preview_from_shape(&ghost_shape, options, "Tweak preview").is_none() {
            log::warn!("Tweak preview could not be tessellated");
            return false;
        }
        true
    }

    /// Moves the ghost by `delta`.
    fn preview(&mut self, delta: &Matrix4) {
        self.preview.set_preview_transform(decompose_matrix(delta));
    }

    /// Runs the tweak on the B-Rep. On success the selection is cleared — the
    /// part re-tessellates, so face indices no longer identify the same
    /// sub-geometry.
    fn commit(
        &mut self,
        target: FaceTarget,
        delta: Matrix4,
        selection: &mut SelectionManager,
        notifications: &Notifications,
    ) {
        self.preview.cancel();
        self.resolved = None;

        let result = self.document.lock().unwrap().tweak_faces(target.part, &[target.face_index], delta);
        match result {
            Ok(()) => selection.clear(),
            Err(e) => notifications.failure("Face tweak", &e),
        }
    }

    /// Discards the drag without touching the B-Rep.
    fn cancel(&mut self) {
        self.preview.cancel();
    }

    /// Drops everything held for the current face. Idempotent.
    fn abort(&mut self) {
        self.preview.cancel();
        self.resolved = None;
    }
}

#[cfg(test)]
mod tests {
    use duck_engine_viewer::common::{
        consts, quaternion_from_axis_angle_safe, EuclideanSpace, LengthUnit,
    };
    use duck_engine_viewer::input::{KeyEvent, Modifiers, NamedKey};
    use glam::DVec3;

    use super::*;
    use crate::testing::{
        face_item_along, key, named_key, volume, workspace_with_box, workspace_with_box_and_cube,
        ViewState,
    };

    const EPSILON: Real = 1e-6;

    fn node_position(ws: &Workspace, node: NodeId) -> Point3 {
        let scene = ws.document.lock().unwrap().scene().clone();
        let position = scene.lock().get_node(node).unwrap().transform().position;
        position
    }

    fn frame_of(ws: &Workspace, node: NodeId) -> Transform {
        let document = ws.document.lock().unwrap();
        document.part_at(node).unwrap().frame()
    }

    fn press(event: KeyEvent) -> Event {
        Event::Device(DeviceEvent::KeyboardInput { event, is_synthetic: false })
    }

    fn select(items: &[SelectionItem]) -> SelectionManager {
        let mut selection = SelectionManager::new();
        selection.extend(items.iter().copied());
        selection
    }

    #[test]
    fn a_part_frame_sits_on_the_bounds_centre_along_the_primary() {
        let (ws, main, other) = workspace_with_box_and_cube();
        let document = ws.document.lock().unwrap();
        // The box spans -1..1 and the cube 3..4, so their centres are the origin
        // and (3.5, 3.5, 3.5).
        let frame = parts_frame(&[main, other], Some(other), &document).unwrap();
        assert!((frame.pivot - Point3::new(1.75, 1.75, 1.75)).magnitude() < EPSILON, "{:?}", frame.pivot);
        assert!(!frame.uniform_scale);
    }

    /// A face drilled into routes to a tweak about the face; the part itself to
    /// a whole-part move.
    #[test]
    fn the_selection_picks_the_route() {
        let (ws, node) = workspace_with_box();
        let mut tool = TransformTool::new(TransformMode::Translate, &ws);

        let face = select(&[face_item_along(&ws, node, DVec3::Y)]);
        let (route, frame) = tool.resolve(&face).expect("a face offers a tweak");
        assert!(matches!(route, Route::Face(_)));
        assert!(frame.uniform_scale);

        let part = select(&[SelectionItem::Node(node)]);
        let (route, _) = tool.resolve(&part).expect("a part offers a move");
        assert!(route == Route::Parts(vec![node]));
    }

    #[test]
    fn a_preview_moves_the_nodes_and_a_restore_puts_them_back() {
        let (ws, node) = workspace_with_box();
        let scene = ws.document.lock().unwrap().scene().clone();
        let mut parts = Parts::default();
        assert!(parts.begin(&[node], &scene));

        parts.preview(&Matrix4::from_translation(Vector3::new(4.0, 0.0, 0.0)), &scene);
        assert!((node_position(&ws, node) - Point3::new(4.0, 0.0, 0.0)).magnitude() < EPSILON);
        parts.restore(&scene);
        assert!((node_position(&ws, node) - Point3::new(0.0, 0.0, 0.0)).magnitude() < EPSILON);
    }

    /// The bake is exact: a quarter turn turns the part's frame an exact quarter.
    #[test]
    fn a_commit_bakes_the_delta_into_the_part() {
        let (ws, node) = workspace_with_box();
        let scene = ws.document.lock().unwrap().scene().clone();
        let mut parts = Parts::default();
        parts.begin(&[node], &scene);

        let turn = quaternion_from_axis_angle_safe(Vector3::unit_z(), consts::PI / 2.0);
        let delta = Matrix4::from_translation(Vector3::new(0.0, 0.0, 5.0)) * Matrix4::from(turn);
        parts.preview(&delta, &scene);
        parts.commit(delta, &ws);

        let frame = frame_of(&ws, node);
        assert!((frame.position - Point3::new(0.0, 0.0, 5.0)).magnitude() < EPSILON, "{frame:?}");
        let x = frame.rotation * Vector3::unit_x();
        assert!((x - Vector3::unit_y()).magnitude() < EPSILON, "{x:?}");
        // The node rests on the new frame.
        assert!((node_position(&ws, node) - frame.position).magnitude() < EPSILON);
    }

    #[test]
    fn a_scale_commit_reshapes_the_part() {
        let (ws, node) = workspace_with_box();
        let scene = ws.document.lock().unwrap().scene().clone();
        let mut parts = Parts::default();
        parts.begin(&[node], &scene);
        parts.commit(Matrix4::from_scale(2.0), &ws);
        assert!((volume(&ws, node) - 64.0).abs() < 1e-6, "{}", volume(&ws, node));
    }

    #[test]
    fn a_multi_part_commit_undoes_as_one_step() {
        let (ws, main, other) = workspace_with_box_and_cube();
        let scene = ws.document.lock().unwrap().scene().clone();
        let mut parts = Parts::default();
        parts.begin(&[main, other], &scene);
        parts.commit(Matrix4::from_translation(Vector3::new(1.0, 0.0, 0.0)), &ws);

        ws.document.lock().unwrap().undo().unwrap();
        for node in [main, other] {
            assert!(frame_of(&ws, node).position.x.abs() < EPSILON);
        }
    }

    /// Reversed faces included: local Z is the outward normal of every face.
    #[test]
    fn a_face_frame_faces_out_of_the_body() {
        let (ws, node) = workspace_with_box();
        let mut face = FaceTweak::new(Arc::clone(&ws.document));
        let part = ws.document.lock().unwrap().part_for_node(node).unwrap();
        for face_index in 0..6 {
            let frame = face.frame_for(FaceTarget { node, part, face_index }).unwrap();
            let out = frame.axis(Axis::Z, Space::Local);
            // The box is centred on the origin, so out is away from it.
            let along = out.dot(frame.pivot.to_vec());
            assert!((along - 1.0).abs() < EPSILON, "face {face_index}: {out:?} at {:?}", frame.pivot);
        }
    }

    #[test]
    fn a_face_commit_re_solves_the_body() {
        let (ws, node) = workspace_with_box();
        let selection = select(&[face_item_along(&ws, node, DVec3::X)]);
        let target = selected_face(&selection, &ws.document.lock().unwrap()).unwrap();
        let mut face = FaceTweak::new(Arc::clone(&ws.document));
        let frame = face.frame_for(target).unwrap();
        assert!(frame.uniform_scale);
        assert!((frame.pivot - Point3::new(1.0, 0.0, 0.0)).magnitude() < EPSILON, "{:?}", frame.pivot);

        assert!(face.begin(target, &ws.geometry_options()));
        let mut selection = selection;
        face.commit(target, Matrix4::from_translation(Vector3::new(1.0, 0.0, 0.0)), &mut selection, &ws.notifications);
        // The 2-unit box grows a unit along X.
        assert!((volume(&ws, node) - 12.0).abs() < 1e-6, "{}", volume(&ws, node));
        assert!(selection.primary().is_none());
    }

    /// The whole way through the tool: R, then X 9 0 ↵, turns the part an exact
    /// quarter about its X.
    #[test]
    fn typing_turns_a_part_an_exact_quarter() {
        let (ws, node) = workspace_with_box();
        let mut tool = TransformTool::new(TransformMode::Rotate, &ws);
        tool.activate();
        let mut view = ViewState::default();
        view.selection.extend([SelectionItem::Node(node)]);

        let update = Event::Device(DeviceEvent::Update { delta_time: 0.016 });
        assert!(!tool.dispatch(&update, &mut view.ctx()));
        for event in [key('x'), key('9'), key('0'), named_key(NamedKey::Enter)] {
            assert!(tool.dispatch(&press(event), &mut view.ctx()));
        }

        let y = frame_of(&ws, node).rotation * Vector3::unit_y();
        assert!((y - Vector3::unit_z()).magnitude() < EPSILON, "{y:?}");
    }

    /// What `values_ui` paints for `manipulation`.
    fn painted(manipulation: &mut Manipulation, units: LengthDisplay) -> Vec<String> {
        let ctx = egui::Context::default();
        let mut output = None;
        // A grid measures its columns in a first pass.
        for _ in 0..2 {
            output = Some(ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| values_ui(ui, manipulation, units));
            }));
        }
        output
            .unwrap()
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::epaint::Shape::Text(text) => Some(text.galley.text().to_owned()),
                _ => None,
            })
            .collect()
    }

    /// Part-way through a number, the field reads as typed.
    #[test]
    fn the_panel_heads_the_values_and_shows_them_as_typed() {
        let view = ViewState::default();
        let units = LengthDisplay::new(LengthUnit::Millimeter, 2);
        let modifiers = Modifiers::default();
        let input = Input { camera: &view.camera, size: (800, 600), cursor: None, modifiers, units };
        let pivot = Point3::new(0.0, 0.0, 0.0);
        let frame = Frame { pivot, rotation: Transform::IDENTITY.rotation, uniform_scale: false };
        let mut manipulation = Manipulation::new(TransformMode::Translate);
        manipulation.set_orientation(Space::World);
        for c in ['x', '1', '2', '.'] {
            manipulation.dispatch(&press(key(c)), Some(frame), &input);
        }

        let painted = painted(&mut manipulation, units);
        for expected in ["Along global X", "X", "12. mm"] {
            assert!(painted.iter().any(|text| text == expected), "no {expected:?} in {painted:?}");
        }
    }

    #[test]
    fn an_idle_transform_shows_no_values() {
        let mut manipulation = Manipulation::new(TransformMode::Rotate);
        let painted = painted(&mut manipulation, LengthDisplay::new(LengthUnit::Millimeter, 2));
        assert!(painted.is_empty(), "{painted:?}");
    }
}
