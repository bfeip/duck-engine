use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_common::{consts, InnerSpace, Plane, Point3, Real, Vector3};
use duck_engine_scene::resource::SubGeometryKind;
use duck_engine_viewer::{
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, KeyEvent, MouseButton, NamedKey},
    operator::{
        DragKind, Handle, HandleDrag, HandleEvent, HandleId, HandleReach, HandleShape, Operator,
        SelectionKinds, SelectionMode,
    },
    selection::{SelectionItem, SelectionManager},
};

use crate::document::Document;
use crate::extrude::{
    build_extrusion, execute_extrude, ExtrudeFrame, ExtrudeParams, ExtrudeTarget, SourceFate,
};
use crate::notifications::Notifications;
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, PanelContext, ToolInfo};
use crate::ui::icons;
use super::tweak::{
    angle_field, handle_tweak, length_field, tweak_panel, TweakAction, TweakParams,
};
use super::ConstructionOptions;

/// The extrusion's grips.
const DISTANCE_HANDLE: HandleId = HandleId(0);
const DRAFT_HANDLE: HandleId = HandleId(1);
const DIRECTION_HANDLE: HandleId = HandleId(2);
const THICKNESS_HANDLE: HandleId = HandleId(3);

/// Steepest draft either way: short of the right angle at which the walls
/// would lie flat.
const MAX_DRAFT: Real = 85.0 * consts::PI / 180.0;

/// Furthest the direction may lean from the profile normal: short of the right
/// angle at which the sweep would lie in the profile's own plane.
const MAX_TILT: Real = 85.0 * consts::PI / 180.0;

/// A tilt below this reads as straight out of the profile.
const TILT_EPSILON: Real = 1e-4;

enum Phase {
    /// No face or edge selected.
    AwaitingSelection,
    /// A face or edge is selected but nothing has been edited, so the target
    /// still follows the selection.
    Targeted(ExtrudeParams),
    /// Editing has begun: the target is locked until the extrusion is applied
    /// or cancelled.
    Editing(ExtrudeParams),
}

impl Phase {
    fn params(&self) -> Option<&ExtrudeParams> {
        match self {
            Phase::AwaitingSelection => None,
            Phase::Targeted(params) | Phase::Editing(params) => Some(params),
        }
    }
}

impl TweakParams for ExtrudeParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let min_distance = self.min_distance();
        let mut changed = length_field(ui, "Distance", &mut self.distance, min_distance..=Real::MAX);
        changed |= angle_field(ui, "Draft", &mut self.draft, MAX_DRAFT);
        changed |= length_field(ui, "Thickness", &mut self.thickness, Real::MIN..=Real::MAX);

        ui.label("Direction");
        ui.horizontal(|ui| {
            let tilt = self.tilt();
            if tilt < TILT_EPSILON {
                ui.label("Normal");
            } else {
                ui.label(format!("{:.1}° from normal", tilt.to_degrees()));
                if ui.small_button("Reset").clicked() {
                    self.direction = self.frame.normal;
                    changed = true;
                }
            }
        });
        ui.end_row();

        changed
    }

    /// An arrow at the tip, tied back to the origin by a leader that reads as
    /// the extrusion's axis. Once there is a length to shape, a ring around the
    /// tip sets the draft, a quad halfway up the axis tilts it, and a grip on
    /// an arm across the base sets the wall thickness.
    fn handles(&self) -> Vec<Handle> {
        let origin = self.frame.origin;
        // The arrow points the way the extrusion grows, which a negative
        // distance reverses.
        let growth = self.direction * self.distance.signum();
        let mut handles = vec![
            Handle::new(DISTANCE_HANDLE, HandleShape::Cone, self.tip())
                .with_direction(growth)
                .with_reach(HandleReach::Leader(origin)),
        ];
        if !self.is_degenerate() {
            let across = thickness_axis(self);
            handles.extend([
                Handle::new(DRAFT_HANDLE, HandleShape::Ring, self.tip())
                    .with_direction(self.direction)
                    .with_drag(DragKind::Plane),
                Handle::new(DIRECTION_HANDLE, HandleShape::Quad, midpoint(self))
                    .with_direction(self.direction)
                    .with_drag(DragKind::Plane),
                Handle::new(THICKNESS_HANDLE, HandleShape::Cube, origin + across * self.thickness)
                    .with_direction(across)
                    .with_reach(HandleReach::Arm),
            ]);
        }
        handles
    }

    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        match drag.id {
            // The tip follows the cursor along the axis, one for one.
            DISTANCE_HANDLE => {
                let distance = grabbed.distance + drag.distance_along(grabbed.direction);
                self.distance = distance.max(grabbed.min_distance());
            }
            // Turning the ring about the axis adds the swept angle to the draft.
            DRAFT_HANDLE => {
                let swept = drag.angle_about(grabbed.tip(), grabbed.direction);
                self.draft = (grabbed.draft + swept).clamp(-MAX_DRAFT, MAX_DRAFT);
            }
            // The thickness follows its grip across the profile, one for one.
            THICKNESS_HANDLE => {
                self.thickness = grabbed.thickness + drag.distance_along(thickness_axis(grabbed));
            }
            // Swing the axis through the dragged midpoint, so the quad stays
            // under the cursor. The length is untouched.
            DIRECTION_HANDLE if !grabbed.is_degenerate() => {
                let lean = midpoint(grabbed) + drag.delta() - grabbed.frame.origin;
                let direction = lean.normalize() * grabbed.distance.signum();
                self.direction = limit_tilt(direction, grabbed.frame.normal);
            }
            _ => {}
        }
    }
}

/// The middle of the extrusion's axis, where the direction grip rides.
fn midpoint(params: &ExtrudeParams) -> Point3 {
    params.frame.origin + params.direction * (0.5 * params.distance)
}

/// The axis in the profile's plane the thickness grip slides along.
fn thickness_axis(params: &ExtrudeParams) -> Vector3 {
    Plane::from_point(params.frame.normal, params.frame.origin).basis().0
}

/// `direction`, swung back toward `normal` if it leans further than
/// [`MAX_TILT`] from it.
fn limit_tilt(direction: Vector3, normal: Vector3) -> Vector3 {
    let along = direction.dot(normal);
    if along >= MAX_TILT.cos() {
        return direction;
    }
    let across = direction - normal * along;
    if across.magnitude2() < Real::EPSILON {
        // Straight back through the profile: there is no side to lean to.
        return normal;
    }
    normal * MAX_TILT.cos() + across.normalize() * MAX_TILT.sin()
}

pub struct ExtrudeOperator {
    phase: Phase,
    /// The primary selection the tool last acted on, so it reacts only when
    /// the selection changes.
    followed: Option<ExtrudeTarget>,
    /// Parameters as they were when the held grip was grabbed; `None` when no
    /// grip is held. A drag reports its total offset, so it is applied to this
    /// rather than to the live parameters.
    grabbed: Option<ExtrudeParams>,
    /// The parameters the preview was last built for.
    built: Option<ExtrudeParams>,
    /// Why the current parameters could not be built, shown in the panel.
    error: Option<String>,
    /// The previewed extrusion, and the bare sketch it stands in for.
    preview: PreviewSession,
    /// Set once the extrusion is applied or cancelled, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,

    document: Arc<Mutex<Document>>,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    notifications: Notifications,
}

impl ExtrudeOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
        notifications: Notifications,
    ) -> Self {
        let preview = PreviewSession::new(Arc::clone(&document));
        Self {
            phase: Phase::AwaitingSelection,
            followed: None,
            grabbed: None,
            built: None,
            error: None,
            preview,
            finished: false,
            document,
            construction_options,
            notifications,
        }
    }

    fn is_editing(&self) -> bool {
        matches!(self.phase, Phase::Editing(_))
    }

    /// The face/edge currently chosen as the primary selection, if any.
    fn selected_target(selection: &SelectionManager) -> Option<ExtrudeTarget> {
        match selection.primary()? {
            SelectionItem::SubGeometry { node_id, element } => match element.kind {
                SubGeometryKind::Face => {
                    Some(ExtrudeTarget::Face { node: node_id, face_index: element.index })
                }
                SubGeometryKind::Edge => {
                    Some(ExtrudeTarget::Edge { node: node_id, edge_index: element.index })
                }
                SubGeometryKind::Pointset => None,
            },
            SelectionItem::Node(_) => None,
        }
    }

    /// Points an unedited extrusion at the primary selection, once it changes.
    /// Editing locks the target, so while editing this does nothing.
    fn follow_selection(&mut self, selection: &SelectionManager) {
        let target = Self::selected_target(selection);
        if self.is_editing() || target == self.followed {
            return;
        }
        self.followed = target;

        let Some(target) = target else {
            self.phase = Phase::AwaitingSelection;
            return;
        };
        // Edges extrude out of the sketch (construction) plane; faces ignore this
        // and use their own normal.
        let sketch_normal = self.construction_options.borrow().construction_plane.normal;
        let frame = ExtrudeFrame::new(&self.document.lock().unwrap(), target, sketch_normal);
        self.phase = match frame {
            Ok(frame) => Phase::Targeted(ExtrudeParams::new(frame)),
            Err(e) => {
                log::warn!("Extrude target could not be resolved: {e:#}");
                Phase::AwaitingSelection
            }
        };
    }

    /// Rebuilds the preview when the parameters have moved on since it was last
    /// built. A build that fails keeps the last good preview and says why.
    fn refresh_preview(&mut self) {
        let Phase::Editing(params) = self.phase else { return };
        if self.built == Some(params) {
            return;
        }
        self.built = Some(params);

        if params.is_degenerate() {
            // Nothing to show, and a hidden sketch comes back.
            self.preview.clear_previews();
            self.error = None;
            return;
        }

        let shape = match build_extrusion(&self.document.lock().unwrap(), &params) {
            Ok(shape) => shape,
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                return;
            }
        };
        let options = self.construction_options.borrow().preview_options();
        if self.preview.try_replace_preview(&shape, &options, "Extrude preview").is_none() {
            self.error = Some("The extrusion could not be tessellated".to_owned());
            return;
        }
        self.error = None;
        // The extrusion stands in for a bare sketch. A solid source stays in
        // view: the pad overlapping it already looks like the fused result.
        if params.frame.fate == SourceFate::Replace {
            self.preview.hide_source_node(params.frame.target.node());
        }
    }

    /// Commit the extrusion and finish the tool. A failure keeps the preview
    /// and panel so the parameters can be corrected.
    fn apply(&mut self, selection: &mut SelectionManager) -> anyhow::Result<()> {
        let Some(params) = self.phase.params().copied() else { return Ok(()) };
        let options = self.construction_options.borrow().geometry_options.clone();
        execute_extrude(&mut self.document.lock().unwrap(), &params, &options)?;

        // The only source the preview hid was the sketch the extrusion just
        // replaced, and that part is already gone.
        let _ = self.preview.commit();
        // The selected face or edge belonged to a part the extrusion replaced.
        selection.clear();
        self.reset();
        self.finished = true;
        Ok(())
    }

    /// Apply, reporting a failure. For the gestures that keep the tool active and
    /// so must report for themselves: the panel's Apply button, Enter, right-click.
    fn apply_and_report(&mut self, selection: &mut SelectionManager) {
        if let Err(e) = self.apply(selection) {
            log::error!("Extrude failed: {e:#}");
            self.notifications.error(format!("Extrude failed: {e:#}"));
        }
    }

    /// Abandon the extrusion and finish the tool, restoring anything the
    /// preview hid.
    fn cancel(&mut self) {
        self.preview.cancel();
        self.reset();
        self.finished = true;
    }

    /// Forget the extrusion in progress.
    fn reset(&mut self) {
        self.phase = Phase::AwaitingSelection;
        self.grabbed = None;
        self.built = None;
        self.error = None;
    }

    /// Enter applies an extrusion being edited; Escape abandons the tool.
    fn on_key(&mut self, event: &KeyEvent, selection: &mut SelectionManager) -> bool {
        if event.state != ElementState::Pressed || event.repeat {
            return false;
        }
        match event.logical_key {
            Key::Named(NamedKey::Enter) if self.is_editing() => self.apply_and_report(selection),
            Key::Named(NamedKey::Escape) if self.phase.params().is_some() => self.cancel(),
            _ => return false,
        }
        true
    }
}

impl ModelingTool for ExtrudeOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "extrude", icon: icons::EXTRUDE, shortcut: None }
    }

    fn deactivate(&mut self) {
        self.preview.cancel();
        self.reset();
        self.followed = None;
        self.finished = false;
    }

    fn selection_mode(&self) -> SelectionMode {
        // Extrude operates on a face (→ solid) or an edge (→ face).
        SelectionMode::SubGeometry(SelectionKinds::FACE | SelectionKinds::EDGE)
    }

    /// An extrusion being edited is a finished one, so leaving the tool commits
    /// it. A zero-length one is nothing to commit.
    fn finalize(&mut self, selection: &mut SelectionManager) -> anyhow::Result<()> {
        match self.phase {
            Phase::Editing(params) if !params.is_degenerate() => self.apply(selection),
            _ => Ok(()),
        }
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn handles(&self) -> Vec<Handle> {
        self.phase.params().map(TweakParams::handles).unwrap_or_default()
    }

    /// Grabbing a grip begins the edit, which locks the target.
    fn on_handle(&mut self, event: &HandleEvent) {
        if let (HandleEvent::Begin(_), Phase::Targeted(params)) = (event, &self.phase) {
            self.phase = Phase::Editing(*params);
        }
        let Phase::Editing(params) = self.phase else { return };
        if let Some(edited) = handle_tweak(params, &mut self.grabbed, event) {
            self.phase = Phase::Editing(edited);
        }
    }

    fn panel_title(&self) -> Option<&str> {
        Some("Extrude")
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        let Some(mut params) = self.phase.params().copied() else {
            ui.label("Select a face or edge to extrude.");
            return;
        };
        if let Some(error) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        match tweak_panel(ui, &mut params) {
            // An edit begins the operation, which locks the target.
            TweakAction::Changed => self.phase = Phase::Editing(params),
            TweakAction::Apply => self.apply_and_report(panel.selection),
            TweakAction::Cancel => self.cancel(),
            TweakAction::None => {}
        }
    }
}

impl Operator for ExtrudeOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::Update { .. } => {
                self.follow_selection(ctx.selection);
                self.refresh_preview();
                false
            }
            // Once editing, the grips and panel own the extrusion: a stray pick
            // must not reselect anything.
            DeviceEvent::MouseClick { button: MouseButton::Left, .. } => self.is_editing(),
            // Right-click finalizes, matching the Boolean/Line convention.
            DeviceEvent::MouseClick { button: MouseButton::Right, .. } if self.is_editing() => {
                self.apply_and_report(ctx.selection);
                true
            }
            DeviceEvent::KeyboardInput { event, .. } => self.on_key(event, ctx.selection),
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Extrude"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::cad::CadTessellationOptions;
    use duck_engine_scene::resource::{NodeId, SubGeometryElement};
    use duck_engine_scene::Scene;
    use duck_engine_viewer::input::Modifiers;
    use opencascade::primitives::Shape;

    const EPSILON: Real = 1e-5;

    /// A document holding a 2×2×2 box centred on the origin.
    fn document_with_box() -> (Arc<Mutex<Document>>, NodeId) {
        let mut doc = Document::new(Scene::default());
        let part = doc
            .add_part("box", Shape::box_centered(2.0, 2.0, 2.0), &CadTessellationOptions::default())
            .expect("box tessellates");
        let node = doc.node_for_part(part).expect("part has a node");
        (Arc::new(Mutex::new(doc)), node)
    }

    fn operator(document: &Arc<Mutex<Document>>) -> ExtrudeOperator {
        let construction = Rc::new(RefCell::new(ConstructionOptions::new()));
        ExtrudeOperator::new(construction, Arc::clone(document), Notifications::default())
    }

    fn select_face(selection: &mut SelectionManager, node: NodeId, index: u32) {
        selection.set(SelectionItem::SubGeometry {
            node_id: node,
            element: SubGeometryElement::new(SubGeometryKind::Face, index),
        });
    }

    /// A pad of `distance` on the first face of a box.
    fn pad(distance: Real) -> ExtrudeParams {
        let (document, node) = document_with_box();
        let doc = document.lock().unwrap();
        let target = ExtrudeTarget::Face { node, face_index: 0 };
        let frame = ExtrudeFrame::new(&doc, target, Vector3::unit_y()).expect("face resolves");
        ExtrudeParams { distance, ..ExtrudeParams::new(frame) }
    }

    /// A drag of `offset` on the grip `id`, as the handle machinery reports it.
    fn drag(id: HandleId, grab: Point3, offset: Vector3) -> HandleDrag {
        HandleDrag { id, grab, point: grab + offset, modifiers: Modifiers::default() }
    }

    /// Some direction square to the pad's axis.
    fn across(params: &ExtrudeParams) -> Vector3 {
        let normal = params.frame.normal;
        let other = if normal.x.abs() < 0.9 { Vector3::unit_x() } else { Vector3::unit_y() };
        normal.cross(other).normalize()
    }

    #[test]
    fn a_zero_length_extrusion_shows_only_the_distance_arrow() {
        let params = pad(0.0);
        let handles = params.handles();
        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0].id, DISTANCE_HANDLE);
        assert!((handles[0].anchor - params.frame.origin).magnitude() < EPSILON);
    }

    #[test]
    fn the_arrow_sits_on_the_tip_with_a_leader_to_the_origin() {
        let params = pad(3.0);
        let handles = params.handles();
        assert_eq!(handles.len(), 4);

        let arrow = handles.iter().find(|h| h.id == DISTANCE_HANDLE).expect("arrow is present");
        assert!((arrow.anchor - params.tip()).magnitude() < EPSILON);
        assert!((arrow.direction - params.direction).magnitude() < EPSILON);
        let HandleReach::Leader(origin) = arrow.reach else { panic!("the arrow has no leader") };
        assert!((origin - params.frame.origin).magnitude() < EPSILON);

        let quad = handles.iter().find(|h| h.id == DIRECTION_HANDLE).expect("quad is present");
        assert!((quad.anchor - (params.frame.origin + params.direction * 1.5)).magnitude() < EPSILON);
        assert_eq!(quad.drag, DragKind::Plane);
    }

    /// The draft ring circles the tip in the plane square to the axis, and the
    /// thickness grip rides an arm across the base, out by the thickness.
    #[test]
    fn the_shaping_grips_surround_the_axis() {
        let params = ExtrudeParams { thickness: 0.2, ..pad(3.0) };
        let handles = params.handles();

        let ring = handles.iter().find(|h| h.id == DRAFT_HANDLE).expect("ring is present");
        assert!((ring.anchor - params.tip()).magnitude() < EPSILON);
        assert!((ring.direction - params.direction).magnitude() < EPSILON);
        assert_eq!(ring.drag, DragKind::Plane);

        let across = thickness_axis(&params);
        assert!(across.dot(params.frame.normal).abs() < EPSILON, "the grip leaves the profile plane");
        let grip = handles.iter().find(|h| h.id == THICKNESS_HANDLE).expect("grip is present");
        assert!((grip.anchor - (params.frame.origin + across * 0.2)).magnitude() < EPSILON);
        assert!((grip.direction - across).magnitude() < EPSILON);
        assert_eq!(grip.reach, HandleReach::Arm);
    }

    #[test]
    fn turning_the_ring_adds_the_swept_angle_to_the_draft() {
        let grabbed = pad(2.0);
        let (tip, axis) = (grabbed.tip(), grabbed.direction);
        let from = thickness_axis(&grabbed);
        let toward = |angle: Real| {
            let sideways = axis.cross(from);
            tip + from * angle.cos() + sideways * angle.sin()
        };
        let turn = |angle: Real| HandleDrag {
            id: DRAFT_HANDLE,
            grab: tip + from,
            point: toward(angle),
            modifiers: Modifiers::default(),
        };

        let mut params = grabbed;
        params.apply_handle(&turn(Real::to_radians(10.0)), &grabbed);
        assert!((params.draft - Real::to_radians(10.0)).abs() < 1e-4, "got {}", params.draft.to_degrees());

        params.apply_handle(&turn(Real::to_radians(-120.0)), &grabbed);
        assert_eq!(params.draft, -MAX_DRAFT, "the draft is held short of flat");
    }

    #[test]
    fn the_thickness_follows_its_grip_one_for_one() {
        let grabbed = ExtrudeParams { thickness: 0.2, ..pad(2.0) };
        let across = thickness_axis(&grabbed);
        let grab = grabbed.frame.origin + across * 0.2;

        let mut params = grabbed;
        params.apply_handle(&drag(THICKNESS_HANDLE, grab, across * 0.3), &grabbed);
        assert!((params.thickness - 0.5).abs() < EPSILON);

        // Past zero it grows the walls on the other side.
        params.apply_handle(&drag(THICKNESS_HANDLE, grab, across * -0.5), &grabbed);
        assert!((params.thickness + 0.3).abs() < EPSILON);
    }

    #[test]
    fn the_distance_follows_the_arrow_one_for_one() {
        let grabbed = pad(1.0);
        let mut params = grabbed;
        params.apply_handle(&drag(DISTANCE_HANDLE, grabbed.tip(), grabbed.direction * 1.5), &grabbed);
        assert!((params.distance - 2.5).abs() < EPSILON);
    }

    #[test]
    fn a_pad_stops_at_its_face() {
        let grabbed = pad(1.0);
        let mut params = grabbed;
        params.apply_handle(&drag(DISTANCE_HANDLE, grabbed.tip(), grabbed.direction * -5.0), &grabbed);
        assert_eq!(params.distance, 0.0);
    }

    /// Dragging the quad sideways leans the axis toward the drag and leaves the
    /// length alone; the quad lands back under the cursor.
    #[test]
    fn the_quad_leans_the_axis_toward_the_drag() {
        let grabbed = pad(2.0);
        let side = across(&grabbed);
        let mut params = grabbed;
        params.apply_handle(&drag(DIRECTION_HANDLE, midpoint(&grabbed), side * 1.0), &grabbed);

        assert!((params.distance - grabbed.distance).abs() < EPSILON);
        assert!((params.direction.magnitude() - 1.0).abs() < EPSILON);
        // Midpoint one unit up the normal, dragged one unit across: 45°.
        assert!((params.tilt() - consts::FRAC_PI_4).abs() < 1e-4);
        assert!(params.direction.dot(side) > 0.0, "leaned away from the drag");
        let expected = grabbed.frame.origin + (grabbed.frame.normal + side).normalize();
        assert!((midpoint(&params) - expected).magnitude() < 1e-4);
    }

    #[test]
    fn the_lean_is_held_short_of_the_profile_plane() {
        let grabbed = pad(2.0);
        let mut params = grabbed;
        params.apply_handle(&drag(DIRECTION_HANDLE, midpoint(&grabbed), across(&grabbed) * 1e4), &grabbed);
        assert!(params.tilt() <= MAX_TILT + 1e-4);
        assert!(params.direction.dot(params.frame.normal) > 0.0);
    }

    #[test]
    fn a_lean_straight_back_through_the_profile_snaps_to_the_normal() {
        let normal = Vector3::unit_z();
        assert!((limit_tilt(-normal, normal) - normal).magnitude() < EPSILON);
    }

    /// A drag carries its total offset from the grab, so applying successive
    /// reports to the same snapshot must not compound them.
    #[test]
    fn successive_drags_from_one_grab_do_not_compound() {
        let grabbed = pad(1.0);
        let mut params = grabbed;
        let step = drag(DISTANCE_HANDLE, grabbed.tip(), grabbed.direction * 1.0);
        params.apply_handle(&step, &grabbed);
        params.apply_handle(&step, &grabbed);
        assert!((params.distance - 2.0).abs() < EPSILON);
    }

    /// Until something is edited, the extrusion follows the selection.
    #[test]
    fn an_unedited_extrusion_follows_the_selection() {
        let (document, node) = document_with_box();
        let mut op = operator(&document);
        let mut selection = SelectionManager::new();

        select_face(&mut selection, node, 0);
        op.follow_selection(&selection);
        let first = op.phase.params().expect("targeted").frame.target;

        select_face(&mut selection, node, 1);
        op.follow_selection(&selection);
        let second = op.phase.params().expect("still targeted").frame.target;
        assert_ne!(first, second, "the target did not follow the selection");

        selection.clear();
        op.follow_selection(&selection);
        assert!(matches!(op.phase, Phase::AwaitingSelection));
    }

    /// Grabbing a grip begins the edit, after which the selection no longer moves
    /// the target.
    #[test]
    fn grabbing_a_grip_locks_the_target() {
        let (document, node) = document_with_box();
        let mut op = operator(&document);
        let mut selection = SelectionManager::new();
        select_face(&mut selection, node, 0);
        op.follow_selection(&selection);
        assert!(!op.is_editing());

        op.on_handle(&HandleEvent::Begin(DISTANCE_HANDLE));
        assert!(op.is_editing());
        let locked = op.phase.params().expect("editing").frame.target;

        select_face(&mut selection, node, 1);
        op.follow_selection(&selection);
        assert_eq!(op.phase.params().expect("still editing").frame.target, locked);
    }

    /// An arrow dragged and released drives the preview through one rebuild.
    #[test]
    fn a_grip_drag_rebuilds_the_preview() {
        let (document, node) = document_with_box();
        let mut op = operator(&document);
        let mut selection = SelectionManager::new();
        select_face(&mut selection, node, 0);
        op.follow_selection(&selection);

        let tip = op.phase.params().expect("targeted").tip();
        let direction = op.phase.params().expect("targeted").direction;
        op.on_handle(&HandleEvent::Begin(DISTANCE_HANDLE));
        op.on_handle(&HandleEvent::Drag(drag(DISTANCE_HANDLE, tip, direction * 2.0)));
        op.on_handle(&HandleEvent::End(DISTANCE_HANDLE));
        op.refresh_preview();

        assert!((op.phase.params().expect("editing").distance - 2.0).abs() < EPSILON);
        assert!(op.error.is_none(), "{:?}", op.error);
        assert_eq!(op.preview.preview_nodes().len(), 1);
    }
}
