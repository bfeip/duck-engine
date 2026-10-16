use anyhow::Result;
use duck_engine_common::{consts, InnerSpace, LengthDisplay, Plane, Point3, Real, Vector3};
use duck_engine_scene::resource::{NodeId, SubGeometryKind};
use duck_engine_viewer::{
    operator::{
        DragKind, Handle, HandleDrag, HandleId, HandleReach, HandleShape, SelectionKinds,
        SelectionMode,
    },
    selection::{SelectionItem, SelectionManager},
};
use opencascade::primitives::Shape;

use crate::construction::ConstructionOptions;
use crate::document::{Document, SourceFate};
use crate::ops::revolve::{
    build_revolve, spans_axis, Axis, Profile, RevolveFrame, RevolveParams, RevolveTarget,
};
use crate::tools::{Pointer, ToolInfo};
use crate::ui::icons;
use super::edit::{angle_field, Params, MIN_DIMENSION};
use super::feature::{EditLock, Feature, FeatureTool};
use super::targets::{count_summary, selected_on_part, Selected};

/// The revolve's grip.
const ANGLE_HANDLE: HandleId = HandleId(0);

impl Params for RevolveParams {
    fn ui(&mut self, ui: &mut egui::Ui, _units: LengthDisplay) -> bool {
        let mut changed = angle_field(ui, "Angle", &mut self.angle, consts::TAU);

        ui.label("Axis");
        ui.horizontal(|ui| match self.axis {
            Axis::Unpicked => {
                ui.label("Click its start");
            }
            Axis::From(_) => {
                ui.label("Click its end");
            }
            Axis::Through(..) => {
                ui.label("Picked");
                if ui.small_button("Re-pick (A)").clicked() {
                    self.axis = Axis::Unpicked;
                    changed = true;
                }
            }
        });
        ui.end_row();

        changed
    }

    /// A ball where the sweep ends, tied by a leader to the axis it turns
    /// about. Shown at any angle once there is an axis, so a sweep dragged to
    /// nothing can be dragged back.
    fn handles(&self) -> Vec<Handle> {
        let Some(lever) = Lever::of(self) else { return Vec::new() };
        vec![
            Handle::new(ANGLE_HANDLE, HandleShape::Ball, lever.swung(self.angle))
                .with_direction(lever.axis)
                .with_drag(DragKind::Plane)
                .with_reach(HandleReach::Leader(lever.pivot)),
        ]
    }

    /// Swinging the ball about the axis sweeps the profile with it, one for
    /// one.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        if drag.id != ANGLE_HANDLE {
            return;
        }
        let Some(lever) = Lever::of(grabbed) else { return };
        // The swing wraps at half a turn either way: take the turn nearest the
        // sweep so far, so a drag carries on round.
        let swept = drag.angle_about(lever.pivot, lever.axis);
        let so_far = self.angle - grabbed.angle;
        let swept = swept + consts::TAU * ((so_far - swept) / consts::TAU).round();
        self.angle = (grabbed.angle + swept).clamp(-consts::TAU, consts::TAU);
    }

    fn is_degenerate(&self) -> bool {
        RevolveParams::is_degenerate(self)
    }
}

/// The arm the angle grip swings on: from its pivot on the axis, square to the
/// axis, out to the lever.
struct Lever {
    pivot: Point3,
    axis: Vector3,
    arm: Vector3,
}

impl Lever {
    /// The lever of `params`, once they have an axis. A lever on the axis
    /// itself swings at the profile's size instead, along some square
    /// direction.
    fn of(params: &RevolveParams) -> Option<Self> {
        let (origin, axis) = params.line()?;
        let lever = params.frame.lever;
        let pivot = origin + axis * (lever - origin).dot(axis);
        let arm = lever - pivot;
        let arm = if arm.magnitude() > MIN_DIMENSION {
            arm
        } else {
            Plane::from_point(axis, pivot).basis().0 * params.frame.size.max(MIN_DIMENSION)
        };
        Some(Self { pivot, axis, arm })
    }

    /// The end of the arm, swung `angle` counter-clockwise looking down the
    /// axis.
    fn swung(&self, angle: Real) -> Point3 {
        self.pivot + self.arm * angle.cos() + self.axis.cross(self.arm) * angle.sin()
    }
}

/// Revolves the selected curves or faces about an axis picked with two
/// clicks.
pub type RevolveTool = FeatureTool<Revolve>;

/// The revolve operation, on the faces or edges of the primary's kind on one
/// part. Once the profile is designated, plain clicks pick the axis; editing
/// locks the part, though its faces or edges can still be shift-clicked in and
/// out.
#[derive(Default)]
pub struct Revolve;

impl Feature for Revolve {
    type Target = RevolveTarget;
    type Params = RevolveParams;

    const TOOL: ToolInfo = ToolInfo { id: "revolve", icon: icons::REVOLVE, shortcut: None };
    /// A face revolves into a solid, an edge into a sheet.
    const SELECTS: SelectionMode = SelectionMode::SubGeometry(SelectionKinds::FACE.union(SelectionKinds::EDGE));
    const LOCK: EditLock = EditLock::Part;

    /// The faces, if the primary is a face, else the edges, on one part.
    fn select(&self, selection: &SelectionManager, locked: Option<NodeId>) -> Option<Selected<RevolveTarget>> {
        let kind = match selection.primary() {
            Some(SelectionItem::SubGeometry { element, .. }) if element.kind == SubGeometryKind::Face => {
                SubGeometryKind::Face
            }
            _ => SubGeometryKind::Edge,
        };
        let selected = selected_on_part(selection, kind, locked)?;
        Some(selected.map(|node, indices| {
            let profile = match kind {
                SubGeometryKind::Face => Profile::Faces(indices),
                _ => Profile::Edges(indices),
            };
            RevolveTarget { node, profile }
        }))
    }

    /// An edit keeps its axis and sweep.
    fn resolve(
        &self,
        doc: &Document,
        target: &RevolveTarget,
        _construction: &ConstructionOptions,
        edited: Option<&RevolveParams>,
    ) -> Result<RevolveParams> {
        let frame = RevolveFrame::new(doc, target)?;
        Ok(match edited {
            Some(params) => RevolveParams { frame, ..*params },
            None => RevolveParams::new(frame),
        })
    }

    fn build(&self, doc: &Document, target: &RevolveTarget, params: &RevolveParams) -> Result<Shape> {
        build_revolve(doc, target, params)
    }

    /// The frame's: a revolve replaces a bare sketch, fuses into the solid
    /// whose face it sweeps, and stands beside the solid whose edge it sweeps.
    fn fate(&self, params: &RevolveParams) -> SourceFate {
        params.frame.fate
    }

    fn title(&self, _params: Option<&RevolveParams>) -> &'static str {
        "Revolve"
    }

    fn new_part_name(&self) -> &'static str {
        "Revolution"
    }

    fn prompt(&self, _selection: &SelectionManager) -> &'static str {
        "Select a curve or face to revolve."
    }

    fn summary(&self, target: &RevolveTarget, ignored: usize) -> Option<String> {
        let noun = match target.profile {
            Profile::Faces(_) => "face",
            Profile::Edges(_) => "edge",
        };
        Some(count_summary(noun, target.profile.count(), ignored))
    }

    /// A re-picks the axis.
    fn on_char(&self, c: char, params: &RevolveParams) -> Option<RevolveParams> {
        (c == 'a' && params.axis != Axis::Unpicked).then_some(RevolveParams { axis: Axis::Unpicked, ..*params })
    }

    /// Until both ends of the axis are picked.
    fn picking(&self, params: &RevolveParams) -> bool {
        !matches!(params.axis, Axis::Through(..))
    }

    /// The axis's start, then its end, at the snapped point.
    fn pick(&self, params: &RevolveParams, pointer: &Pointer) -> Option<RevolveParams> {
        let at = pointer.snap?.position;
        let axis = match params.axis {
            Axis::Unpicked => Axis::From(at),
            Axis::From(start) if spans_axis(start, at) => Axis::Through(start, at),
            Axis::From(_) | Axis::Through(..) => return None,
        };
        Some(RevolveParams { axis, ..*params })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::resource::Visibility;
    use duck_engine_viewer::input::Modifiers;
    use duck_engine_viewer::operator::HandleEvent;
    use opencascade::primitives::ShapeType;

    use crate::testing::{
        face_item, pointer_at, visibility, workspace_with, workspace_with_box, xy_region,
    };
    use crate::tools::{Gesture, ModelingTool, Workspace};

    const EPSILON: Real = 1e-6;

    fn point(x: Real, y: Real) -> Point3 {
        Point3::new(x, y, 0.0)
    }

    /// A tool targeting a 1 × 2 region standing on the origin, its left edge
    /// on the Y axis.
    fn tool() -> (RevolveTool, Workspace, NodeId, SelectionManager) {
        let (ws, node) = workspace_with(xy_region(&[(0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (0.0, 2.0)]));
        let mut op = RevolveTool::new(&ws);
        let mut selection = SelectionManager::new();
        selection.set(face_item(node, 0));
        op.on_gesture(Gesture::Frame, &mut selection);
        (op, ws, node, selection)
    }

    /// Picks the region's left edge as the axis, pointing up.
    fn pick_axis(op: &mut RevolveTool) {
        op.take_pick(&pointer_at(point(0.0, 0.0)));
        op.take_pick(&pointer_at(point(0.0, 2.0)));
    }

    /// A full turn about the Y axis of the region `tool` targets.
    fn full_turn() -> RevolveParams {
        let (op, ..) = tool();
        RevolveParams { axis: Axis::Through(point(0.0, 0.0), point(0.0, 1.0)), ..*op.params().expect("targeted") }
    }

    /// A drag swinging the grip from where it sits for `from` to where it
    /// would sit for `to`.
    fn swing(params: &RevolveParams, from: Real, to: Real) -> HandleDrag {
        let lever = Lever::of(params).expect("the axis is picked");
        HandleDrag { id: ANGLE_HANDLE, grab: lever.swung(from), point: lever.swung(to), modifiers: Modifiers::default() }
    }

    #[test]
    fn a_designated_profile_waits_for_its_axis() {
        let (op, ..) = tool();
        assert!(op.is_picking());
        assert!(!op.is_editing());
        let shift = Modifiers { shift: true, ..Modifiers::default() };
        assert!(!op.swallows_click(shift), "shift-clicks still refine the profile");
    }

    #[test]
    fn two_picks_make_the_axis_and_the_first_locks_the_target() {
        let (mut op, ..) = tool();
        op.take_pick(&pointer_at(point(0.0, 0.0)));
        assert!(op.is_editing());
        assert!(op.is_picking());

        // The same point again spans no axis, and is not taken.
        op.take_pick(&pointer_at(point(0.0, 0.0)));
        assert!(op.is_picking());

        op.take_pick(&pointer_at(point(0.0, 2.0)));
        assert!(!op.is_picking());
        assert_eq!(op.params().expect("editing").axis, Axis::Through(point(0.0, 0.0), point(0.0, 2.0)));
    }

    #[test]
    fn a_pick_needs_a_snap() {
        let (mut op, ..) = tool();
        let unsnapped = Pointer { snap: None, ..pointer_at(point(0.0, 0.0)) };
        op.take_pick(&unsnapped);
        assert_eq!(op.params().expect("targeted").axis, Axis::Unpicked);
        assert!(!op.is_editing());
    }

    /// The hovered axis previews while the sketch stays snappable; once picked,
    /// the revolution takes the sketch's place.
    #[test]
    fn the_sketch_stays_visible_until_the_axis_is_picked() {
        let (mut op, ws, node, mut selection) = tool();
        op.take_pick(&pointer_at(point(0.0, 0.0)));
        op.hover(&pointer_at(point(0.0, 2.0)));
        op.on_gesture(Gesture::Frame, &mut selection);
        assert!(op.error().is_none(), "{:?}", op.error());
        assert_eq!(op.preview().preview_nodes().len(), 1);
        assert_eq!(visibility(&ws, node), Visibility::Visible);

        op.take_pick(&pointer_at(point(0.0, 2.0)));
        op.on_gesture(Gesture::Frame, &mut selection);
        assert_eq!(op.preview().preview_nodes().len(), 1);
        assert_eq!(visibility(&ws, node), Visibility::Invisible);
    }

    #[test]
    fn the_cursor_shows_only_while_picking() {
        let (mut op, ..) = tool();
        op.hover(&pointer_at(point(0.0, 0.0)));
        assert_eq!(op.cursor_target(), Some(point(0.0, 0.0)));

        pick_axis(&mut op);
        assert_eq!(op.cursor_target(), None);
    }

    /// Finishing halfway through the axis drops it and stays in the tool.
    #[test]
    fn finishing_mid_axis_starts_over() {
        let (mut op, _ws, _node, mut selection) = tool();
        op.take_pick(&pointer_at(point(0.0, 0.0)));

        assert!(op.on_gesture(Gesture::Finish, &mut selection));
        op.on_gesture(Gesture::Frame, &mut selection);
        assert!(!op.is_finished());
        assert!(!op.is_editing());
        assert_eq!(op.params().expect("still targeted").axis, Axis::Unpicked);
        assert!(op.preview().is_empty());
    }

    /// With nothing picked, there is nothing of the tool's to finish.
    #[test]
    fn finishing_before_any_pick_is_left_to_others() {
        let (mut op, _ws, _node, mut selection) = tool();
        assert!(!op.on_gesture(Gesture::Finish, &mut selection));
        assert!(!op.is_finished());
    }

    #[test]
    fn finishing_with_the_axis_picked_revolves_the_region_into_a_solid() {
        let (mut op, ws, _node, mut selection) = tool();
        pick_axis(&mut op);
        op.on_gesture(Gesture::Frame, &mut selection);

        assert!(op.on_gesture(Gesture::Finish, &mut selection));
        assert!(op.is_finished());
        let doc = ws.document.lock().unwrap();
        let shapes: Vec<_> = doc.parts().map(|part| part.shape()).collect();
        assert_eq!(shapes.len(), 1, "the revolution replaces the region");
        assert!(shapes[0].contains_type(ShapeType::Solid));
        assert!((shapes[0].volume() - 2.0 * std::f64::consts::PI).abs() < 1e-6);
        assert_eq!(doc.undo_label(), Some("Revolve"));
    }

    #[test]
    fn a_repicks_the_axis_and_keeps_the_sweep() {
        let (mut op, _ws, _node, mut selection) = tool();
        pick_axis(&mut op);
        let params = *op.params().expect("editing");
        op.on_handle(&HandleEvent::Begin(ANGLE_HANDLE));
        op.on_handle(&HandleEvent::Drag(swing(&params, consts::TAU, consts::TAU - consts::FRAC_PI_2)));
        op.on_handle(&HandleEvent::End(ANGLE_HANDLE));

        assert!(op.on_gesture(Gesture::Key('a'), &mut selection));
        assert!(op.is_picking());
        let params = op.params().expect("editing");
        assert_eq!(params.axis, Axis::Unpicked);
        assert!((params.angle - 1.5 * consts::PI).abs() < EPSILON);
    }

    #[test]
    fn the_grip_appears_once_the_axis_is_picked() {
        let (op, ..) = tool();
        assert!(op.handles().is_empty());

        let params = full_turn();
        let [grip] = <[Handle; 1]>::try_from(params.handles()).expect("one grip");
        // A full turn brings the lever back where it started, on the profile.
        assert!((grip.anchor - params.frame.lever).magnitude() < EPSILON);
        let HandleReach::Leader(pivot) = grip.reach else { panic!("the grip has no leader") };
        assert!(pivot.x.abs() < EPSILON && pivot.z.abs() < EPSILON, "the leader runs to the axis");
    }

    #[test]
    fn swinging_the_grip_back_a_quarter_turn_leaves_three_quarters() {
        let grabbed = full_turn();
        let mut params = grabbed;
        params.apply_handle(&swing(&grabbed, consts::TAU, consts::TAU - consts::FRAC_PI_2), &grabbed);
        assert!((params.angle - 1.5 * consts::PI).abs() < EPSILON, "{}", params.angle.to_degrees());
    }

    /// The swing wraps at half a turn, so the grip must carry the sweep on
    /// round rather than jump back.
    #[test]
    fn the_grip_carries_on_past_half_a_turn() {
        let grabbed = full_turn();
        let mut params = grabbed;
        for degrees in [90.0, 170.0, 190.0, 270.0] {
            let to = consts::TAU - Real::to_radians(degrees);
            params.apply_handle(&swing(&grabbed, consts::TAU, to), &grabbed);
        }
        assert!((params.angle - consts::FRAC_PI_2).abs() < EPSILON, "{}", params.angle.to_degrees());
    }

    #[test]
    fn the_sweep_is_held_within_a_turn_either_way() {
        let grabbed = full_turn();
        let mut params = grabbed;
        params.apply_handle(&swing(&grabbed, consts::TAU, consts::TAU + consts::FRAC_PI_2), &grabbed);
        assert_eq!(params.angle, consts::TAU);

        let grabbed = RevolveParams { angle: consts::PI, ..full_turn() };
        let mut params = grabbed;
        for degrees in (1..=6).map(|step| step as Real * 90.0) {
            let to = consts::PI - Real::to_radians(degrees);
            params.apply_handle(&swing(&grabbed, consts::PI, to), &grabbed);
        }
        assert_eq!(params.angle, -consts::TAU);
    }

    /// A lever on the axis has no arm of its own, so the grip swings at the
    /// profile's size instead.
    #[test]
    fn a_lever_on_the_axis_still_has_an_arm() {
        let params = RevolveParams {
            axis: Axis::Through(point(0.5, 0.0), point(0.5, 1.0)),
            ..full_turn()
        };
        let params = RevolveParams { frame: RevolveFrame { lever: point(0.5, 1.0), ..params.frame }, ..params };
        let lever = Lever::of(&params).expect("the axis is picked");
        assert!((lever.arm.magnitude() - params.frame.size).abs() < EPSILON);
        assert!(lever.arm.dot(lever.axis).abs() < EPSILON);
    }

    /// Shift-clicking more faces into a locked revolve keeps its axis and
    /// sweep.
    #[test]
    fn refining_the_profile_keeps_the_axis_and_sweep() {
        let (ws, node) = workspace_with_box();
        let mut op = RevolveTool::new(&ws);
        let mut selection = SelectionManager::new();
        selection.set(face_item(node, 0));
        op.on_gesture(Gesture::Frame, &mut selection);
        op.take_pick(&pointer_at(point(5.0, 0.0)));
        op.take_pick(&pointer_at(point(5.0, 1.0)));
        let picked = *op.params().expect("editing");

        selection.add(face_item(node, 1));
        op.on_gesture(Gesture::Frame, &mut selection);
        assert_eq!(*op.target().expect("still targeted"), RevolveTarget { node, profile: Profile::Faces(vec![0, 1]) });
        let params = op.params().expect("still editing");
        assert_eq!(params.axis, picked.axis);
        assert_eq!(params.angle, picked.angle);
    }
}
