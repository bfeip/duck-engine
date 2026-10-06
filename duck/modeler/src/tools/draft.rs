use anyhow::{Context, Result};
use duck_engine_common::{consts, Real};
use duck_engine_scene::resource::{NodeId, SubGeometryKind};
use duck_engine_viewer::{
    operator::{
        DragKind, Handle, HandleDrag, HandleId, HandleReach, HandleShape, SelectionKinds,
        SelectionMode,
    },
    selection::SelectionManager,
};
use opencascade::primitives::Shape;

use crate::document::{Document, SourceFate};
use crate::ops::draft::{build_draft, DraftFrame, DraftParams, DraftTarget};
use crate::tools::ToolInfo;
use crate::ui::icons;
use super::targeted::{
    count_summary, selected_on_part, EditLock, PreviewStyle, TargetedOp, TargetedTool,
};
use super::tweak::{angle_field, TweakParams};
use crate::construction::ConstructionOptions;

/// The draft's grip.
const ANGLE_HANDLE: HandleId = HandleId(0);

/// Steepest draft either way: short of the right angle at which the faces
/// would lie along the neutral plane.
const MAX_DRAFT: Real = 85.0 * consts::PI / 180.0;

impl TweakParams for DraftParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        angle_field(ui, "Angle", &mut self.angle, MAX_DRAFT)
    }

    /// A ball on the middle of the grip's face, tied by a leader to the hinge
    /// it turns about.
    fn handles(&self) -> Vec<Handle> {
        vec![
            Handle::new(ANGLE_HANDLE, HandleShape::Ball, self.grip())
                .with_direction(self.frame.axis)
                .with_drag(DragKind::Plane)
                .with_reach(HandleReach::Leader(self.frame.hinge)),
        ]
    }

    /// Swinging the grip about the hinge turns the face with it, one for one.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        if drag.id == ANGLE_HANDLE {
            let swept = drag.angle_about(grabbed.frame.hinge, grabbed.frame.axis);
            self.angle = (grabbed.angle + swept).clamp(-MAX_DRAFT, MAX_DRAFT);
        }
    }
}

/// Tilts selected faces of a part about where they cross a neutral face.
pub type DraftTool = TargetedTool<Draft>;

/// The draft operation. The primary selection is the neutral face and the
/// other selected faces on its part are drafted. Editing locks the part,
/// though faces can still be shift-clicked in and out.
#[derive(Default)]
pub struct Draft;

impl TargetedOp for Draft {
    type Target = DraftTarget;
    type Params = DraftParams;

    const LOCK: EditLock = EditLock::Part;

    fn info(&self) -> ToolInfo {
        ToolInfo { id: "draft", icon: icons::DRAFT, shortcut: None }
    }

    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::SubGeometry(SelectionKinds::FACE)
    }

    fn title(&self, _params: Option<&DraftParams>) -> &'static str {
        "Draft"
    }

    fn prompt(&self, selection: &SelectionManager) -> &'static str {
        if selection.is_empty() {
            "Select the neutral face, then shift-click faces to draft."
        } else {
            "Shift-click faces to draft."
        }
    }

    fn node(&self, target: &DraftTarget) -> NodeId {
        target.node
    }

    fn select(
        &self,
        selection: &SelectionManager,
        locked: Option<&DraftTarget>,
    ) -> (Option<DraftTarget>, usize) {
        let locked = locked.map(|target| target.node);
        let (selected, ignored) = selected_on_part(selection, SubGeometryKind::Face, locked);
        let target = selected.and_then(|(node, faces)| match faces.as_slice() {
            [neutral, faces @ ..] if !faces.is_empty() => {
                Some(DraftTarget { node, neutral: *neutral, faces: faces.to_vec() })
            }
            _ => None,
        });
        (target, ignored)
    }

    fn summary(&self, target: &DraftTarget, ignored: usize) -> Option<String> {
        Some(count_summary("face", target.faces.len(), ignored))
    }

    /// The grip rides on the first face to draft; an edit keeps its angle.
    fn resolve(
        &self,
        doc: &Document,
        target: &DraftTarget,
        _construction: &ConstructionOptions,
        edited: Option<&DraftParams>,
    ) -> Result<DraftParams> {
        let frame = DraftFrame::new(doc, target)?;
        Ok(match edited {
            Some(params) => DraftParams { frame, ..*params },
            None => DraftParams::new(frame),
        })
    }

    fn is_degenerate(&self, params: &DraftParams) -> bool {
        params.is_degenerate()
    }

    fn preview_style(&self, _params: &DraftParams) -> PreviewStyle {
        PreviewStyle::InPlace
    }

    fn build(&self, doc: &Document, target: &DraftTarget, params: &DraftParams) -> Result<Shape> {
        build_draft(doc, target, params)
    }

    fn apply(
        &self,
        doc: &mut Document,
        target: &DraftTarget,
        params: &DraftParams,
        construction: &ConstructionOptions,
    ) -> Result<()> {
        let shape = build_draft(doc, target, params)?;
        let part = doc.part_for_node(target.node).context("Draft target is not a known CAD part")?;
        doc.commit_result(part, shape, SourceFate::Reshape, "Draft", "Draft", &construction.geometry_options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_common::{InnerSpace, Matrix3, Point3, Rad, Vector3};
    use duck_engine_scene::resource::Visibility;
    use duck_engine_viewer::input::Modifiers;
    use duck_engine_viewer::operator::HandleEvent;
    use duck_engine_viewer::selection::SelectionItem;
    use glam::DVec3;

    use crate::testing::{face_item_along, visibility, volume, workspace_with_box};
    use crate::tools::targeted::Phase;
    use crate::tools::{ModelingTool, Workspace};

    const EPSILON: Real = 1e-5;

    fn index(item: SelectionItem) -> u32 {
        match item {
            SelectionItem::SubGeometry { element, .. } => element.index,
            SelectionItem::Node(_) => panic!("not a face"),
        }
    }

    /// A tool drafting the box's +X wall off its floor.
    fn targeting_wall(ws: &Workspace, node: NodeId) -> (DraftTool, SelectionManager) {
        let mut op = DraftTool::new(ws);
        let mut selection = SelectionManager::new();
        selection.set(face_item_along(ws, node, DVec3::NEG_Y));
        selection.add(face_item_along(ws, node, DVec3::X));
        op.follow_selection(&selection);
        (op, selection)
    }

    fn params(op: &DraftTool) -> DraftParams {
        *op.phase.params().expect("a draft is targeted")
    }

    /// A drag that swings the grip of `grabbed` by `angle` about its hinge.
    fn swing(grabbed: &DraftParams, angle: Real) -> HandleDrag {
        let frame = grabbed.frame;
        let point = frame.hinge + Matrix3::from_axis_angle(frame.axis, Rad(angle)) * frame.lever;
        HandleDrag { id: ANGLE_HANDLE, grab: grabbed.grip(), point, modifiers: Modifiers::default() }
    }

    /// Grabs the grip and swings it by `angle`, then lets go.
    fn drag_by(op: &mut DraftTool, angle: Real) {
        let grabbed = params(op);
        op.on_handle(&HandleEvent::Begin(ANGLE_HANDLE));
        op.on_handle(&HandleEvent::Drag(swing(&grabbed, angle)));
        op.on_handle(&HandleEvent::End(ANGLE_HANDLE));
    }

    #[test]
    fn the_grip_sits_on_the_face_with_a_leader_to_its_hinge() {
        let (ws, node) = workspace_with_box();
        let (op, _) = targeting_wall(&ws, node);

        let [grip] = <[Handle; 1]>::try_from(params(&op).handles()).expect("one grip");
        assert!((grip.anchor - Point3::new(1.0, 0.0, 0.0)).magnitude() < EPSILON);
        assert_eq!(grip.drag, DragKind::Plane);
        assert!((grip.direction - Vector3::unit_z()).magnitude() < EPSILON);
        let HandleReach::Leader(hinge) = grip.reach else { panic!("the grip has no leader") };
        assert!((hinge - Point3::new(1.0, -1.0, 0.0)).magnitude() < EPSILON);
    }

    /// Swung in over the part, the grip leans the wall in: a positive draft,
    /// by the angle swept.
    #[test]
    fn swinging_the_grip_in_over_the_part_drafts_by_the_angle_swept() {
        let (ws, node) = workspace_with_box();
        let (op, _) = targeting_wall(&ws, node);
        let grabbed = params(&op);

        let drag = swing(&grabbed, Real::to_radians(10.0));
        assert!(drag.point.x < 1.0, "the swing leans the wall in");
        let mut edited = grabbed;
        edited.apply_handle(&drag, &grabbed);
        assert!((edited.angle - Real::to_radians(10.0)).abs() < EPSILON, "got {}°", edited.angle.to_degrees());
        assert!((edited.grip() - drag.point).magnitude() < EPSILON, "the grip follows the cursor");
    }

    /// A drag carries its total offset from the grab, so applying successive
    /// reports to the same snapshot must not compound them.
    #[test]
    fn successive_drags_from_one_grab_do_not_compound() {
        let (ws, node) = workspace_with_box();
        let (op, _) = targeting_wall(&ws, node);
        let grabbed = params(&op);
        let step = swing(&grabbed, Real::to_radians(5.0));

        let mut edited = grabbed;
        edited.apply_handle(&step, &grabbed);
        edited.apply_handle(&step, &grabbed);
        assert!((edited.angle - Real::to_radians(5.0)).abs() < EPSILON);
    }

    #[test]
    fn the_draft_is_held_short_of_flat() {
        let (ws, node) = workspace_with_box();
        let (op, _) = targeting_wall(&ws, node);
        let grabbed = params(&op);

        let mut edited = grabbed;
        edited.apply_handle(&swing(&grabbed, Real::to_radians(-120.0)), &grabbed);
        assert_eq!(edited.angle, -MAX_DRAFT);
    }

    /// The first face picked is the neutral one; until another is added there
    /// is nothing to draft.
    #[test]
    fn the_primary_face_is_neutral_and_the_rest_are_drafted() {
        let (ws, node) = workspace_with_box();
        let mut op = DraftTool::new(&ws);
        let mut selection = SelectionManager::new();
        assert_eq!(Draft.prompt(&selection), "Select the neutral face, then shift-click faces to draft.");

        let floor = face_item_along(&ws, node, DVec3::NEG_Y);
        selection.set(floor);
        op.follow_selection(&selection);
        assert!(matches!(op.phase, Phase::AwaitingSelection));
        assert_eq!(Draft.prompt(&selection), "Shift-click faces to draft.");

        let walls = [face_item_along(&ws, node, DVec3::X), face_item_along(&ws, node, DVec3::Z)];
        selection.extend(walls);
        op.follow_selection(&selection);
        let target = op.phase.target().expect("a draft is targeted");
        assert_eq!(target.neutral, index(floor));
        assert_eq!(target.faces, walls.map(index));
    }

    /// Faces shift-clicked in mid-edit are drafted at the angle already set.
    #[test]
    fn a_face_added_mid_edit_takes_the_angle_already_set() {
        let (ws, node) = workspace_with_box();
        let (mut op, mut selection) = targeting_wall(&ws, node);
        drag_by(&mut op, Real::to_radians(5.0));
        assert!(op.is_editing());
        assert!(!op.swallows_click(Modifiers { shift: true, ..Default::default() }));

        selection.add(face_item_along(&ws, node, DVec3::NEG_X));
        op.follow_selection(&selection);
        assert!(op.is_editing());
        assert_eq!(op.phase.target().expect("still targeted").faces.len(), 2);
        assert!((params(&op).angle - Real::to_radians(5.0)).abs() < EPSILON);
    }

    #[test]
    fn a_grip_drag_previews_the_draft_in_place_and_apply_reshapes_the_part() {
        let (ws, node) = workspace_with_box();
        let (mut op, mut selection) = targeting_wall(&ws, node);
        let angle = Real::to_radians(10.0);

        drag_by(&mut op, angle);
        op.refresh_preview();
        assert!(op.error.is_none(), "{:?}", op.error);
        assert_eq!(op.preview.preview_nodes().len(), 1);
        assert_eq!(visibility(&ws, node), Visibility::Invisible);

        op.apply(&mut selection).expect("the draft applies");
        assert!(op.is_finished());
        assert!(selection.is_empty(), "the part's faces were renumbered");
        assert!(op.preview.is_empty());
        assert_eq!(visibility(&ws, node), Visibility::Visible);
        let expected = 8.0 - 4.0 * f64::from(angle).tan();
        assert!((volume(&ws, node) - expected).abs() < 1e-6, "got {}", volume(&ws, node));
        assert_eq!(ws.document.lock().unwrap().undo_label(), Some("Draft"));
    }

    #[test]
    fn cancel_restores_the_part() {
        let (ws, node) = workspace_with_box();
        let (mut op, _) = targeting_wall(&ws, node);
        drag_by(&mut op, Real::to_radians(10.0));
        op.refresh_preview();

        op.cancel();
        assert!(op.is_finished());
        assert!(op.preview.is_empty());
        assert_eq!(visibility(&ws, node), Visibility::Visible);
        assert!((volume(&ws, node) - 8.0).abs() < 1e-9);
    }
}
