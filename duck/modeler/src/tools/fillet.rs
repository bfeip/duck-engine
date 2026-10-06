use anyhow::{Context, Result};
use duck_engine_scene::resource::{NodeId, SubGeometryKind};
use duck_engine_viewer::{
    operator::{Handle, HandleDrag, HandleId, HandleReach, HandleShape, SelectionKinds, SelectionMode},
    selection::SelectionManager,
};
use duck_engine_viewer::common::Real;
use opencascade::primitives::Shape;

use crate::document::{Document, SourceFate};
use crate::ops::fillet::{build_fillet, BlendKind, FilletFrame, FilletParams, FilletTarget};
use crate::tools::ToolInfo;
use crate::ui::icons;
use super::targeted::{
    count_summary, selected_on_part, EditLock, PreviewStyle, TargetedOp, TargetedTool,
};
use super::tweak::{length_field, TweakParams};
use crate::construction::ConstructionOptions;

/// The blend's grip.
const SIZE_HANDLE: HandleId = HandleId(0);

impl TweakParams for FilletParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Type");
        ui.horizontal(|ui| {
            for kind in [BlendKind::Fillet, BlendKind::Chamfer] {
                changed |= ui.selectable_value(&mut self.kind, kind, kind.name()).changed();
            }
        });
        ui.end_row();

        let label = match self.kind {
            BlendKind::Fillet => "Radius",
            BlendKind::Chamfer => "Distance",
        };
        changed |= length_field(ui, label, &mut self.size, 0.0..=Real::MAX);
        changed
    }

    /// A grip on the primary edge, tied back to it by a leader. It rides out of
    /// the corner as a fillet grows and into the part as a chamfer does, and
    /// points the way it has moved.
    fn handles(&self) -> Vec<Handle> {
        let direction = if self.signed_size() < 0.0 { -self.frame.outward } else { self.frame.outward };
        vec![
            Handle::new(SIZE_HANDLE, HandleShape::Cone, self.grip())
                .with_direction(direction)
                .with_reach(HandleReach::Leader(self.frame.apex)),
        ]
    }

    /// The grip follows the cursor along the corner's bisector, one for one.
    /// Dragged back past the edge, a fillet turns into a chamfer, and back.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        if drag.id == SIZE_HANDLE {
            self.set_signed_size(grabbed.signed_size() + drag.distance_along(grabbed.frame.outward));
        }
    }
}

/// Rounds or bevels selected edges of a part: one tool, whose grip makes a
/// fillet on one side of the edge and a chamfer on the other.
pub type FilletTool = TargetedTool<Fillet>;

/// The fillet/chamfer operation, on the selected edges of one part. Editing
/// locks the part, though its edges can still be shift-clicked in and out.
#[derive(Default)]
pub struct Fillet;

impl TargetedOp for Fillet {
    type Target = FilletTarget;
    type Params = FilletParams;

    const LOCK: EditLock = EditLock::Part;

    fn info(&self) -> ToolInfo {
        ToolInfo { id: "fillet", icon: icons::FILLET, shortcut: Some('f') }
    }

    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::SubGeometry(SelectionKinds::EDGE)
    }

    fn title(&self, params: Option<&FilletParams>) -> &'static str {
        params.map_or(BlendKind::Fillet, |params| params.kind).name()
    }

    fn prompt(&self, _selection: &SelectionManager) -> &'static str {
        "Select edges to fillet or chamfer."
    }

    fn node(&self, target: &FilletTarget) -> NodeId {
        target.node
    }

    fn select(
        &self,
        selection: &SelectionManager,
        locked: Option<&FilletTarget>,
    ) -> (Option<FilletTarget>, usize) {
        let locked = locked.map(|target| target.node);
        let (selected, ignored) = selected_on_part(selection, SubGeometryKind::Edge, locked);
        (selected.map(|(node, edges)| FilletTarget { node, edges }), ignored)
    }

    fn summary(&self, target: &FilletTarget, ignored: usize) -> Option<String> {
        Some(count_summary("edge", target.edges.len(), ignored))
    }

    /// The grip rides on the primary edge; an edit keeps its kind and size.
    fn resolve(
        &self,
        doc: &Document,
        target: &FilletTarget,
        _construction: &ConstructionOptions,
        edited: Option<&FilletParams>,
    ) -> Result<FilletParams> {
        let frame = FilletFrame::new(doc, target)?;
        Ok(match edited {
            Some(params) => FilletParams { frame, ..*params },
            None => FilletParams::new(frame),
        })
    }

    fn is_degenerate(&self, params: &FilletParams) -> bool {
        params.is_degenerate()
    }

    fn preview_style(&self, _params: &FilletParams) -> PreviewStyle {
        PreviewStyle::InPlace
    }

    fn build(&self, doc: &Document, target: &FilletTarget, params: &FilletParams) -> Result<Shape> {
        build_fillet(doc, target, params)
    }

    fn apply(
        &self,
        doc: &mut Document,
        target: &FilletTarget,
        params: &FilletParams,
        construction: &ConstructionOptions,
    ) -> Result<()> {
        let shape = build_fillet(doc, target, params)?;
        let part = doc.part_for_node(target.node).context("Fillet target is not a known CAD part")?;
        let name = params.kind.name();
        doc.commit_result(part, shape, SourceFate::Reshape, name, name, &construction.geometry_options)
    }

    /// F and C switch between fillet and chamfer.
    fn on_char(&self, c: char, params: &FilletParams) -> Option<FilletParams> {
        let kind = match c {
            'f' => BlendKind::Fillet,
            'c' => BlendKind::Chamfer,
            _ => return None,
        };
        Some(FilletParams { kind, ..*params })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::common::{InnerSpace, Point3, Vector3};
    use duck_engine_scene::resource::Visibility;
    use duck_engine_viewer::input::Modifiers;
    use duck_engine_viewer::operator::HandleEvent;

    use crate::testing::{edge_item, key, visibility, volume, workspace_with_box_and_cube};
    use crate::tools::targeted::Phase;
    use crate::tools::{ModelingTool, Workspace};

    const EPSILON: Real = 1e-5;

    /// A tool targeting edge 0 of the box alone.
    fn targeting_edge(ws: &Workspace, node: NodeId) -> (FilletTool, SelectionManager) {
        let mut op = FilletTool::new(ws);
        let mut selection = SelectionManager::new();
        selection.set(edge_item(node, 0));
        op.follow_selection(&selection);
        (op, selection)
    }

    fn params(op: &FilletTool) -> FilletParams {
        *op.phase.params().expect("a blend is targeted")
    }

    fn edges(op: &FilletTool) -> Vec<u32> {
        op.phase.target().expect("a blend is targeted").edges.clone()
    }

    /// A drag of `offset` on the grip, as the handle machinery reports it.
    fn drag(grab: Point3, offset: Vector3) -> HandleDrag {
        crate::testing::drag(SIZE_HANDLE, grab, offset)
    }

    /// Grabs the grip and drags it `distance` out of the corner, then lets go.
    fn drag_out(op: &mut FilletTool, distance: Real) {
        let grabbed = params(op);
        op.on_handle(&HandleEvent::Begin(SIZE_HANDLE));
        op.on_handle(&HandleEvent::Drag(drag(grabbed.grip(), grabbed.frame.outward * distance)));
        op.on_handle(&HandleEvent::End(SIZE_HANDLE));
    }

    #[test]
    fn the_grip_sits_out_of_the_corner_by_the_signed_size() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (op, _) = targeting_edge(&ws, node);
        let fillet = FilletParams { size: 0.5, ..params(&op) };
        let (apex, outward) = (fillet.frame.apex, fillet.frame.outward);

        let [grip] = <[Handle; 1]>::try_from(fillet.handles()).expect("one grip");
        assert!((grip.anchor - (apex + outward * 0.5)).magnitude() < EPSILON);
        assert!((grip.direction - outward).magnitude() < EPSILON);
        let HandleReach::Leader(origin) = grip.reach else { panic!("the grip has no leader") };
        assert!((origin - apex).magnitude() < EPSILON);

        let chamfer = FilletParams { kind: BlendKind::Chamfer, ..fillet };
        let [grip] = <[Handle; 1]>::try_from(chamfer.handles()).expect("one grip");
        assert!((grip.anchor - (apex - outward * 0.5)).magnitude() < EPSILON);
        assert!((grip.direction + outward).magnitude() < EPSILON, "a chamfer's grip points into the part");
    }

    #[test]
    fn dragging_out_of_the_corner_grows_a_fillet_one_for_one() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (op, _) = targeting_edge(&ws, node);
        let grabbed = FilletParams { size: 0.2, ..params(&op) };

        let mut edited = grabbed;
        edited.apply_handle(&drag(grabbed.grip(), grabbed.frame.outward * 0.3), &grabbed);
        assert_eq!(edited.kind, BlendKind::Fillet);
        assert!((edited.size - 0.5).abs() < EPSILON);
    }

    #[test]
    fn dragging_back_past_the_edge_turns_a_fillet_into_a_chamfer() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (op, _) = targeting_edge(&ws, node);
        let grabbed = FilletParams { size: 0.2, ..params(&op) };
        let outward = grabbed.frame.outward;

        let mut edited = grabbed;
        edited.apply_handle(&drag(grabbed.grip(), outward * -0.5), &grabbed);
        assert_eq!(edited.kind, BlendKind::Chamfer);
        assert!((edited.size - 0.3).abs() < EPSILON);

        // And back out again.
        edited.apply_handle(&drag(grabbed.grip(), outward * 0.1), &grabbed);
        assert_eq!(edited.kind, BlendKind::Fillet);
        assert!((edited.size - 0.3).abs() < EPSILON);
    }

    /// A drag carries its total offset from the grab, so applying successive
    /// reports to the same snapshot must not compound them.
    #[test]
    fn successive_drags_from_one_grab_do_not_compound() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (op, _) = targeting_edge(&ws, node);
        let grabbed = FilletParams { size: 0.2, ..params(&op) };
        let step = drag(grabbed.grip(), grabbed.frame.outward * 0.1);

        let mut edited = grabbed;
        edited.apply_handle(&step, &grabbed);
        edited.apply_handle(&step, &grabbed);
        assert!((edited.size - 0.3).abs() < EPSILON);
    }

    /// Until something is edited, the blend follows the selected edges.
    #[test]
    fn an_unedited_blend_follows_the_selection() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, mut selection) = targeting_edge(&ws, node);
        assert_eq!(edges(&op), [0]);

        selection.add(edge_item(node, 5));
        op.follow_selection(&selection);
        assert_eq!(edges(&op), [0, 5], "the primary edge leads");

        selection.set(edge_item(node, 3));
        op.follow_selection(&selection);
        assert_eq!(edges(&op), [3]);

        selection.clear();
        op.follow_selection(&selection);
        assert!(matches!(op.phase, Phase::AwaitingSelection));
    }

    /// One blend works on one part: edges on others are counted, not blended.
    #[test]
    fn edges_on_other_parts_are_left_out() {
        let (ws, main, other) = workspace_with_box_and_cube();
        let (mut op, mut selection) = targeting_edge(&ws, main);

        selection.add(edge_item(other, 0));
        op.follow_selection(&selection);
        assert_eq!(op.phase.target().unwrap().node, main);
        assert_eq!(edges(&op), [0]);
        assert_eq!(op.ignored, 1);
        assert_eq!(count_summary("edge", 1, op.ignored), "1 edge (1 on other parts ignored)");
    }

    /// Grabbing the grip locks the part: a plain click stops at the tool, but a
    /// shift-click still reaches the selection, and the edges it toggles on the
    /// part are blended at the size already set.
    #[test]
    fn grabbing_the_grip_locks_the_part_but_shift_clicks_still_edit_its_edges() {
        let (ws, main, other) = workspace_with_box_and_cube();
        let (mut op, mut selection) = targeting_edge(&ws, main);
        assert!(!op.swallows_click(Modifiers::default()));

        drag_out(&mut op, 0.3);
        assert!(op.is_editing());
        assert!(op.swallows_click(Modifiers::default()));
        assert!(!op.swallows_click(Modifiers { shift: true, ..Default::default() }));

        selection.add(edge_item(main, 5));
        op.follow_selection(&selection);
        assert_eq!(edges(&op), [0, 5]);
        assert!((params(&op).size - 0.3).abs() < EPSILON, "the edit keeps its size");
        assert!(op.is_editing());

        selection.add(edge_item(other, 0));
        op.follow_selection(&selection);
        assert_eq!(op.phase.target().unwrap().node, main, "the part stays locked");
        assert_eq!(edges(&op), [0, 5]);
    }

    #[test]
    fn dropping_the_last_edge_drops_the_blend_and_restores_the_part() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, mut selection) = targeting_edge(&ws, node);
        drag_out(&mut op, 0.3);
        op.refresh_preview();
        assert_eq!(visibility(&ws, node), Visibility::Invisible);

        selection.clear();
        op.follow_selection(&selection);
        assert!(matches!(op.phase, Phase::AwaitingSelection));
        assert!(op.preview.is_empty());
        assert_eq!(visibility(&ws, node), Visibility::Visible);
    }

    /// A grip drag builds the blend and shows it in place of its part.
    #[test]
    fn a_grip_drag_previews_the_blend_in_place_of_its_part() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, _) = targeting_edge(&ws, node);

        drag_out(&mut op, 0.3);
        op.refresh_preview();
        assert!(op.error.is_none(), "{:?}", op.error);
        assert_eq!(op.preview.preview_nodes().len(), 1);
        assert_eq!(visibility(&ws, node), Visibility::Invisible);
    }

    #[test]
    fn an_oversized_blend_keeps_the_last_preview_and_says_why() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, _) = targeting_edge(&ws, node);
        drag_out(&mut op, 0.3);
        op.refresh_preview();
        let shown = op.preview.preview_nodes().to_vec();

        drag_out(&mut op, 5.0);
        op.refresh_preview();
        let error = op.error.as_deref().expect("the oversized blend is reported");
        assert!(error.contains("too large"), "got {error}");
        assert_eq!(op.preview.preview_nodes(), shown, "the last good preview stays");
    }

    #[test]
    fn apply_reshapes_the_part_in_place_and_finishes() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, mut selection) = targeting_edge(&ws, node);
        drag_out(&mut op, 0.3);
        op.refresh_preview();

        op.apply(&mut selection).expect("the fillet applies");
        assert!(op.is_finished());
        assert!(selection.is_empty(), "the part's edges were renumbered");
        assert!(op.preview.is_empty());
        assert_eq!(visibility(&ws, node), Visibility::Visible);
        let expected = 8.0 - 2.0 * 0.09 * (1.0 - std::f64::consts::FRAC_PI_4);
        assert!((volume(&ws, node) - expected).abs() < 1e-5, "got {}", volume(&ws, node));
        assert_eq!(ws.document.lock().unwrap().undo_label(), Some("Fillet"));
    }

    #[test]
    fn a_chamfer_applies_under_its_own_name() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, mut selection) = targeting_edge(&ws, node);
        assert!(op.on_key(&key('c'), Modifiers::default(), &mut selection));
        drag_out(&mut op, -0.3);

        op.apply(&mut selection).expect("the chamfer applies");
        assert!((volume(&ws, node) - (8.0 - 0.09)).abs() < 1e-5, "got {}", volume(&ws, node));
        assert_eq!(ws.document.lock().unwrap().undo_label(), Some("Chamfer"));
    }

    #[test]
    fn cancel_restores_the_part() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, _) = targeting_edge(&ws, node);
        drag_out(&mut op, 0.3);
        op.refresh_preview();

        op.cancel();
        assert!(op.is_finished());
        assert!(op.preview.is_empty());
        assert_eq!(visibility(&ws, node), Visibility::Visible);
        assert!((volume(&ws, node) - 8.0).abs() < 1e-9);
    }

    /// Leaving the tool commits an edit, but a zero-size one is nothing to commit.
    #[test]
    fn finalize_commits_only_a_blend_with_size() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, mut selection) = targeting_edge(&ws, node);
        op.on_handle(&HandleEvent::Begin(SIZE_HANDLE));
        op.finalize(&mut selection).expect("nothing to commit");
        assert!((volume(&ws, node) - 8.0).abs() < 1e-9);

        drag_out(&mut op, 0.3);
        op.finalize(&mut selection).expect("the edit commits");
        assert!(volume(&ws, node) < 8.0);
    }

    #[test]
    fn the_f_and_c_keys_switch_the_kind() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, mut selection) = targeting_edge(&ws, node);

        assert!(op.on_key(&key('c'), Modifiers::default(), &mut selection));
        assert_eq!(params(&op).kind, BlendKind::Chamfer);
        assert_eq!(op.panel_title(), Some("Chamfer"));
        assert!(op.is_editing(), "switching kind is an edit");

        assert!(op.on_key(&key('f'), Modifiers::default(), &mut selection));
        assert_eq!(params(&op).kind, BlendKind::Fillet);
        assert_eq!(op.panel_title(), Some("Fillet"));

        let control = Modifiers { control: true, ..Default::default() };
        assert!(!op.on_key(&key('c'), control, &mut selection), "a chord is someone else's");
    }
}
