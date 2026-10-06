use anyhow::{Context, Result};
use duck_engine_common::Real;
use duck_engine_scene::resource::NodeId;
use duck_engine_viewer::{
    operator::{
        Handle, HandleDrag, HandleId, HandleReach, HandleShape, SelectionKinds, SelectionMode,
    },
    selection::SelectionManager,
};
use opencascade::primitives::Shape;

use crate::document::{Document, SourceFate};
use crate::ops::hollow::{build_hollow, HollowFrame, HollowParams, HollowTarget};
use crate::tools::ToolInfo;
use crate::ui::icons;
use super::targeted::{
    count_summary, selected_faces_or_part, EditLock, PreviewStyle, TargetedOp, TargetedTool,
};
use super::tweak::{length_field, TweakParams};
use crate::construction::ConstructionOptions;

/// The wall's grip.
const THICKNESS_HANDLE: HandleId = HandleId(0);

impl TweakParams for HollowParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        length_field(ui, "Thickness", &mut self.thickness, Real::MIN..=Real::MAX)
    }

    /// A grip on the far side of the wall, tied by a leader to the surface it
    /// grows from and pointing the way it has moved.
    fn handles(&self) -> Vec<Handle> {
        let inward = -self.frame.normal;
        let direction = if self.thickness < 0.0 { -inward } else { inward };
        vec![
            Handle::new(THICKNESS_HANDLE, HandleShape::Cone, self.grip())
                .with_direction(direction)
                .with_reach(HandleReach::Leader(self.frame.anchor)),
        ]
    }

    /// The grip follows the cursor across the wall, one for one: in thickens
    /// the walls inward, out past the surface grows them outward.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        if drag.id == THICKNESS_HANDLE {
            self.thickness = grabbed.thickness + drag.distance_along(-grabbed.frame.normal);
        }
    }
}

/// Shells a solid into walls, opening any of its faces selected.
pub type HollowTool = TargetedTool<Hollow>;

/// The hollow operation, on a solid selected whole, which closes around a
/// void, or on faces of one to open. Editing locks the part, though its faces
/// can still be shift-clicked in and out.
#[derive(Default)]
pub struct Hollow;

impl TargetedOp for Hollow {
    type Target = HollowTarget;
    type Params = HollowParams;

    const LOCK: EditLock = EditLock::Part;

    fn info(&self) -> ToolInfo {
        ToolInfo { id: "hollow", icon: icons::HOLLOW, shortcut: None }
    }

    /// The part first; a second click on it picks a face to open.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::Progressive(SelectionKinds::FACE)
    }

    fn title(&self, _params: Option<&HollowParams>) -> &'static str {
        "Hollow"
    }

    fn prompt(&self, _selection: &SelectionManager) -> &'static str {
        "Select a solid to hollow, then click it again for faces to open."
    }

    fn node(&self, target: &HollowTarget) -> NodeId {
        target.node
    }

    fn select(
        &self,
        selection: &SelectionManager,
        locked: Option<&HollowTarget>,
    ) -> (Option<HollowTarget>, usize) {
        let (selected, ignored) = selected_faces_or_part(selection, locked.map(|target| target.node));
        (selected.map(|(node, faces)| HollowTarget { node, faces }), ignored)
    }

    fn summary(&self, target: &HollowTarget, ignored: usize) -> Option<String> {
        if !target.is_closed() {
            return Some(format!("Opening {}", count_summary("face", target.faces.len(), ignored)));
        }
        Some(match ignored {
            0 => "Closed around a void".to_owned(),
            ignored => format!("Closed around a void ({ignored} on other parts ignored)"),
        })
    }

    /// The grip rides the rim of the primary opening; an edit keeps its
    /// thickness.
    fn resolve(
        &self,
        doc: &Document,
        target: &HollowTarget,
        _construction: &ConstructionOptions,
        edited: Option<&HollowParams>,
    ) -> Result<HollowParams> {
        let frame = HollowFrame::new(doc, target)?;
        Ok(match edited {
            Some(params) => HollowParams { frame, ..*params },
            None => HollowParams::new(frame),
        })
    }

    fn is_degenerate(&self, params: &HollowParams) -> bool {
        params.is_degenerate()
    }

    fn preview_style(&self, _params: &HollowParams) -> PreviewStyle {
        PreviewStyle::InPlace
    }

    fn build(&self, doc: &Document, target: &HollowTarget, params: &HollowParams) -> Result<Shape> {
        build_hollow(doc, target, params)
    }

    fn apply(
        &self,
        doc: &mut Document,
        target: &HollowTarget,
        params: &HollowParams,
        construction: &ConstructionOptions,
    ) -> Result<()> {
        let shape = build_hollow(doc, target, params)?;
        let part = doc.part_for_node(target.node).context("Hollow target is not a known CAD part")?;
        doc.commit_result(part, shape, SourceFate::Reshape, "Hollow", "Hollow", &construction.geometry_options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_common::InnerSpace;
    use duck_engine_scene::resource::{NodeFlags, Visibility};
    use duck_engine_viewer::input::Modifiers;
    use duck_engine_viewer::operator::HandleEvent;
    use duck_engine_viewer::selection::SelectionItem;
    use glam::DVec3;
    use opencascade::primitives::{Face, Wire};

    use crate::testing::{face_item_along, visibility, volume, workspace_with, workspace_with_box_and_cube};
    use crate::tools::targeted::Phase;
    use crate::tools::{ModelingTool, Workspace};

    const EPSILON: Real = 1e-5;

    /// A tool targeting the selection `items`.
    fn targeting(ws: &Workspace, items: &[SelectionItem]) -> (HollowTool, SelectionManager) {
        let mut op = HollowTool::new(ws);
        let mut selection = SelectionManager::new();
        selection.extend(items.iter().copied());
        op.follow_selection(&selection);
        (op, selection)
    }

    fn params(op: &HollowTool) -> HollowParams {
        *op.phase.params().expect("a hollow is targeted")
    }

    /// Grabs the grip and drags it `distance` into the part, then lets go.
    fn drag_in(op: &mut HollowTool, distance: Real) {
        let grabbed = params(op);
        let grab = grabbed.grip();
        let drag = crate::testing::drag(THICKNESS_HANDLE, grab, -grabbed.frame.normal * distance);
        op.on_handle(&HandleEvent::Begin(THICKNESS_HANDLE));
        op.on_handle(&HandleEvent::Drag(drag));
        op.on_handle(&HandleEvent::End(THICKNESS_HANDLE));
    }

    #[test]
    fn a_part_selected_whole_is_hollowed_closed() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (op, _) = targeting(&ws, &[SelectionItem::Node(node)]);
        let target = op.phase.target().expect("the part is targeted");
        assert!(target.is_closed());
        assert_eq!(Hollow.summary(target, 0).as_deref(), Some("Closed around a void"));
    }

    #[test]
    fn a_face_selected_is_opened() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let top = face_item_along(&ws, node, DVec3::Y);
        let (op, _) = targeting(&ws, &[top]);
        let target = op.phase.target().expect("the face is targeted");
        assert_eq!(target.faces.len(), 1);
        assert_eq!(Hollow.summary(target, 1).as_deref(), Some("Opening 1 face (1 on other parts ignored)"));
    }

    /// The grip sits across the wall from its anchor and points the way it
    /// has moved: inward for walls inside the surface, outward otherwise.
    #[test]
    fn the_grip_points_the_way_the_wall_grows() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (op, _) = targeting(&ws, &[SelectionItem::Node(node)]);
        let inward = HollowParams { thickness: 0.25, ..params(&op) };
        let normal = inward.frame.normal;

        let [grip] = <[Handle; 1]>::try_from(inward.handles()).expect("one grip");
        assert!((grip.anchor - (inward.frame.anchor - normal * 0.25)).magnitude() < EPSILON);
        assert!((grip.direction + normal).magnitude() < EPSILON, "points into the part");
        let HandleReach::Leader(origin) = grip.reach else { panic!("the grip has no leader") };
        assert!((origin - inward.frame.anchor).magnitude() < EPSILON);

        let outward = HollowParams { thickness: -0.25, ..inward };
        let [grip] = <[Handle; 1]>::try_from(outward.handles()).expect("one grip");
        assert!((grip.anchor - (inward.frame.anchor + normal * 0.25)).magnitude() < EPSILON);
        assert!((grip.direction - normal).magnitude() < EPSILON, "points out of the part");
    }

    #[test]
    fn the_wall_follows_its_grip_one_for_one() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, _) = targeting(&ws, &[SelectionItem::Node(node)]);
        drag_in(&mut op, 0.3);
        assert!(op.is_editing());
        assert!((params(&op).thickness - 0.3).abs() < EPSILON);

        // Dragged back out past the surface, the walls grow outward.
        drag_in(&mut op, -0.5);
        assert!((params(&op).thickness + 0.2).abs() < EPSILON);
    }

    /// While editing, shift-clicking a face of the part opens it at the
    /// thickness already set, and shift-clicking it back closes the walls.
    #[test]
    fn shift_clicked_faces_open_and_close_the_part_mid_edit() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, mut selection) = targeting(&ws, &[SelectionItem::Node(node)]);
        drag_in(&mut op, 0.25);
        assert!(!op.swallows_click(Modifiers { shift: true, ..Default::default() }));

        let top = face_item_along(&ws, node, DVec3::Y);
        selection.toggle(top);
        op.follow_selection(&selection);
        assert!(op.is_editing());
        assert_eq!(op.phase.target().expect("still targeted").faces.len(), 1);
        assert!((params(&op).thickness - 0.25).abs() < EPSILON);
        let rim = params(&op).frame;
        assert!((rim.anchor.y - 1.0).abs() < EPSILON, "the grip moves to the rim: {:?}", rim.anchor);

        selection.toggle(top);
        op.follow_selection(&selection);
        assert!(op.phase.target().expect("still targeted").is_closed());
        assert!((params(&op).thickness - 0.25).abs() < EPSILON);
    }

    #[test]
    fn a_grip_drag_previews_in_place_and_apply_hollows_the_part() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let top = face_item_along(&ws, node, DVec3::Y);
        let (mut op, mut selection) = targeting(&ws, &[top]);
        drag_in(&mut op, 0.2);
        op.refresh_preview();
        assert!(op.error.is_none(), "{:?}", op.error);
        assert_eq!(visibility(&ws, node), Visibility::Invisible);
        let [preview] = <[NodeId; 1]>::try_from(op.preview.preview_nodes()).expect("one preview");
        let scene = ws.document.lock().unwrap().scene().clone();
        assert!(scene.lock().get_node(preview).unwrap().flags().contains(NodeFlags::DO_NOT_SELECT));

        op.apply(&mut selection).expect("the hollow applies");
        assert!(op.is_finished());
        assert!(selection.is_empty());
        assert_eq!(visibility(&ws, node), Visibility::Visible);
        let expected = 8.0 - 1.6 * 1.8 * 1.6;
        assert!((volume(&ws, node) - expected).abs() < 1e-6, "got {}", volume(&ws, node));
        assert_eq!(ws.document.lock().unwrap().undo_label(), Some("Hollow"));
    }

    /// A wall too thick to build keeps the last good preview and says why.
    #[test]
    fn a_wall_too_thick_keeps_the_last_preview_and_says_why() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, _) = targeting(&ws, &[SelectionItem::Node(node)]);
        drag_in(&mut op, 0.2);
        op.refresh_preview();
        let shown = op.preview.preview_nodes().to_vec();

        drag_in(&mut op, 1.0);
        op.refresh_preview();
        let error = op.error.as_deref().expect("the wall is reported");
        assert!(error.contains("too thick"), "got {error}");
        assert_eq!(op.preview.preview_nodes(), shown);
    }

    #[test]
    fn cancel_restores_the_part() {
        let (ws, node, _) = workspace_with_box_and_cube();
        let (mut op, _) = targeting(&ws, &[SelectionItem::Node(node)]);
        drag_in(&mut op, 0.2);
        op.refresh_preview();

        op.cancel();
        assert!(op.preview.is_empty());
        assert_eq!(visibility(&ws, node), Visibility::Visible);
        assert!((volume(&ws, node) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn a_sheet_is_refused_in_the_panel() {
        let region = Face::from_wire(&Wire::rect(2.0, 2.0).unwrap()).unwrap();
        let (ws, node) = workspace_with(region.into());

        let (op, _) = targeting(&ws, &[SelectionItem::Node(node)]);
        assert!(matches!(op.phase, Phase::AwaitingSelection));
        assert!(op.error.as_deref().is_some_and(|error| error.contains("Only a solid")), "{:?}", op.error);
    }
}
