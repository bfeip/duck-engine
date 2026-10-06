use anyhow::Result;
use duck_engine_common::Real;
use duck_engine_scene::resource::NodeId;
use duck_engine_viewer::{
    operator::{
        Handle, HandleDrag, HandleId, HandleReach, HandleShape, SelectionKinds, SelectionMode,
    },
    selection::SelectionManager,
};
use opencascade::primitives::Shape;

use crate::construction::ConstructionOptions;
use crate::document::{Document, SourceFate};
use crate::ops::thicken::{build_thicken, ThickenFrame, ThickenParams, ThickenTarget};
use crate::tools::ToolInfo;
use crate::ui::icons;
use super::edit::{length_field, Params};
use super::feature::{EditLock, Feature, FeatureTool};
use super::targets::{count_summary, selected_faces_or_part, Selected};

/// The thickness grips.
const FRONT_HANDLE: HandleId = HandleId(0);
const BACK_HANDLE: HandleId = HandleId(1);

impl Params for ThickenParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        if self.frame.on_solid {
            ui.label("Result");
            ui.horizontal(|ui| {
                changed |= ui.selectable_value(&mut self.new_body, false, "Join").changed();
                changed |= ui.selectable_value(&mut self.new_body, true, "New body").changed();
            });
            ui.end_row();
        }

        let mut front = self.front;
        let label = if self.back_applies() { "Front" } else { "Thickness" };
        if length_field(ui, label, &mut front, 0.0..=Real::MAX) {
            self.set_front(front);
            changed = true;
        }
        if self.back_applies() {
            let mut back = self.back;
            if length_field(ui, "Back", &mut back, 0.0..=Real::MAX) {
                self.set_back(back);
                changed = true;
            }
            ui.label("Sides");
            let mut lock = self.lock;
            if ui.checkbox(&mut lock, "Equal  S").changed() {
                self.set_lock(lock);
                changed = true;
            }
            ui.end_row();
        }
        changed
    }

    /// An arrow out along the faces' normal for the front and, where the back
    /// counts, one against it, each on an arm so that the two stay apart at
    /// zero.
    fn handles(&self) -> Vec<Handle> {
        let normal = self.frame.normal;
        let mut handles = vec![
            Handle::new(FRONT_HANDLE, HandleShape::Cone, self.front_grip())
                .with_direction(normal)
                .with_reach(HandleReach::Arm),
        ];
        if self.back_applies() {
            handles.push(
                Handle::new(BACK_HANDLE, HandleShape::Cone, self.back_grip())
                    .with_direction(-normal)
                    .with_reach(HandleReach::Arm),
            );
        }
        handles
    }

    /// Each grip follows the cursor along the normal, one for one, stopping
    /// at the faces.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        let normal = grabbed.frame.normal;
        match drag.id {
            FRONT_HANDLE => self.set_front(grabbed.front + drag.distance_along(normal)),
            BACK_HANDLE => self.set_back(grabbed.back + drag.distance_along(-normal)),
            _ => {}
        }
    }

    fn is_degenerate(&self) -> bool {
        ThickenParams::is_degenerate(self)
    }
}

/// Grows selected faces, or a whole sheet, into a solid slab.
pub type ThickenTool = FeatureTool<Thicken>;

/// The thicken operation, on faces of one part or a sheet selected whole. A
/// face of a solid joins its slab to the solid unless it makes a new body.
/// Editing locks the part, though its faces can still be shift-clicked in and
/// out.
#[derive(Default)]
pub struct Thicken;

impl Feature for Thicken {
    type Target = ThickenTarget;
    type Params = ThickenParams;

    const TOOL: ToolInfo = ToolInfo { id: "thicken", icon: icons::THICKEN, shortcut: None };
    const SELECTS: SelectionMode = SelectionMode::SubGeometry(SelectionKinds::FACE);
    const LOCK: EditLock = EditLock::Part;

    fn select(&self, selection: &SelectionManager, locked: Option<NodeId>) -> Option<Selected<ThickenTarget>> {
        let selected = selected_faces_or_part(selection, locked)?;
        Some(selected.map(|node, faces| ThickenTarget { node, faces }))
    }

    /// The grips ride the primary face; an edit keeps its thicknesses and
    /// choices.
    fn resolve(
        &self,
        doc: &Document,
        target: &ThickenTarget,
        _construction: &ConstructionOptions,
        edited: Option<&ThickenParams>,
    ) -> Result<ThickenParams> {
        let frame = ThickenFrame::new(doc, target)?;
        Ok(match edited {
            Some(params) => ThickenParams { frame, ..*params },
            None => ThickenParams::new(frame),
        })
    }

    fn build(&self, doc: &Document, target: &ThickenTarget, params: &ThickenParams) -> Result<Shape> {
        build_thicken(doc, target, params)
    }

    /// The slab joins a solid unless it makes a new body, replaces a sheet it
    /// covers whole, and otherwise stands beside its part.
    fn fate(&self, params: &ThickenParams) -> SourceFate {
        params.fate()
    }

    fn title(&self, _params: Option<&ThickenParams>) -> &'static str {
        "Thicken"
    }

    fn new_part_name(&self) -> &'static str {
        "Thickened"
    }

    fn prompt(&self, _selection: &SelectionManager) -> &'static str {
        "Select faces or a sheet to thicken."
    }

    fn summary(&self, target: &ThickenTarget, ignored: usize) -> Option<String> {
        if !target.faces.is_empty() {
            return Some(count_summary("face", target.faces.len(), ignored));
        }
        Some(match ignored {
            0 => "The whole sheet".to_owned(),
            ignored => format!("The whole sheet ({ignored} on other parts ignored)"),
        })
    }

    /// S holds front and back equal; N switches faces of a solid between
    /// joining it and a new body.
    fn on_char(&self, c: char, params: &ThickenParams) -> Option<ThickenParams> {
        let mut edited = *params;
        match c {
            // Taken even while the back doesn't count, so that S never falls
            // through to the Scale shortcut mid-edit.
            's' => edited.set_lock(!params.lock),
            'n' if params.frame.on_solid => edited.new_body = !params.new_body,
            _ => return None,
        }
        Some(edited)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_common::InnerSpace;
    use duck_engine_scene::resource::{NodeFlags, Visibility};
    use duck_engine_viewer::operator::HandleEvent;
    use duck_engine_viewer::selection::SelectionItem;
    use glam::DVec3;

    use crate::testing::{
        face_item, face_item_along, tube, visibility, volumes, workspace_with, workspace_with_box,
    };
    use crate::tools::{Gesture, ModelingTool, Workspace};

    const EPSILON: Real = 1e-5;

    fn targeting(ws: &Workspace, items: &[SelectionItem]) -> (ThickenTool, SelectionManager) {
        let mut op = ThickenTool::new(ws);
        let mut selection = SelectionManager::new();
        selection.extend(items.iter().copied());
        op.follow_selection(&selection);
        (op, selection)
    }

    fn params(op: &ThickenTool) -> ThickenParams {
        *op.params().expect("a thickening is targeted")
    }

    /// Grabs grip `id` and drags it `distance` the way it points, then lets go.
    fn drag_out(op: &mut ThickenTool, id: HandleId, distance: Real) {
        let grabbed = params(op);
        let (grab, outward) = match id {
            FRONT_HANDLE => (grabbed.front_grip(), grabbed.frame.normal),
            _ => (grabbed.back_grip(), -grabbed.frame.normal),
        };
        op.on_handle(&HandleEvent::Begin(id));
        op.on_handle(&HandleEvent::Drag(crate::testing::drag(id, grab, outward * distance)));
        op.on_handle(&HandleEvent::End(id));
    }

    /// Joining shows one arrow; a new body adds one against the normal, the
    /// two on arms pointing apart.
    #[test]
    fn the_back_grip_shows_only_where_the_back_counts() {
        let (ws, node) = workspace_with_box();
        let (op, _) = targeting(&ws, &[face_item_along(&ws, node, DVec3::Y)]);
        let joined = params(&op);
        assert_eq!(joined.fate(), SourceFate::Fuse);
        let [front] = <[Handle; 1]>::try_from(joined.handles()).expect("one grip");
        assert_eq!(front.id, FRONT_HANDLE);
        assert_eq!(front.reach, HandleReach::Arm);
        assert!((front.direction - joined.frame.normal).magnitude() < EPSILON);

        let apart = ThickenParams { new_body: true, ..joined };
        let [front, back] = <[Handle; 2]>::try_from(apart.handles()).expect("two grips");
        assert!((front.anchor - back.anchor).magnitude() < EPSILON, "both start on the face");
        assert!((front.direction + back.direction).magnitude() < EPSILON, "they point apart");
        assert_eq!(back.reach, HandleReach::Arm);
    }

    #[test]
    fn the_grips_set_each_side_one_for_one_and_stop_at_the_faces() {
        let (ws, node) = workspace_with(tube());
        let (mut op, _) = targeting(&ws, &[face_item(node, 0)]);
        drag_out(&mut op, FRONT_HANDLE, 0.3);
        drag_out(&mut op, BACK_HANDLE, 0.2);
        assert!((params(&op).front - 0.3).abs() < EPSILON);
        assert!((params(&op).back - 0.2).abs() < EPSILON);

        drag_out(&mut op, FRONT_HANDLE, -1.0);
        assert!(params(&op).front.abs() < EPSILON, "dragged back through the face, it stops there");
    }

    /// S evens the sides and holds them, and is taken even while joining, so
    /// that it never reaches the Scale shortcut.
    #[test]
    fn s_locks_the_sides_together_and_is_always_taken() {
        let (ws, node) = workspace_with_box();
        let (mut op, mut selection) = targeting(&ws, &[face_item_along(&ws, node, DVec3::Y)]);
        assert!(op.on_gesture(Gesture::Key('s'), &mut selection), "taken while joining");
        assert!(params(&op).lock);
        assert!(op.is_editing());

        assert!(op.on_gesture(Gesture::Key('n'), &mut selection));
        drag_out(&mut op, FRONT_HANDLE, 0.4);
        assert!((params(&op).back - 0.4).abs() < EPSILON, "the back follows the front");
    }

    /// N switches a solid's faces between joining it and a new body, which
    /// changes nothing on a sheet.
    #[test]
    fn n_switches_a_solids_faces_to_a_new_body() {
        let (ws, node) = workspace_with_box();
        let (mut op, mut selection) = targeting(&ws, &[face_item_along(&ws, node, DVec3::Y)]);
        assert!(op.on_gesture(Gesture::Key('n'), &mut selection));
        assert_eq!(params(&op).fate(), SourceFate::Keep);
        assert!(op.on_gesture(Gesture::Key('n'), &mut selection));
        assert_eq!(params(&op).fate(), SourceFate::Fuse);

        let (ws, node) = workspace_with(tube());
        let (mut op, mut selection) = targeting(&ws, &[face_item(node, 0)]);
        assert!(!op.on_gesture(Gesture::Key('n'), &mut selection), "a sheet has no solid to join");
    }

    #[test]
    fn a_whole_solid_says_to_pick_its_faces() {
        let (ws, node) = workspace_with_box();
        let (op, _) = targeting(&ws, &[SelectionItem::Node(node)]);
        assert!(op.target().is_none());
        let error = op.error().expect("the refusal is reported");
        assert!(error.contains("faces of the solid"), "got {error}");
    }

    /// Joining previews the slab beside the solid, which stays in view; apply
    /// grows the solid in place.
    #[test]
    fn a_joined_slab_previews_beside_its_solid_and_applies_in_place() {
        let (ws, node) = workspace_with_box();
        let (mut op, mut selection) = targeting(&ws, &[face_item_along(&ws, node, DVec3::Y)]);
        drag_out(&mut op, FRONT_HANDLE, 0.5);
        op.refresh_preview();
        assert!(op.error().is_none(), "{:?}", op.error());
        assert_eq!(visibility(&ws, node), Visibility::Visible);
        let [preview] = <[NodeId; 1]>::try_from(op.preview().preview_nodes()).expect("one preview");
        let scene = ws.document.lock().unwrap().scene().clone();
        assert!(scene.lock().get_node(preview).unwrap().flags().contains(NodeFlags::DO_NOT_SELECT));

        op.apply(&mut selection).expect("the slab joins");
        assert!(op.is_finished());
        assert!(op.preview().is_empty());
        let doc = ws.document.lock().unwrap();
        let part = doc.part_for_node(node).expect("the solid keeps its node");
        assert!((doc.get_part(part).unwrap().shape.volume() - 10.0).abs() < 1e-6);
        assert_eq!(doc.undo_label(), Some("Thicken"));
    }

    #[test]
    fn a_new_body_applies_beside_its_solid() {
        let (ws, node) = workspace_with_box();
        let (mut op, mut selection) = targeting(&ws, &[face_item_along(&ws, node, DVec3::Y)]);
        assert!(op.on_gesture(Gesture::Key('n'), &mut selection));
        drag_out(&mut op, FRONT_HANDLE, 0.5);
        drag_out(&mut op, BACK_HANDLE, 0.25);
        op.refresh_preview();

        op.apply(&mut selection).expect("the slab is added");
        let volumes = volumes(&ws.document.lock().unwrap());
        assert_eq!(volumes.len(), 2);
        assert!((volumes[0] - 8.0).abs() < 1e-9);
        assert!((volumes[1] - 3.0).abs() < 1e-6, "got {}", volumes[1]);
        assert_eq!(visibility(&ws, node), Visibility::Visible);
        let doc = ws.document.lock().unwrap();
        let names: Vec<_> = doc.parts().map(|part| part.name.as_str()).collect();
        assert_eq!(names, ["part", "Thickened-001"]);
    }

    /// The slab hides a sheet it would replace. Shift-clicking one of the
    /// sheet's faces out leaves the sheet standing, and it comes back into
    /// view.
    #[test]
    fn a_sheet_shows_again_once_it_would_be_kept() {
        let (ws, node) = workspace_with(tube());
        let all: Vec<_> = (0..4).map(|index| face_item(node, index)).collect();
        let (mut op, mut selection) = targeting(&ws, &all);
        drag_out(&mut op, FRONT_HANDLE, 0.1);
        op.refresh_preview();
        assert_eq!(params(&op).fate(), SourceFate::Replace);
        assert_eq!(visibility(&ws, node), Visibility::Invisible);

        selection.toggle(face_item(node, 3));
        op.follow_selection(&selection);
        op.refresh_preview();
        assert_eq!(params(&op).fate(), SourceFate::Keep);
        assert!((params(&op).front - 0.1).abs() < EPSILON, "the edit keeps its thickness");
        assert_eq!(visibility(&ws, node), Visibility::Visible);

        op.apply(&mut selection).expect("the slab is added");
        assert_eq!(volumes(&ws.document.lock().unwrap()).len(), 2, "the tube stays");
        assert_eq!(visibility(&ws, node), Visibility::Visible);
    }

    #[test]
    fn a_replacing_slab_consumes_its_sheet_and_cancel_brings_it_back() {
        let (ws, node) = workspace_with(tube());
        let all: Vec<_> = (0..4).map(|index| face_item(node, index)).collect();
        let (mut op, mut selection) = targeting(&ws, &all);
        drag_out(&mut op, FRONT_HANDLE, 0.1);
        op.refresh_preview();

        op.cancel();
        assert_eq!(visibility(&ws, node), Visibility::Visible);

        let (mut op, _) = targeting(&ws, &all);
        drag_out(&mut op, FRONT_HANDLE, 0.1);
        op.refresh_preview();
        op.apply(&mut selection).expect("the walls replace the tube");
        let doc = ws.document.lock().unwrap();
        assert_eq!(doc.parts().count(), 1);
        assert!(doc.part_for_node(node).is_none(), "the tube was consumed");
    }
}
