use anyhow::{Context, Result};
use duck_engine_scene::cad::{tessellate_into_with_materials, CadTessellationOptions};
use duck_engine_scene::common::RgbaColor;
use duck_engine_scene::resource::{FaceMaterial, LineMaterial, NodeFlags, NodeId};
use duck_engine_viewer::{
    event::{Event, EventContext},
    operator::SelectionMode,
    selection::{SelectionItem, SelectionManager},
};
use opencascade::primitives::Shape;

use crate::ops::boolean::{
    build_boolean, commit_boolean, preview_boolean, BooleanKind, BooleanTarget,
};
use crate::preview::PreviewSession;
use crate::tools::{Gesture, ModelingTool, PanelContext, ToolInfo, Workspace};
use crate::ui::icons;
use super::edit::{apply_row, error_line, PanelAction};
use super::targets::selected_boolean;

/// Combines the selected parts: the primary part is the target, cut by, joined
/// with or intersected with the others.
///
/// The result previews live, with the material it removes in translucent red.
/// Enter, right-click, Apply or switching tools applies it once there is a part
/// to combine with; Escape or Cancel discards it. Either leaves the tool.
pub struct BooleanTool {
    kind: BooleanKind,
    workspace: Workspace,
    preview: PreviewSession,
    /// The parts and kind the preview was last built for: a target and at least
    /// one tool, or `None` while the selection designates no such boolean.
    previewed: Option<(BooleanTarget, BooleanKind)>,
    /// Why the preview could not be built, shown in the panel.
    error: Option<String>,
    /// Set once the boolean is applied or discarded, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}

impl BooleanTool {
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            kind: BooleanKind::default(),
            workspace: workspace.clone(),
            preview: workspace.preview_session(),
            previewed: None,
            error: None,
            finished: false,
        }
    }

    /// Acts on `gesture`, returning whether it was consumed.
    fn on_gesture(&mut self, gesture: Gesture, selection: &mut SelectionManager) -> bool {
        match gesture {
            Gesture::Frame => {
                self.follow_selection(selection);
                false
            }
            Gesture::Finish if self.previewed.is_some() => {
                self.apply_and_report(selection);
                true
            }
            Gesture::Cancel => {
                self.cancel();
                true
            }
            Gesture::Finish | Gesture::Hover(_) | Gesture::Click { .. } | Gesture::Key(_) => false,
        }
    }

    /// Previews the boolean the selection designates, once it or the kind
    /// changes.
    fn follow_selection(&mut self, selection: &SelectionManager) {
        let designated = selected_boolean(selection)
            .filter(|target| !target.tools.is_empty())
            .map(|target| (target, self.kind));
        if designated == self.previewed {
            return;
        }
        self.previewed = designated;

        // Drop the old preview, showing again the parts it hid.
        self.preview.clear_previews();
        self.error = None;
        let Some((target, kind)) = self.previewed.clone() else { return };
        if let Err(e) = self.show(&target, kind) {
            self.error = Some(format!("{e:#}"));
        }
    }

    /// Previews `kind` of `target`: the result in place of the parts it
    /// combines, and the material it removes.
    fn show(&mut self, target: &BooleanTarget, kind: BooleanKind) -> Result<()> {
        let preview = preview_boolean(&self.workspace.document.lock().unwrap(), target, kind)?;
        let options = self.workspace.preview_options();
        self.preview
            .add_preview_from_shape(&preview.shape, &options, "Boolean preview")
            .context("The boolean could not be tessellated")?;
        self.preview.hide_source_node(target.target);
        for &tool in &target.tools {
            self.preview.hide_source_node(tool);
        }
        self.show_removed(kind, &target.tools, &preview.removed, &options);
        // No preview is a part: picks must reach the hidden parts beneath.
        self.preview.set_preview_flags(NodeFlags::DO_NOT_SELECT);
        Ok(())
    }

    /// Show the `removed` material in translucent red.
    fn show_removed(
        &mut self,
        kind: BooleanKind,
        tools: &[NodeId],
        removed: &[Shape],
        options: &CadTessellationOptions,
    ) {
        // A subtract removes its tools whole, and their meshes already exist.
        let (ghosted, pieces): (&[NodeId], &[Shape]) =
            if kind == BooleanKind::Subtract { (tools, &[]) } else { (&[], removed) };
        if ghosted.is_empty() && pieces.is_empty() {
            return;
        }

        let scene = self.workspace.document.lock().unwrap().scene().clone();
        let (face, line) = {
            let mut scene = scene.lock();
            (
                scene.add_face_material(removed_face_material()),
                scene.add_line_material(removed_line_material()),
            )
        };
        for &tool in ghosted {
            self.preview.ghost_source_node(tool, &face, &line);
        }
        for piece in pieces {
            match tessellate_into_with_materials(
                piece,
                &scene,
                options,
                &face,
                &line,
                None,
                Some("Boolean removed"),
            ) {
                Ok(node) => self.preview.add_preview_node(node.id()),
                Err(e) => log::warn!("Removed material could not be tessellated: {e}"),
            }
        }
    }

    /// Commits the boolean previewed and finishes the tool. A failure keeps
    /// the preview so the selection can be corrected.
    fn apply(&mut self, selection: &mut SelectionManager) -> Result<()> {
        let Some((target, kind)) = self.previewed.clone() else { return Ok(()) };
        {
            let mut doc = self.workspace.document.lock().unwrap();
            let result = build_boolean(&doc, &target, kind)?;
            commit_boolean(&mut doc, &target, result, &self.workspace.geometry_options())?;
        }
        // The parts the preview hid were consumed by the result.
        let _ = self.preview.commit();
        selection.clear();
        self.previewed = None;
        self.error = None;
        self.finished = true;
        Ok(())
    }

    /// Apply, reporting a failure. For the gestures that keep the tool active
    /// and so must report for themselves: the panel's Apply, Enter,
    /// right-click.
    fn apply_and_report(&mut self, selection: &mut SelectionManager) {
        if let Err(e) = self.apply(selection) {
            self.workspace.notifications.failure("Boolean", &e);
        }
    }

    /// Discards the boolean and finishes the tool, showing again the parts the
    /// preview hid.
    fn cancel(&mut self) {
        self.preview.cancel();
        self.previewed = None;
        self.error = None;
        self.finished = true;
    }
}

/// Translucent red for the material an operation removes. Single-sided: the
/// faces it shares with the result face the other way, so culling hides them
/// rather than letting them z-fight.
fn removed_face_material() -> FaceMaterial {
    FaceMaterial::new()
        .with_base_color_factor(RgbaColor { r: 0.85, g: 0.08, b: 0.12, a: 0.4 })
        .with_roughness_factor(0.32)
}

fn removed_line_material() -> LineMaterial {
    LineMaterial::new(RgbaColor { r: 0.6, g: 0.05, b: 0.08, a: 1.0 })
}

impl ModelingTool for BooleanTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "boolean", icon: icons::BOOLEAN, shortcut: None }
    }

    /// Whole parts.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::Node
    }

    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Some(gesture) = Gesture::read(event, ctx.modifiers) else { return false };
        self.on_gesture(gesture, ctx.selection)
    }

    fn panel_title(&self) -> Option<&str> {
        Some("Boolean Operation")
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        if let Some(error) = &self.error {
            error_line(ui, error);
        }
        ui.label("Operation");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.kind, BooleanKind::Subtract, "Subtract");
            ui.selectable_value(&mut self.kind, BooleanKind::Union, "Union");
            ui.selectable_value(&mut self.kind, BooleanKind::Intersect, "Intersect");
        });
        ui.separator();

        let designated = selected_boolean(panel.selection);
        ui.label("Target");
        let target = designated.as_ref().map(|designated| self.workspace.part_name(designated.target));
        ui.label(target.as_deref().unwrap_or("(none — click a part)"));
        ui.separator();

        ui.label("Tools");
        let tools = designated.map(|designated| designated.tools).unwrap_or_default();
        if tools.is_empty() {
            ui.label("(shift-click parts to add tools)");
        }
        for tool in tools {
            let name = self.workspace.part_name(tool);
            ui.horizontal(|ui| {
                ui.label(name);
                if ui.small_button("×").clicked() {
                    panel.selection.remove(&SelectionItem::Node(tool));
                }
            });
        }
        ui.separator();

        match apply_row(ui) {
            PanelAction::Apply => self.apply_and_report(panel.selection),
            PanelAction::Cancel => self.cancel(),
            PanelAction::None | PanelAction::Changed => {}
        }
    }

    /// A boolean with a part to combine with is complete, so leaving the tool
    /// applies it.
    fn finalize(&mut self, selection: &mut SelectionManager) -> Result<()> {
        self.apply(selection)
    }

    fn deactivate(&mut self) {
        self.preview.cancel();
        self.previewed = None;
        self.error = None;
        self.finished = false;
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::resource::Visibility;

    use crate::testing::{doc_with_box_and_sphere, visibility, workspace};

    /// A tool on a box and a sphere cutting its corner, with both selected,
    /// the box as the target.
    fn tool() -> (BooleanTool, Workspace, SelectionManager, [NodeId; 2]) {
        let (doc, target, tool) = doc_with_box_and_sphere();
        let ws = workspace(doc);
        let mut selection = SelectionManager::new();
        selection.extend([SelectionItem::Node(target), SelectionItem::Node(tool)]);
        (BooleanTool::new(&ws), ws, selection, [target, tool])
    }

    fn part_count(ws: &Workspace) -> usize {
        ws.document.lock().unwrap().parts().count()
    }

    /// Each kind shows what it removes exactly once: a subtract as a ghost of
    /// its tool, an intersect as the two removed pieces, a union not at all.
    #[test]
    fn preview_shows_each_kinds_removed_material_once() {
        let (mut op, ws, selection, _) = tool();

        for (kind, removed_nodes) in
            [(BooleanKind::Subtract, 1), (BooleanKind::Intersect, 2), (BooleanKind::Union, 0)]
        {
            op.kind = kind;
            op.follow_selection(&selection);

            // The result node plus the removed material.
            let previews = op.preview.preview_nodes();
            assert_eq!(previews.len(), 1 + removed_nodes);
            let scene = ws.document.lock().unwrap().scene().clone();
            let scene = scene.lock();
            for &node in previews {
                assert!(scene.get_node(node).unwrap().flags().contains(NodeFlags::DO_NOT_SELECT));
            }
        }
    }

    #[test]
    fn finishing_combines_the_parts_and_leaves() {
        let (mut op, ws, mut selection, _) = tool();
        op.on_gesture(Gesture::Frame, &mut selection);

        assert!(op.on_gesture(Gesture::Finish, &mut selection));
        assert!(op.is_finished());
        assert!(op.preview.is_empty());
        assert!(selection.is_empty());
        assert_eq!(part_count(&ws), 1, "the result replaces the target and the tool");
    }

    /// A target alone is no boolean yet, so there is nothing to finish.
    #[test]
    fn a_boolean_needs_a_part_to_combine_with() {
        let (mut op, ws, mut selection, [_, tool]) = tool();
        selection.remove(&SelectionItem::Node(tool));
        op.on_gesture(Gesture::Frame, &mut selection);

        assert!(!op.on_gesture(Gesture::Finish, &mut selection));
        assert!(!op.is_finished());
        assert!(op.preview.is_empty());
        assert_eq!(part_count(&ws), 2);
    }

    #[test]
    fn cancelling_shows_the_parts_again_and_leaves() {
        let (mut op, ws, mut selection, parts) = tool();
        op.on_gesture(Gesture::Frame, &mut selection);
        assert_eq!(visibility(&ws, parts[0]), Visibility::Invisible);

        assert!(op.on_gesture(Gesture::Cancel, &mut selection));
        assert!(op.is_finished());
        for part in parts {
            assert_eq!(visibility(&ws, part), Visibility::Visible);
        }
        assert_eq!(part_count(&ws), 2);
    }

    /// A subtract that misses says why in the panel rather than previewing
    /// nothing.
    #[test]
    fn a_failed_preview_says_why() {
        let (mut op, ws, mut selection, [target, _]) = tool();
        let far = {
            let mut doc = ws.document.lock().unwrap();
            let sphere = Shape::sphere(1.0).at(glam::DVec3::splat(10.0)).build();
            let part = doc.add_part("far", sphere, &ws.geometry_options()).expect("tessellates");
            doc.node_for_part(part).expect("part has a node")
        };
        selection.clear();
        selection.extend([SelectionItem::Node(target), SelectionItem::Node(far)]);
        op.on_gesture(Gesture::Frame, &mut selection);

        assert!(op.error.is_some());
        assert!(op.preview.is_empty());
    }
}
