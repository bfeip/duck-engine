use anyhow::{Context, Result};
use duck_engine_scene::resource::{SubGeometryElement, SubGeometryKind};
use duck_engine_viewer::{
    event::{Event, EventContext},
    operator::{SelectionKinds, SelectionMode},
    selection::{SelectionItem, SelectionManager},
};

use crate::ops::loft::{build_loft, LoftProfile};
use crate::preview::PreviewSession;
use crate::tools::{Gesture, ModelingTool, PanelContext, ToolInfo, Workspace};
use crate::ui::icons;
use super::edit::{apply_row, error_line, PanelAction};
use super::targets::selected_profiles;

/// Skins a surface through the wires of the selected edges, in the order they
/// were picked.
///
/// The surface previews live. Enter, right-click, Apply or switching tools adds
/// it as a part once there are two profiles; Escape or Cancel discards it.
/// Either leaves the tool.
pub struct LoftTool {
    workspace: Workspace,
    preview: PreviewSession,
    /// The profiles the preview was last built for, so it is rebuilt only when
    /// they change.
    previewed: Vec<LoftProfile>,
    /// Why the preview could not be built, shown in the panel.
    error: Option<String>,
    /// Set once the loft is applied or discarded, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}

impl LoftTool {
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            workspace: workspace.clone(),
            preview: workspace.preview_session(),
            previewed: Vec::new(),
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
            Gesture::Finish if self.is_complete() => {
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

    /// Whether there are profiles enough to skin.
    fn is_complete(&self) -> bool {
        self.previewed.len() >= 2
    }

    /// Previews the loft through the selected profiles, once they change.
    fn follow_selection(&mut self, selection: &SelectionManager) {
        let profiles = selected_profiles(selection);
        if profiles == self.previewed {
            return;
        }
        self.previewed = profiles;

        self.preview.clear_previews();
        self.error = None;
        if self.is_complete()
            && let Err(e) = self.show()
        {
            self.error = Some(format!("{e:#}"));
        }
    }

    /// Previews the loft through the profiles.
    fn show(&mut self) -> Result<()> {
        let loft = build_loft(&self.workspace.document.lock().unwrap(), &self.previewed)?;
        let options = self.workspace.preview_options();
        self.preview
            .add_preview_from_shape(&loft, &options, "Loft preview")
            .context("The loft could not be tessellated")?;
        Ok(())
    }

    /// Adds the loft through the profiles as a part of its own and finishes
    /// the tool. A failure keeps the preview so the profiles can be corrected.
    fn apply(&mut self, selection: &mut SelectionManager) -> Result<()> {
        if !self.is_complete() {
            return Ok(());
        }
        {
            let mut doc = self.workspace.document.lock().unwrap();
            let loft = build_loft(&doc, &self.previewed)?;
            doc.undo_scope("Loft")
                .add_numbered_part("Loft", loft, &self.workspace.geometry_options())
                .context("Failed to tessellate the loft")?;
        }
        // The profiles stand; only the preview goes.
        self.preview.cancel();
        selection.clear();
        self.previewed.clear();
        self.error = None;
        self.finished = true;
        Ok(())
    }

    /// Apply, reporting a failure. For the gestures that keep the tool active
    /// and so must report for themselves: the panel's Apply, Enter,
    /// right-click.
    fn apply_and_report(&mut self, selection: &mut SelectionManager) {
        if let Err(e) = self.apply(selection) {
            self.workspace.notifications.failure("Loft", &e);
        }
    }

    /// Discards the loft and finishes the tool.
    fn cancel(&mut self) {
        self.preview.cancel();
        self.previewed.clear();
        self.error = None;
        self.finished = true;
    }
}

impl ModelingTool for LoftTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "loft", icon: icons::LOFT, shortcut: None }
    }

    /// The profiles' edges.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::SubGeometry(SelectionKinds::EDGE)
    }

    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Some(gesture) = Gesture::read(event, ctx.modifiers) else { return false };
        self.on_gesture(gesture, ctx.selection)
    }

    fn panel_title(&self) -> Option<&str> {
        Some("Loft")
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        if let Some(error) = &self.error {
            error_line(ui, error);
        }
        ui.label("Profiles");
        let profiles = selected_profiles(panel.selection);
        if profiles.is_empty() {
            ui.label("(click an edge per profile)");
        }
        for (number, profile) in (1..).zip(profiles) {
            let name = self.workspace.part_name(profile.node);
            ui.horizontal(|ui| {
                ui.label(format!("{number}. {name} · edge {}", profile.edge_index));
                if ui.small_button("×").clicked() {
                    let edge = SubGeometryElement::new(SubGeometryKind::Edge, profile.edge_index);
                    let item = SelectionItem::SubGeometry { node_id: profile.node, element: edge };
                    panel.selection.remove(&item);
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

    /// Two profiles make a complete loft, so leaving the tool adds it.
    fn finalize(&mut self, selection: &mut SelectionManager) -> Result<()> {
        self.apply(selection)
    }

    fn deactivate(&mut self) {
        self.preview.cancel();
        self.previewed.clear();
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

    use crate::testing::{doc_with_two_squares, edge_item, workspace};

    /// A tool on two stacked squares, with an edge of each selected.
    fn tool() -> (LoftTool, Workspace, SelectionManager) {
        let (doc, profiles) = doc_with_two_squares();
        let ws = workspace(doc);
        let mut selection = SelectionManager::new();
        selection.extend(profiles.map(|profile| edge_item(profile.node, profile.edge_index)));
        (LoftTool::new(&ws), ws, selection)
    }

    fn part_count(ws: &Workspace) -> usize {
        ws.document.lock().unwrap().parts().count()
    }

    #[test]
    fn finishing_adds_the_loft_and_leaves() {
        let (mut op, ws, mut selection) = tool();
        op.on_gesture(Gesture::Frame, &mut selection);
        assert!(!op.preview.is_empty());

        assert!(op.on_gesture(Gesture::Finish, &mut selection));
        assert!(op.is_finished());
        assert!(op.preview.is_empty());
        assert_eq!(part_count(&ws), 3, "the loft stands beside its profiles");
    }

    /// One profile is no loft yet, so there is nothing to finish.
    #[test]
    fn a_loft_needs_two_profiles() {
        let (mut op, ws, mut selection) = tool();
        let second = selection.as_slice()[1];
        selection.remove(&second);
        op.on_gesture(Gesture::Frame, &mut selection);

        assert!(!op.on_gesture(Gesture::Finish, &mut selection));
        assert!(!op.is_finished());
        assert!(op.error.is_none());
        assert_eq!(part_count(&ws), 2);
    }

    #[test]
    fn cancelling_discards_the_loft_and_leaves() {
        let (mut op, ws, mut selection) = tool();
        op.on_gesture(Gesture::Frame, &mut selection);

        assert!(op.on_gesture(Gesture::Cancel, &mut selection));
        assert!(op.is_finished());
        assert!(op.preview.is_empty());
        assert_eq!(part_count(&ws), 2);
    }
}
