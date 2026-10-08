//! Top menu bar; selections are emitted as [`AppAction`]s.

use crate::AppAction;

#[derive(Default)]
pub(crate) struct MenuBar;

impl MenuBar {
    /// Render the menu bar, appending any selected action to `actions`.
    /// `undo_label`/`redo_label` name the steps the Edit items would replay;
    /// `None` disables the item.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        undo_label: Option<&str>,
        redo_label: Option<&str>,
        actions: &mut Vec<AppAction>,
    ) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Import…").clicked() {
                        actions.push(AppAction::ImportCad);
                    }
                    if ui.button("Export…").clicked() {
                        actions.push(AppAction::ExportCad);
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        actions.push(AppAction::Quit);
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if history_item(ui, "Undo", undo_label, "Ctrl+Z") {
                        actions.push(AppAction::Undo);
                    }
                    if history_item(ui, "Redo", redo_label, "Ctrl+Shift+Z") {
                        actions.push(AppAction::Redo);
                    }
                });
            });
        });
    }
}

/// A menu item like "Undo Boolean  Ctrl+Z", disabled without a step label.
fn history_item(ui: &mut egui::Ui, verb: &str, label: Option<&str>, shortcut: &str) -> bool {
    let text = match label {
        Some(label) => format!("{verb} {label}"),
        None => verb.to_string(),
    };
    let button = egui::Button::new(text).shortcut_text(shortcut);
    ui.add_enabled(label.is_some(), button).clicked()
}
