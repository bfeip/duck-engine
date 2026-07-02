//! Top menu bar; selections are emitted as [`UiAction`]s.

use super::UiAction;

#[derive(Default)]
pub(crate) struct MenuBar;

impl MenuBar {
    /// Render the menu bar, appending any selected action to `actions`.
    pub fn show(&mut self, ctx: &egui::Context, actions: &mut Vec<UiAction>) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Import…").clicked() {
                        actions.push(UiAction::ImportCad);
                    }
                    if ui.button("Export…").clicked() {
                        actions.push(UiAction::ExportCad);
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        actions.push(UiAction::Quit);
                    }
                });
            });
        });
    }
}
