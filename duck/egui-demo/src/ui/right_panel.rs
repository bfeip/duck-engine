use duck_engine_viewer::ViewMut;

use super::info_sections;

/// Tab selection for the right panel. Currently only one tab, but establishes the pattern.
#[derive(Default)]
pub enum RightPanelTab {
    #[default]
    Info,
}

#[derive(Default)]
pub struct RightPanel {
    pub active_tab: RightPanelTab,
}

impl RightPanel {
    pub fn show(&self, ctx: &egui::Context, view: &ViewMut<'_>) {
        egui::SidePanel::new(egui::panel::Side::Right, "Viewer Info")
            .default_width(200.0)
            .show(ctx, |ui| match self.active_tab {
                RightPanelTab::Info => info_sections::show(ui, view),
            });
    }
}
