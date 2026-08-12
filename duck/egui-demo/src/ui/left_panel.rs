use duck_engine_viewer::ViewMut;

use super::{environment_tab, lights_tab, scene_tab, UiActions};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LeftPanelTab {
    #[default]
    Scene,
    Lights,
    Environment,
}

#[derive(Default)]
pub struct LeftPanel {
    pub active_tab: LeftPanelTab,
}

impl LeftPanel {
    pub fn show(&mut self, ctx: &egui::Context, view: &mut ViewMut<'_>, actions: &mut UiActions) {
        egui::SidePanel::new(egui::panel::Side::Left, "Left Panel")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.active_tab, LeftPanelTab::Scene, "Scene");
                    ui.selectable_value(&mut self.active_tab, LeftPanelTab::Lights, "Lights");
                    ui.selectable_value(&mut self.active_tab, LeftPanelTab::Environment, "Env");
                });

                ui.separator();

                match self.active_tab {
                    LeftPanelTab::Scene => scene_tab::show(ui, view, actions),
                    LeftPanelTab::Lights => lights_tab::show(ui, view, actions),
                    LeftPanelTab::Environment => environment_tab::show(ui, view, actions),
                }
            });
    }
}
