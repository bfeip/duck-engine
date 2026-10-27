use duck_engine_viewer::{ViewId, Viewer};

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
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        viewer: &mut Viewer,
        view_id: ViewId,
        actions: &mut UiActions,
    ) {
        let scene = viewer.view(view_id).expect("main view").scene();
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
                    LeftPanelTab::Scene => scene_tab::show(ui, &scene, actions),
                    LeftPanelTab::Lights => {
                        let lights = viewer.scene_lights_mut(view_id).expect("main view");
                        lights_tab::show(ui, lights, actions)
                    }
                    LeftPanelTab::Environment => environment_tab::show(ui, &scene, actions),
                }
            });
    }
}
