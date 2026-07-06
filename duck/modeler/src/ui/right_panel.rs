//! The right-hand tabbed panel: the Model tab (part list) and the Scene tab
//! (camera, construction plane, grid, and snap settings).

use duck_engine_viewer::selection::SelectionManager;

use crate::document::Document;
use crate::operators::ConstructionOptions;
use crate::ui::model_tab::ModelTab;
use crate::ui::scene_tab::SceneTab;
use crate::ui::UiAction;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum RightTab {
    #[default]
    Model,
    Scene,
}

impl RightTab {
    const ALL: [RightTab; 2] = [
        RightTab::Model,
        RightTab::Scene,
    ];

    fn label(self) -> &'static str {
        match self {
            RightTab::Model => "Model",
            RightTab::Scene => "Scene",
        }
    }
}

#[derive(Default)]
pub struct RightPanel {
    active_tab: RightTab,
    model: ModelTab,
    scene: SceneTab,
}

impl RightPanel {
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        document: &mut Document,
        construction: &mut ConstructionOptions,
        selection: &mut SelectionManager,
        actions: &mut Vec<UiAction>,
    ) {
        egui::SidePanel::right("right_panel")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    for tab in RightTab::ALL {
                        ui.selectable_value(&mut self.active_tab, tab, tab.label());
                    }
                });
                ui.separator();

                match self.active_tab {
                    RightTab::Model => self.model.show(ui, document, selection),
                    RightTab::Scene => self.scene.show(ui, document, construction, actions),
                }
            });
    }
}
