//! The right-hand tabbed panel (Scene / Props / Material / Light / Snap).
//! Only the Scene tab is populated today; the others render a placeholder.

use duck_engine_viewer::selection::SelectionManager;

use crate::document::Document;
use crate::ui::model_tab::ModelTab;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum RightTab {
    #[default]
    Model,
}

impl RightTab {
    const ALL: [RightTab; 1] = [
        RightTab::Model,
    ];

    fn label(self) -> &'static str {
        match self {
            RightTab::Model => "Model",
        }
    }
}

#[derive(Default)]
pub struct RightPanel {
    active_tab: RightTab,
    model: ModelTab,
}

impl RightPanel {
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        document: &mut Document,
        selection: &mut SelectionManager,
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
                }
            });
    }
}
