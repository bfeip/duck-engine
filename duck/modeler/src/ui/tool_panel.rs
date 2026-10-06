//! The active tool's floating options window. The `ui` module owns the window
//! chrome; the tool fills only the body via [`ModelingTool::panel_ui`].

use duck_engine_viewer::selection::SelectionManager;

use crate::tools::PanelContext;
use crate::tools::ToolManager;

/// The stateless host for the active tool's options window.
#[derive(Default)]
pub struct ToolPanel;

impl ToolPanel {
    /// Render the active tool's options window, if it declares one.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        tools: &mut ToolManager,
        selection: &mut SelectionManager,
    ) {
        let Some(mut tool) = tools.active_tool() else { return };
        let title = match tool.panel_title() {
            Some(title) => title.to_owned(),
            None => return,
        };

        let mut panel = PanelContext { selection };
        egui::Window::new(title)
            .anchor(egui::Align2::RIGHT_TOP, [-8.0, 8.0])
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| tool.panel_ui(ui, &mut panel));
    }
}
