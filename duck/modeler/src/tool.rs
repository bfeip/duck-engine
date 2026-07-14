use duck_engine_viewer::common::Point3;
use duck_engine_viewer::operator::{Operator, SelectionMode};
use duck_engine_viewer::selection::SelectionManager;

use crate::ui::icons::Icon;

/// Static palette identity for a tool. All fields are `'static` so the
/// palette can render without holding tool locks.
#[derive(Clone, Copy)]
pub struct ToolInfo {
    /// Stable identifier ("sphere", "boolean", ...), for debugging and UI keys.
    pub id: &'static str,
    /// Palette button icon, from [`crate::ui::icons`].
    pub icon: Icon,
    /// Activation shortcut key, or `None`. Matches only with no modifier
    /// (including shift) held.
    pub shortcut: Option<char>,
}

/// External state a tool panel may need beyond what the tool already owns
/// (tools own the `Document` and `ConstructionOptions` themselves).
pub struct PanelContext<'a> {
    pub selection: &'a mut SelectionManager,
}

pub trait ModelingTool: Operator {
    /// Palette identity (id + icon). Each tool file owns its own SVG.
    fn info(&self) -> ToolInfo;

    /// Clean up in-progress state (preview nodes, hidden geometry).
    /// Called automatically before any tool switch and on auto-return;
    /// must also reset any `is_finished()` latch.
    fn deactivate(&mut self);

    /// Called when the tool becomes the active tool.
    fn activate(&mut self) {}

    /// True when the tool completed or was cancelled and should cede back to selection.
    fn is_finished(&self) -> bool {
        false
    }

    /// The world-space point this tool wants the modeler's 3D cursor to mark
    /// (e.g. the current snap location), or `None` to hide it. Polled each frame
    /// while the tool is active.
    fn cursor_target(&self) -> Option<Point3> {
        None
    }

    /// Selection granularity the always-on `SelectionOperator` should use
    /// while this tool is active.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::default()
    }

    /// Title of the tool's options window, or `None` if the tool has no panel.
    fn panel_title(&self) -> Option<&str> {
        None
    }

    /// Fill the body of the tool's options window. Called only when
    /// `panel_title()` is `Some`; the `ui` module owns the window chrome.
    /// The tool's mutex is held for the duration of this call — do not
    /// trigger anything that re-dispatches events back into the tool.
    fn panel_ui(&mut self, _ui: &mut egui::Ui, _panel: &mut PanelContext) {}
}
