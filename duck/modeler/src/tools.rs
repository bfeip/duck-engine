//! Modeling tools: the palette entries that build and edit parts, the
//! [`ToolManager`] that drives them, and the [`ModelingTool`] interface between
//! the two.

mod boolean;
mod r#box;
mod circle;
mod curve;
mod cylinder;
mod draft;
mod duplicate;
mod extrude;
mod fillet;
mod hollow;
mod line;
mod loft;
mod manager;
mod rectangle;
mod sphere;
mod targeted;
mod thicken;
mod transform;
mod tweak;

pub use boolean::BooleanTool;
pub use r#box::BoxTool;
pub use circle::CircleTool;
pub use curve::CurveTool;
pub use cylinder::CylinderTool;
pub use draft::DraftTool;
pub use duplicate::DuplicateTool;
pub use extrude::ExtrudeTool;
pub use fillet::FilletTool;
pub use hollow::HollowTool;
pub use line::LineTool;
pub use loft::LoftTool;
pub use manager::{ToolId, ToolManager};
pub use rectangle::RectangleTool;
pub use sphere::SphereTool;
pub use thicken::ThickenTool;
pub use transform::TransformTool;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::resource::NodeId;
use duck_engine_viewer::common::Point3;
use duck_engine_viewer::event::{Event, EventContext};
use duck_engine_viewer::operator::{Handle, HandleEvent, SelectionMode};
use duck_engine_viewer::selection::SelectionManager;
use opencascade::primitives::Shape;

use crate::construction::ConstructionOptions;
use crate::document::{Document, PartId};
use crate::notifications::Notifications;
use crate::preview::PreviewSession;
use crate::snap::{Snap, SnapFlags, SnapInput, SnapProvider};
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

/// External state a tool panel may need beyond the tool's own [`Workspace`].
pub struct PanelContext<'a> {
    pub selection: &'a mut SelectionManager,
}

/// What every tool works with: the document it edits, the construction
/// settings it builds with, and the notices it reports failures to.
#[derive(Clone)]
pub struct Workspace {
    pub document: Arc<Mutex<Document>>,
    pub construction: Rc<RefCell<ConstructionOptions>>,
    pub notifications: Notifications,
}

impl Workspace {
    /// A preview session on the document.
    pub fn preview_session(&self) -> PreviewSession {
        PreviewSession::new(Arc::clone(&self.document))
    }

    /// Tessellation options for committed geometry.
    pub fn geometry_options(&self) -> CadTessellationOptions {
        self.construction.borrow().geometry_options.clone()
    }

    /// Coarser tessellation options for previews.
    pub fn preview_options(&self) -> CadTessellationOptions {
        self.construction.borrow().preview_options()
    }

    /// Adds `shape` as a new part numbered in `base`'s series, tessellated for
    /// committed geometry.
    pub fn add_numbered_part(&self, base: &str, shape: Shape) -> anyhow::Result<PartId> {
        let options = self.geometry_options();
        self.document.lock().unwrap().add_numbered_part(base, shape, &options)
    }

    /// The snapped world point under `cursor`, ignoring the `exclude` nodes,
    /// such as the tool's own preview.
    pub fn snap(&self, cursor: (f32, f32), exclude: &[NodeId], ctx: &EventContext) -> Option<Snap> {
        self.snap_with(cursor, exclude, ctx, &[])
    }

    /// [`snap`](Self::snap), with `extra` candidates competing alongside the
    /// registered providers, such as an in-progress wire's start point.
    pub fn snap_with(
        &self,
        cursor: (f32, f32),
        exclude: &[NodeId],
        ctx: &EventContext,
        extra: &[&dyn SnapProvider],
    ) -> Option<Snap> {
        let construction = self.construction.borrow();
        let input = SnapInput {
            ray: ctx.camera.ray_from_screen_point(cursor.0, cursor.1, ctx.size.0, ctx.size.1),
            cursor,
            viewport: ctx.size,
            camera: &*ctx.camera,
            plane: &construction.construction_plane,
            grid: &construction.grid,
            requested: SnapFlags::all(),
            exclude_nodes: exclude,
        };
        construction.snap.snap(&input, &ctx.scene, extra)
    }
}

/// A palette tool, as the [`ToolManager`] drives it.
pub trait ModelingTool: 'static {
    // Identity

    /// Palette identity: id, icon and shortcut.
    fn info(&self) -> ToolInfo;

    /// Selection granularity the always-on `SelectionOperator` should use
    /// while this tool is active.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::default()
    }

    // Input

    /// Handles one viewport event. Returns `true` to consume it, so that it
    /// reaches neither the selection nor the camera.
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool;

    /// The draggable handles this tool wants shown, or empty for none.
    ///
    /// Polled each frame while the tool is active, like
    /// [`ModelingTool::cursor_target`], so it should be cheap and is free to
    /// return a different set as the tool changes phase.
    fn handles(&self) -> Vec<Handle> {
        Vec::new()
    }

    /// Act on a grab, drag, release or cancel of one of this tool's handles.
    ///
    /// Only called for ids this tool's [`handles`](ModelingTool::handles)
    /// produced. [`HandleEvent::Begin`] is the cue to snapshot whatever
    /// [`HandleEvent::Drag`] edits: a drag reports its total offset from the
    /// grab point, not a per-event increment.
    fn on_handle(&mut self, _event: &HandleEvent) {}

    // Display

    /// The world-space point this tool wants the modeler's 3D cursor to mark
    /// (e.g. the current snap location), or `None` to hide it.
    ///
    /// Polled each frame while the tool is active.
    fn cursor_target(&self) -> Option<Point3> {
        None
    }

    /// Title of the tool's options window, or `None` if the tool has no panel.
    fn panel_title(&self) -> Option<&str> {
        None
    }

    /// Fill the body of the tool's options window.
    ///
    /// Called only when `panel_title()` is `Some`; the `ui` module owns the window chrome.
    /// The tool's mutex is typically held for the duration of this call — do not
    /// trigger anything that re-dispatches events back into the tool.
    fn panel_ui(&mut self, _ui: &mut egui::Ui, _panel: &mut PanelContext) {}

    // Lifecycle

    /// Called when the tool becomes the active tool.
    fn activate(&mut self) {}

    /// Commit whatever fully defined result the tool is holding, as if Apply had
    /// been pressed; a tool with nothing pending should do nothing.
    ///
    /// Called before [`ModelingTool::deactivate`] when the user leaves the
    /// tool by a gesture that isn't an explicit discard, so that switching
    /// tools completes the operation instead of throwing it away.
    ///
    /// The error is reported for the tool by the caller — log or notify nothing
    /// here. `deactivate` follows either way, so a failed commit is discarded.
    fn finalize(&mut self, _selection: &mut SelectionManager) -> anyhow::Result<()> {
        Ok(())
    }

    /// Clean up in-progress state (preview nodes, hidden geometry).
    ///
    /// Called automatically before any tool switch and on auto-return;
    /// must also reset any `is_finished()` latch.
    fn deactivate(&mut self);

    /// True when the tool completed or was cancelled and should cede back to selection.
    fn is_finished(&self) -> bool {
        false
    }
}
