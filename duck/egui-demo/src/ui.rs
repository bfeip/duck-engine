mod environment_tab;
mod info_sections;
mod left_panel;
mod lights_tab;
mod right_panel;
mod scene_tab;

pub use left_panel::LeftPanel;
pub use right_panel::RightPanel;

use std::sync::{Arc, Mutex};
use duck_engine_viewer::operator::{NavigationMode, NavigationOperator};
use duck_engine_viewer::scene::{LightType, PositionedCamera};
use duck_engine_viewer::scene::resource::{NodeId, Visibility};
use duck_engine_viewer::ViewMut;

/// Minimum window width (in logical points) for the GUI to be shown.
/// Below this the demo renders only the 3D scene.
const MIN_UI_WIDTH: f32 = 600.0;

/// A visibility change requested by the UI.
pub struct VisibilityChange {
    pub node_id: NodeId,
    pub new_visibility: Visibility,
}

/// Actions requested by the UI that need to be handled by the application.
#[derive(Default)]
pub struct UiActions {
    pub load_scene: bool,
    pub clear_scene: bool,
    pub add_light: Option<LightType>,
    pub load_environment: bool,
    pub clear_environment: bool,
    pub visibility_changes: Vec<VisibilityChange>,
    /// When set, replace the view's camera with this value.
    pub set_camera: Option<PositionedCamera>,
}

/// Information about the current navigation mode, shared across panels.
pub struct ModeInfo {
    pub mode: NavigationMode,
}

/// Persistent UI state stored in the application.
#[derive(Default)]
pub struct UiState {
    pub left: LeftPanel,
    pub right: RightPanel,
}

impl UiState {
    pub fn build(
        &mut self,
        ctx: &egui::Context,
        view: &mut ViewMut<'_>,
        nav_op: &Arc<Mutex<NavigationOperator>>,
    ) -> UiActions {
        let mut actions = UiActions::default();

        // On small / mobile windows the panels are unusable; show only the 3D scene.
        if ctx.content_rect().width() < MIN_UI_WIDTH {
            return actions;
        }

        let mode_info = ModeInfo { mode: nav_op.lock().unwrap().mode() };

        build_performance_panel(ctx, &mode_info);
        self.left.show(ctx, view, &mut actions);
        self.right.show(ctx, view);

        actions
    }
}

fn build_performance_panel(ctx: &egui::Context, mode: &ModeInfo) {
    egui::TopBottomPanel::new(egui::panel::TopBottomSide::Top, "Performance")
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("FPS: {:.1}", ctx.input(|i| i.stable_dt).recip()));
                ui.separator();
                match mode.mode {
                    NavigationMode::Walk => ui.label("Mode: Walk"),
                    NavigationMode::Turntable => ui.label("Mode: Turntable"),
                    NavigationMode::Trackball => ui.label("Mode: Trackball"),
                };
            });
        });
}
