//! The modeler's panel UI: a thin orchestrator over docked egui panels.

pub(crate) mod icons;
mod lights_tab;
mod menu_bar;
mod right_panel;
mod model_tab;
mod scene_tab;
mod selection_info;
mod tool_palette;
mod tool_panel;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_viewer::scene::PositionedCamera;
use duck_engine_viewer::selection::SelectionManager;

use crate::construction::ConstructionOptions;
use crate::document::Document;
use crate::notifications::{Notifications, Severity};
use crate::tools::ToolManager;
use crate::AppAction;

use menu_bar::MenuBar;
use right_panel::RightPanel;
use tool_palette::ToolPalette;
use tool_panel::ToolPanel;

pub use lights_tab::Lighting;

/// Widest a notice may get before its text wraps.
const NOTICE_MAX_WIDTH: f32 = 380.0;

/// Owns the modeler's persistent panel state.
#[derive(Default)]
pub struct ModelerUi {
    menu: MenuBar,
    palette: ToolPalette,
    right: RightPanel,
    tool_panel: ToolPanel,
}

impl ModelerUi {
    /// Render the panels for this frame; returns the actions requested by the
    /// UI this frame.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        document: &Arc<Mutex<Document>>,
        camera: &mut PositionedCamera,
        lighting: &mut Lighting,
        construction: &Rc<RefCell<ConstructionOptions>>,
        selection: &mut SelectionManager,
        tools: &ToolManager,
        notifications: &Notifications,
    ) -> Vec<AppAction> {
        let mut actions = Vec::new();
        let (undo_label, redo_label) = {
            let document = document.lock().unwrap();
            (
                document.undo_label().map(str::to_owned),
                document.redo_label().map(str::to_owned),
            )
        };
        self.menu.show(ctx, undo_label.as_deref(), redo_label.as_deref(), &mut actions);
        self.palette.show(ctx, tools, &mut actions);
        {
            // The document lock must be released before drawing the tool panel,
            // which may also lock the document, causing a deadlock.
            // TODO: would be better to just pass the Arc instead of locking here.
            let mut document = document.lock().unwrap();
            let mut construction = construction.borrow_mut();
            self.right.show(
                ctx,
                &mut document,
                camera,
                lighting,
                &mut construction,
                selection,
                &mut actions,
            );
        }
        self.tool_panel.show(ctx, tools, selection);
        show_notifications(ctx, notifications);
        actions
    }
}

/// egui's default fonts, with the monospace Hack as the proportional text's
/// last fallback for the arrows and symbols Ubuntu lacks.
pub fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        // The Hack typeface: egui's bundled monospace font.
        .push("Hack".to_owned());
    ctx.set_fonts(fonts);
}

/// Bottom-anchored stack of transient notices; expires on its own.
fn show_notifications(ctx: &egui::Context, notifications: &Notifications) {
    let live = notifications.live();
    if live.is_empty() {
        return;
    }
    // The panels are already drawn, so this is the 3D viewport's rect.
    let viewport = ctx.available_rect();
    let max_width = NOTICE_MAX_WIDTH.min((viewport.width() - 48.0).max(120.0));
    egui::Area::new(egui::Id::new("notifications"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -24.0))
        .constrain_to(viewport)
        .order(egui::Order::Foreground)
        .interactable(false)
        .default_width(max_width)
        .show(ctx, |ui| {
            // An area lays its contents out within last frame's measured size,
            // so wrapped text would ratchet narrower every frame. Pin the width
            // the labels wrap at instead.
            ui.set_max_width(max_width);
            for notice in &live {
                let color = match notice.severity {
                    Severity::Error => ui.visuals().error_fg_color,
                    Severity::Info => ui.visuals().text_color(),
                };
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.colored_label(color, &notice.text);
                });
            }
        });
    // Keep repainting while notices are visible so they expire without input.
    ctx.request_repaint_after(std::time::Duration::from_millis(250));
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG_NOTICE: &str =
        "Boolean failed: boolean cut operation failed: BOPAlgo_AlertBOPNotAllowed";

    /// Draw one frame of the notice stack and report its measured width.
    fn notice_width(ctx: &egui::Context, notifications: &Notifications) -> f32 {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1200.0, 800.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(input, |ctx| show_notifications(ctx, notifications));
        egui::AreaState::load(ctx, egui::Id::new("notifications"))
            .and_then(|state| state.size)
            .expect("the notice area must have been laid out")
            .x
    }

    /// Every `.rs` file under `dir`.
    fn rust_sources(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                files.extend(rust_sources(&path));
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path);
            }
        }
        files
    }

    /// A glyph the fonts lack draws as an empty box, so every non-ASCII
    /// character the code (not its comments) uses must have one.
    #[test]
    fn every_symbol_in_the_source_has_a_glyph() {
        let ctx = egui::Context::default();
        install_fonts(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let font = egui::FontId::proportional(14.0);

        let mut missing = Vec::new();
        for path in rust_sources(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")) {
            let source = std::fs::read_to_string(&path).unwrap();
            for (number, line) in source.lines().enumerate() {
                let code = line.split("//").next().unwrap_or_default();
                for c in code.chars().filter(|c| !c.is_ascii()) {
                    if !ctx.fonts_mut(|fonts| fonts.has_glyph(&font, c)) {
                        missing.push(format!("{c:?} at {}:{}", path.display(), number + 1));
                    }
                }
            }
        }
        assert!(missing.is_empty(), "no glyph for {missing:#?}");
    }

    #[test]
    fn a_short_notice_does_not_shrink_the_next_long_one() {
        let ctx = egui::Context::default();
        let short = Notifications::default();
        short.error("Boom");
        let long = Notifications::default();
        long.error(LONG_NOTICE);

        let mut short_width = 0.0;
        for _ in 0..3 {
            short_width = notice_width(&ctx, &short);
        }
        assert!(
            short_width < NOTICE_MAX_WIDTH,
            "a short notice must shrink to fit, got {short_width}px"
        );

        let mut long_width = 0.0;
        for _ in 0..3 {
            long_width = notice_width(&ctx, &long);
        }
        assert!(
            long_width > 200.0,
            "a long notice must not inherit the previous notice's width, got {long_width}px"
        );
    }
}
