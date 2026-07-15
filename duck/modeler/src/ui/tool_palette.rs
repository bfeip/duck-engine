//! The left tool palette: a Select button plus one button per registered tool.

use crate::tool_manager::{ToolId, ToolManager};
use crate::ui::icons;

/// The left icon strip. Stateless, tied to [`ToolManager`]
#[derive(Default)]
pub struct ToolPalette;

impl ToolPalette {
    /// Render the palette. Tool clicks are applied after the panel closure
    /// returns so no tool lock is held while egui renders.
    pub fn show(&mut self, ctx: &egui::Context, tools: &mut ToolManager) {
        let entries = tools.palette_entries();
        let active = tools.active_id();

        let mut clicked: Option<Option<ToolId>> = None;

        egui::SidePanel::left("tool_palette")
            .resizable(false)
            .exact_width(40.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);

                let (cursor_uri, cursor_bytes) = icons::CURSOR;
                let select_btn = ui
                    .add(
                        egui::Button::image(
                            egui::Image::from_bytes(cursor_uri, cursor_bytes)
                                .fit_to_exact_size(egui::vec2(16.0, 16.0)),
                        )
                        .selected(active.is_none()),
                    )
                    .on_hover_text("select");
                if select_btn.clicked() {
                    clicked = Some(None);
                }

                for (id, info) in entries.iter() {
                    ui.add_space(4.0);
                    let (icon_uri, icon_bytes) = info.icon;
                    let btn = ui
                        .add(
                            egui::Button::image(
                                egui::Image::from_bytes(icon_uri, icon_bytes)
                                    .fit_to_exact_size(egui::vec2(16.0, 16.0)),
                            )
                            .selected(active == Some(*id)),
                        )
                        .on_hover_text(match info.shortcut {
                            Some(c) => format!("{} ({})", info.id, c.to_ascii_uppercase()),
                            None => info.id.to_string(),
                        });
                    if btn.clicked() {
                        clicked = Some(Some(*id));
                    }
                }
            });

        if let Some(index) = clicked {
            tools.activate(index);
        }
    }
}
