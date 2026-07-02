//! The left tool palette: a Select button plus one button per registered tool.

use crate::tool_manager::ToolManager;
use crate::ui::icons;

/// The left icon strip. Stateless, tied to [`ToolManager`]
#[derive(Default)]
pub struct ToolPalette;

impl ToolPalette {
    /// Render the palette. Tool clicks are applied after the panel closure
    /// returns so no tool lock is held while egui renders.
    pub fn show(&mut self, ctx: &egui::Context, tools: &mut ToolManager) {
        let entries = tools.palette_entries();
        let selecting = entries.iter().all(|(_, selected)| !selected);

        let mut clicked: Option<Option<usize>> = None;

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
                        .selected(selecting),
                    )
                    .on_hover_text("select");
                if select_btn.clicked() {
                    clicked = Some(None);
                }

                for (i, (info, selected)) in entries.iter().enumerate() {
                    ui.add_space(4.0);
                    let (icon_uri, icon_bytes) = info.icon;
                    let btn = ui
                        .add(
                            egui::Button::image(
                                egui::Image::from_bytes(icon_uri, icon_bytes)
                                    .fit_to_exact_size(egui::vec2(16.0, 16.0)),
                            )
                            .selected(*selected),
                        )
                        .on_hover_text(info.id);
                    if btn.clicked() {
                        clicked = Some(Some(i));
                    }
                }
            });

        if let Some(index) = clicked {
            tools.activate(index);
        }
    }
}
