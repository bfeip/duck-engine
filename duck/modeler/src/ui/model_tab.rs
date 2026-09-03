//! The model tab: a filter box over the list of parts in the CAD document,
//! with double-click-to-rename on each row.

use duck_engine_viewer::scene::resource::{NodeId, Visibility};
use duck_engine_viewer::selection::{SelectionItem, SelectionManager};

use crate::document::{Document, PartId, PartKind};
use crate::ui::icons;

/// The model tab, owning the state local to it.
#[derive(Default)]
pub struct ModelTab {
    filter: String,
    rename: Option<Rename>,
}

/// The row being renamed inline and its in-progress text.
struct Rename {
    part: PartId,
    text: String,
    /// Cleared once the field has been given focus for the first time.
    needs_focus: bool,
}

/// What the inline rename field asked for this frame.
enum RenameAction {
    None,
    Commit(String),
    Cancel,
}

/// A render-ready snapshot of one part row.
struct ModelRow {
    part_id: PartId,
    node: NodeId,
    name: String,
    kind: PartKind,
    selected: bool,
    visible: bool,
}

impl ModelTab {
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        document: &mut Document,
        selection: &mut SelectionManager,
    ) {
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.filter)
                    .hint_text("Filter objects…")
                    .desired_width(f32::INFINITY),
            );
        });
        ui.add_space(4.0);

        let search = self.filter.trim().to_lowercase();
        let rows: Vec<ModelRow> = document
            .parts()
            .filter(|part| search.is_empty() || part.name.to_lowercase().contains(&search))
            .filter_map(|part| {
                let node = document.node_for_part(part.id)?;
                Some(ModelRow {
                    part_id: part.id,
                    node,
                    name: part.name.clone(),
                    kind: part.kind(),
                    selected: selection.is_node_selected(node),
                    visible: document.part_visibility(part.id) != Some(Visibility::Invisible),
                })
            })
            .collect();

        // A rename whose row is gone — deleted, undone, or filtered out — is
        // abandoned rather than left editing an invisible part.
        if self.rename.as_ref().is_some_and(|rename| {
            !rows.iter().any(|row| row.part_id == rename.part)
        }) {
            self.rename = None;
        }

        egui::CollapsingHeader::new(format!("Model  ({})", rows.len()))
            .default_open(true)
            .show(ui, |ui| {
                if rows.is_empty() {
                    ui.add_space(4.0);
                    ui.weak("No objects yet");
                    return;
                }
                for row in &rows {
                    row_ui(ui, row, document, selection, &mut self.rename);
                }
            });
    }
}

fn row_ui(
    ui: &mut egui::Ui,
    row: &ModelRow,
    document: &mut Document,
    selection: &mut SelectionManager,
    rename: &mut Option<Rename>,
) {
    let accent = icons::kind_color(row.kind);
    let editing = rename.as_ref().is_some_and(|r| r.part == row.part_id);

    let mut frame = egui::Frame::default().inner_margin(egui::Margin::symmetric(6, 4));
    if row.selected {
        frame = frame.fill(ui.visuals().selection.bg_fill);
    }

    let mut eye_clicked = false;
    let mut rename_action = RenameAction::None;
    let inner = frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            let (kind_uri, kind_bytes) = icons::kind_icon(row.kind);
            ui.add(
                egui::Image::from_bytes(kind_uri, kind_bytes)
                    .fit_to_exact_size(egui::vec2(14.0, 14.0))
                    .tint(accent),
            );

            if let Some(state) = rename.as_mut().filter(|_| editing) {
                // Read Escape before the field runs: egui consumes it to
                // surrender focus, which would otherwise read as a commit.
                let cancelled = ui.input(|i| i.key_pressed(egui::Key::Escape));
                // Leave room for the eye and the kind badge to the right.
                let width = (ui.available_width() - 72.0).max(60.0);
                let field = ui.add(
                    egui::TextEdit::singleline(&mut state.text).desired_width(width),
                );
                if state.needs_focus {
                    state.needs_focus = false;
                    field.request_focus();
                }
                if cancelled {
                    field.surrender_focus();
                    rename_action = RenameAction::Cancel;
                } else if field.lost_focus() {
                    // Covers both Enter and clicking away.
                    rename_action = RenameAction::Commit(state.text.clone());
                }
            } else {
                let name = egui::RichText::new(&row.name);
                let name = if row.visible { name } else { name.weak() };
                ui.add(egui::Label::new(name).selectable(false).truncate());
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (eye_uri, eye_bytes) = if row.visible { icons::EYE } else { icons::EYE_OFF };
                let eye_tint = if row.visible {
                    ui.visuals().text_color()
                } else {
                    ui.visuals().weak_text_color()
                };
                let eye = ui.add(
                    egui::Button::image(
                        egui::Image::from_bytes(eye_uri, eye_bytes)
                            .fit_to_exact_size(egui::vec2(14.0, 14.0))
                            .tint(eye_tint),
                    )
                    .frame(false),
                );
                if eye.clicked() {
                    eye_clicked = true;
                }

                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(row.kind.label())
                        .small()
                        .color(accent.gamma_multiply(0.9)),
                );
            });
        });
    });

    // The eye lives inside the row rect, so resolve it first and let it win.
    if eye_clicked {
        let new = if row.visible { Visibility::Invisible } else { Visibility::Visible };
        document.set_part_visibility(row.part_id, new);
        return;
    }

    match rename_action {
        RenameAction::Commit(name) => {
            document.rename_part(row.part_id, name);
            *rename = None;
            return;
        }
        RenameAction::Cancel => {
            *rename = None;
            return;
        }
        RenameAction::None => {}
    }

    // The rename field owns the row rect while it is up; a click inside it
    // must not fall through to selection.
    if editing {
        return;
    }

    let row_resp = inner.response.interact(egui::Sense::click());
    if row_resp.double_clicked() {
        *rename = Some(Rename {
            part: row.part_id,
            text: row.name.clone(),
            needs_focus: true,
        });
        return;
    }
    if row_resp.clicked() {
        let item = SelectionItem::Node(row.node);
        let multi = ui.input(|i| i.modifiers.command || i.modifiers.shift);
        if multi {
            selection.toggle(item);
        } else {
            selection.set(item);
        }
    }
}
