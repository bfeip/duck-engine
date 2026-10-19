//! The model tab: a filter box over the list of parts in the CAD document,
//! with double-click-to-rename on each row, above a summary of the selection.

use duck_engine_viewer::common::LengthDisplay;
use duck_engine_viewer::scene::resource::{NodeId, Visibility};
use duck_engine_viewer::selection::{SelectionItem, SelectionManager};

use crate::document::{Document, PartId, PartKind};
use crate::ui::icons;
use crate::ui::selection_info::SelectionInfo;

/// The model tab, owning the state local to it.
#[derive(Default)]
pub struct ModelTab {
    filter: String,
    rename: Option<Rename>,
    selection_info: SelectionInfo,
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
        units: LengthDisplay,
    ) {
        // Claimed first so it stays pinned below however long the list grows.
        egui::TopBottomPanel::bottom("selection_info")
            .resizable(false)
            .frame(egui::Frame::NONE.inner_margin(egui::Margin::symmetric(0, 4)))
            .show_inside(ui, |ui| self.selection_info.show(ui, document, selection, units));

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

        egui::ScrollArea::vertical().show(ui, |ui| {
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
    // The row senses clicks beneath its widgets, so the eye and the rename
    // field keep theirs.
    let row_resp = ui.scope_builder(egui::UiBuilder::new().sense(egui::Sense::click()), |ui| {
        frame.show(ui, |ui| {
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
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(row.kind.label())
                                .small()
                                .color(accent.gamma_multiply(0.9)),
                        )
                        .selectable(false),
                    );
                });
            });
        });
    })
    .response;

    if eye_clicked {
        // A hidden part can't be picked, so it doesn't stay selected either.
        let new = if row.visible {
            selection.remove_node(row.node);
            Visibility::Invisible
        } else {
            Visibility::Visible
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::doc_with_box;

    /// What one frame of the tab painted, with where.
    struct Painted {
        texts: Vec<(String, egui::Rect)>,
        images: Vec<egui::Rect>,
    }

    impl Painted {
        /// Where `text` is painted highest, which puts the list above the
        /// selection summary.
        fn text(&self, text: &str) -> egui::Rect {
            self.texts
                .iter()
                .filter(|(painted, _)| painted == text)
                .map(|(_, rect)| *rect)
                .min_by(|a, b| a.min.y.total_cmp(&b.min.y))
                .unwrap_or_else(|| panic!("{text:?} is painted"))
        }

        /// The eye: the rightmost image level with the part's kind badge.
        fn eye(&self) -> egui::Rect {
            let y = self.text("SOLID").center().y;
            self.images
                .iter()
                .filter(|rect| rect.y_range().contains(y))
                .max_by(|a, b| a.center().x.total_cmp(&b.center().x))
                .copied()
                .expect("the eye is painted")
        }
    }

    /// The model tab over a box, with the state one frame needs.
    struct Harness {
        ctx: egui::Context,
        tab: ModelTab,
        document: Document,
        node: NodeId,
        selection: SelectionManager,
    }

    impl Harness {
        fn new() -> Self {
            let ctx = egui::Context::default();
            egui_extras::install_image_loaders(&ctx);
            let (document, node) = doc_with_box();
            let mut harness = Self {
                ctx,
                tab: ModelTab::default(),
                document,
                node,
                selection: SelectionManager::new(),
            };
            // Hit-testing reads the last frame's widgets.
            harness.draw(None);
            harness
        }

        fn draw(&mut self, event: Option<egui::Event>) -> Painted {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(320.0, 600.0),
                )),
                events: event.into_iter().collect(),
                ..Default::default()
            };
            let output = self.ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    self.tab.show(ui, &mut self.document, &mut self.selection, LengthDisplay::default());
                });
            });
            let mut painted = Painted { texts: Vec::new(), images: Vec::new() };
            for clipped in &output.shapes {
                match &clipped.shape {
                    egui::epaint::Shape::Text(text) => painted.texts.push((
                        text.galley.text().to_owned(),
                        text.galley.rect.translate(text.pos.to_vec2()),
                    )),
                    egui::epaint::Shape::Rect(rect)
                        if rect.fill_texture_id() != egui::TextureId::default() =>
                    {
                        painted.images.push(rect.rect);
                    }
                    _ => {}
                }
            }
            painted
        }

        fn click(&mut self, at: egui::Pos2) {
            let press = |pressed| egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            };
            for event in [egui::Event::PointerMoved(at), press(true), press(false)] {
                self.draw(Some(event));
            }
        }

        fn visibility(&self) -> Option<Visibility> {
            let part = self.document.part_for_node(self.node).expect("node has a part");
            self.document.part_visibility(part)
        }
    }

    #[test]
    fn clicking_the_eye_toggles_the_part_without_selecting_it() {
        let mut harness = Harness::new();

        let eye = harness.draw(None).eye();
        harness.click(eye.center());
        assert_eq!(harness.visibility(), Some(Visibility::Invisible));
        assert!(harness.selection.is_empty(), "the eye is not a row click");

        let eye = harness.draw(None).eye();
        harness.click(eye.center());
        assert_eq!(harness.visibility(), Some(Visibility::Visible));
    }

    #[test]
    fn hiding_a_selected_part_deselects_it() {
        let mut harness = Harness::new();
        harness.selection.set(SelectionItem::Node(harness.node));

        let eye = harness.draw(None).eye();
        harness.click(eye.center());
        assert_eq!(harness.visibility(), Some(Visibility::Invisible));
        assert!(harness.selection.is_empty());
    }

    #[test]
    fn clicking_a_row_selects_its_part() {
        for label in ["part", "SOLID"] {
            let mut harness = Harness::new();
            let at = harness.draw(None).text(label).center();
            harness.click(at);
            assert!(harness.selection.is_node_selected(harness.node), "clicking {label:?} selects");
        }
    }
}
