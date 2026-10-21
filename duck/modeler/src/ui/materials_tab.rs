//! The materials tab: the document's material library, with an Assign button
//! per material for the selected bodies, above an editor for the open one.

use duck_engine_viewer::scene::resource::FaceMaterialId;
use duck_engine_viewer::selection::SelectionManager;

use crate::document::{Document, PartId};
use crate::materials::{Finish, DEFAULT_FINISH};
use crate::tools::selected_parts;

/// Side of a material's swatch, in points.
const SWATCH_SIZE: f32 = 14.0;

/// The materials tab, owning the state local to it.
#[derive(Default)]
pub struct MaterialsTab {
    /// The material in the editor.
    open: Option<FaceMaterialId>,
    /// The editor's name field, while it has focus.
    name: String,
}

/// A render-ready snapshot of one library row.
struct MaterialRow {
    id: FaceMaterialId,
    name: String,
    finish: Finish,
    users: usize,
}

/// What a library row asked for this frame.
enum RowAction {
    None,
    Open,
    Assign,
}

impl MaterialsTab {
    pub fn show(&mut self, ui: &mut egui::Ui, document: &mut Document, selection: &SelectionManager) {
        let selected: Vec<PartId> = selected_parts(selection, document)
            .into_iter()
            .filter_map(|node| document.part_for_node(node))
            .collect();
        let bodies: Vec<PartId> =
            selected.iter().copied().filter(|&part| document.takes_material(part)).collect();
        let on: Vec<Option<FaceMaterialId>> =
            bodies.iter().map(|&part| document.part_material(part)).collect();
        let usage = document.material_usage();
        let rows: Vec<MaterialRow> = document
            .materials()
            .iter()
            .map(|material| MaterialRow {
                id: material.id(),
                name: material.name.clone(),
                finish: document.material_finish(material.id()).unwrap_or(DEFAULT_FINISH),
                users: usage.get(&material.id()).copied().unwrap_or(0),
            })
            .collect();

        // A material that has left the library — deleted, or its creation
        // undone — closes.
        if self.open.is_some_and(|id| !rows.iter().any(|row| row.id == id)) {
            self.open = None;
        }
        let open = self.open.and_then(|id| rows.iter().find(|row| row.id == id));

        // Claimed first so it stays pinned below however long the list grows.
        egui::TopBottomPanel::bottom("material_editor")
            .resizable(false)
            .frame(egui::Frame::NONE.inner_margin(egui::Margin::symmetric(0, 4)))
            .show_inside(ui, |ui| match open {
                Some(row) => editor_ui(ui, document, row, &mut self.name),
                None => {
                    ui.weak("Click a material to edit it");
                }
            });

        selection_line(ui, &rows, &selected, &on);
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("+ New").clicked() {
                let finish = open.map_or(DEFAULT_FINISH, |row| row.finish);
                let name = document.numbered_material_name("Material");
                self.open = Some(document.add_material(name, finish));
            }
            let users = open.map_or(0, |row| row.users);
            let delete = ui
                .add_enabled(open.is_some() && users == 0, egui::Button::new("Delete"))
                .on_disabled_hover_text(match (open, users) {
                    (None, _) => "Click a material to delete it".to_owned(),
                    (Some(_), users) => format!("Used by {}", part_count(users)),
                });
            if let Some(row) = open.filter(|_| delete.clicked()) {
                match document.remove_material(row.id) {
                    Ok(()) => self.open = None,
                    Err(e) => log::warn!("Failed to delete {}: {e:#}", row.name),
                }
            }
        });
        ui.add_space(4.0);

        let mut action = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            for row in &rows {
                let can_assign = on.iter().any(|&material| material != Some(row.id));
                match row_ui(ui, row, self.open == Some(row.id), can_assign) {
                    RowAction::None => {}
                    act => action = Some((row.id, act)),
                }
            }
        });
        match action {
            Some((id, RowAction::Assign)) => document.assign_material(&bodies, id),
            Some((id, RowAction::Open)) => {
                self.open = Some(id);
                // The editor above was drawn for the material open before.
                ui.ctx().request_repaint();
            }
            _ => {}
        }
    }
}

/// What the selection is made of, and the material on it.
fn selection_line(
    ui: &mut egui::Ui,
    rows: &[MaterialRow],
    selected: &[PartId],
    on: &[Option<FaceMaterialId>],
) {
    let Some(&first) = on.first() else {
        let hint = if selected.is_empty() {
            "Select bodies to assign a material"
        } else {
            "Sketches keep their sketch appearance"
        };
        ui.weak(hint);
        return;
    };
    let material = if on.iter().any(|&material| material != first) {
        "Mixed"
    } else {
        first
            .and_then(|id| rows.iter().find(|row| row.id == id))
            .map_or("Own appearance", |row| row.name.as_str())
    };
    let bodies = match on.len() {
        1 => "1 body".to_owned(),
        count => format!("{count} bodies"),
    };
    ui.label(format!("{bodies} · {material}"));
}

fn row_ui(ui: &mut egui::Ui, row: &MaterialRow, open: bool, can_assign: bool) -> RowAction {
    let mut frame = egui::Frame::default().inner_margin(egui::Margin::symmetric(6, 4));
    if open {
        frame = frame.fill(ui.visuals().selection.bg_fill);
    }

    let mut assign = false;
    // The row senses clicks beneath its widgets, so Assign keeps its own.
    let row_resp = ui.scope_builder(egui::UiBuilder::new().sense(egui::Sense::click()), |ui| {
        frame.show(ui, |ui| {
            ui.horizontal(|ui| {
                swatch(ui, row.finish);
                ui.add(egui::Label::new(&row.name).selectable(false).truncate());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let button = egui::Button::new("Assign").small();
                    if ui.add_enabled(can_assign, button).clicked() {
                        assign = true;
                    }
                    if row.users > 0 {
                        let users = egui::RichText::new(part_count(row.users)).small().weak();
                        ui.add(egui::Label::new(users).selectable(false));
                    }
                });
            });
        });
    })
    .response;

    if assign {
        RowAction::Assign
    } else if row_resp.clicked() {
        RowAction::Open
    } else {
        RowAction::None
    }
}

/// A square of the finish's color, translucent as the material is.
fn swatch(ui: &mut egui::Ui, finish: Finish) {
    let (rect, _) = ui.allocate_exact_size(egui::Vec2::splat(SWATCH_SIZE), egui::Sense::hover());
    let color = finish.color;
    let fill = egui::Rgba::from_rgba_unmultiplied(color.r, color.g, color.b, color.a);
    let painter = ui.painter();
    painter.rect_filled(rect, 3.0, fill);
    let stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    painter.rect_stroke(rect, 3.0, stroke, egui::StrokeKind::Inside);
}

/// "1 part", "3 parts".
fn part_count(count: usize) -> String {
    match count {
        1 => "1 part".to_owned(),
        count => format!("{count} parts"),
    }
}

/// The open material's name and finish. `name` holds the name field's text
/// while it has focus.
fn editor_ui(ui: &mut egui::Ui, document: &mut Document, row: &MaterialRow, name: &mut String) {
    let mut finish = row.finish;
    let mut changed = false;
    egui::Grid::new("material_editor_grid").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
        ui.label("Name");
        let field_id = egui::Id::new("material_name");
        let editing = ui.memory(|memory| memory.has_focus(field_id));
        if !editing {
            name.clone_from(&row.name);
        }
        // Read Escape before the field runs: egui consumes it to surrender
        // focus, which would otherwise read as a commit.
        let cancelled = editing && ui.input(|i| i.key_pressed(egui::Key::Escape));
        let field = ui.add(egui::TextEdit::singleline(name).id(field_id).desired_width(f32::INFINITY));
        if cancelled {
            field.surrender_focus();
            name.clone_from(&row.name);
        } else if field.lost_focus() {
            // Covers both Enter and clicking away.
            document.rename_material(row.id, name);
        }
        ui.end_row();

        ui.label("Color");
        let mut rgb = [finish.color.r, finish.color.g, finish.color.b];
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            [finish.color.r, finish.color.g, finish.color.b] = rgb;
            changed = true;
        }
        ui.end_row();

        for (label, value) in [
            ("Opacity", &mut finish.color.a),
            ("Metallic", &mut finish.metallic),
            ("Roughness", &mut finish.roughness),
        ] {
            ui.label(label);
            changed |= ui.add(egui::Slider::new(value, 0.0..=1.0)).changed();
            ui.end_row();
        }
    });
    if changed {
        document.set_material_finish(row.id, finish);
    }
}

#[cfg(test)]
mod tests {
    use duck_engine_viewer::scene::resource::NodeId;
    use duck_engine_viewer::selection::SelectionItem;
    use opencascade::primitives::Shape;

    use super::*;
    use crate::testing::{doc_with_shape, xy_region};

    /// The texts one frame of the tab painted, with where.
    struct Painted {
        texts: Vec<(String, egui::Rect)>,
    }

    impl Painted {
        fn text(&self, text: &str) -> egui::Rect {
            self.texts
                .iter()
                .find(|(painted, _)| painted == text)
                .map(|(_, rect)| *rect)
                .unwrap_or_else(|| panic!("{text:?} is painted"))
        }

        fn has(&self, text: &str) -> bool {
            self.texts.iter().any(|(painted, _)| painted == text)
        }

        /// `text` on the row of the material named `row`.
        fn on_row(&self, row: &str, text: &str) -> egui::Rect {
            let y = self.text(row).center().y;
            self.texts
                .iter()
                .find(|(painted, rect)| painted == text && rect.y_range().contains(y))
                .map(|(_, rect)| *rect)
                .unwrap_or_else(|| panic!("{text:?} is painted on {row:?}'s row"))
        }
    }

    /// The materials tab over a document holding one part, selected, and
    /// the preset library.
    struct Harness {
        ctx: egui::Context,
        tab: MaterialsTab,
        document: Document,
        node: NodeId,
        selection: SelectionManager,
    }

    impl Harness {
        fn new(shape: Shape) -> Self {
            let (mut document, node) = doc_with_shape(shape);
            document.add_preset_materials();
            let mut selection = SelectionManager::new();
            selection.set(SelectionItem::Node(node));
            let mut harness =
                Self { ctx: egui::Context::default(), tab: MaterialsTab::default(), document, node, selection };
            // Hit-testing reads the last frame's widgets.
            harness.draw(None);
            harness
        }

        fn with_box() -> Self {
            Self::new(Shape::cube(2.0))
        }

        fn draw(&mut self, event: Option<egui::Event>) -> Painted {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(320.0, 600.0))),
                events: event.into_iter().collect(),
                ..Default::default()
            };
            let output = self.ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    self.tab.show(ui, &mut self.document, &self.selection);
                });
            });
            let texts = output
                .shapes
                .iter()
                .filter_map(|clipped| match &clipped.shape {
                    egui::epaint::Shape::Text(text) => Some((
                        text.galley.text().to_owned(),
                        text.galley.rect.translate(text.pos.to_vec2()),
                    )),
                    _ => None,
                })
                .collect();
            Painted { texts }
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

        fn material(&self, name: &str) -> FaceMaterialId {
            let material = self.document.materials().iter().find(|material| material.name == name);
            material.unwrap_or_else(|| panic!("{name:?} is in the library")).id()
        }

        fn part_material(&self) -> Option<FaceMaterialId> {
            self.document.part_material(self.document.part_for_node(self.node).unwrap())
        }
    }

    #[test]
    fn assign_puts_the_material_on_the_selected_body() {
        let mut harness = Harness::with_box();
        assert!(harness.draw(None).has("1 body · Own appearance"));

        let assign = harness.draw(None).on_row("Steel", "Assign");
        harness.click(assign.center());
        assert_eq!(harness.part_material(), Some(harness.material("Steel")));
        let painted = harness.draw(None);
        assert!(painted.has("1 body · Steel"));
        assert!(painted.has("1 part"), "the row counts its user");
    }

    #[test]
    fn clicking_a_row_opens_it_without_assigning() {
        let mut harness = Harness::with_box();
        let at = harness.draw(None).text("Brass").center();
        harness.click(at);
        assert_eq!(harness.tab.open, Some(harness.material("Brass")));
        assert_eq!(harness.part_material(), None);
    }

    #[test]
    fn a_material_in_use_cannot_be_deleted() {
        let mut harness = Harness::with_box();
        let steel = harness.material("Steel");
        let part = harness.document.part_for_node(harness.node).unwrap();
        harness.document.assign_material(&[part], steel);
        harness.tab.open = Some(steel);

        let delete = harness.draw(None).text("Delete");
        harness.click(delete.center());
        assert!(harness.document.materials().contains(steel));

        harness.document.undo().expect("undo the assignment");
        let delete = harness.draw(None).text("Delete");
        harness.click(delete.center());
        assert!(!harness.document.materials().contains(steel), "unused, it deletes");
        assert_eq!(harness.tab.open, None, "a deleted material closes");
    }

    #[test]
    fn new_copies_the_open_material() {
        let mut harness = Harness::with_box();
        let glass = harness.material("Glass");
        harness.tab.open = Some(glass);

        let new = harness.draw(None).text("+ New");
        harness.click(new.center());
        let copy = harness.material("Material-001");
        assert_eq!(harness.tab.open, Some(copy));
        assert_eq!(harness.document.material_finish(copy), harness.document.material_finish(glass));
        assert_eq!(harness.document.undo_label(), Some("New Material-001"));
    }

    #[test]
    fn a_sketch_keeps_its_appearance() {
        let mut harness = Harness::new(xy_region(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]));
        assert!(harness.draw(None).has("Sketches keep their sketch appearance"));

        let assign = harness.draw(None).on_row("Steel", "Assign");
        harness.click(assign.center());
        assert_eq!(harness.part_material(), None);
    }
}
