use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_scene::resource::SubGeometryKind;
use duck_engine_viewer::{
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, NamedKey},
    operator::{Operator, SelectionKinds, SelectionMode},
    selection::{SelectionItem, SelectionManager},
};

use crate::document::Document;
use crate::loft::{execute_loft, preview_loft, LoftProfile};
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, PanelContext, ToolInfo};
use crate::ui::icons;
use super::ConstructionOptions;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum LoftPhase {
    #[default]
    Configuring,
    Done,
    Cancelled,
}

pub struct LoftOperator {
    phase: LoftPhase,

    preview: PreviewSession,
    /// Profiles the current preview was built from, so we only rebuild on change.
    preview_profiles: Vec<LoftProfile>,

    document: Arc<Mutex<Document>>,
    construction_options: Rc<RefCell<ConstructionOptions>>,
}

impl LoftOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let preview = PreviewSession::new(Arc::clone(&document));
        Self {
            phase: LoftPhase::default(),
            preview,
            preview_profiles: Vec::new(),
            document,
            construction_options,
        }
    }

    /// The selected edges, in click order, as loft profiles.
    fn selection_snapshot(selection: &SelectionManager) -> Vec<LoftProfile> {
        selection
            .iter()
            .filter_map(|item| match item {
                SelectionItem::SubGeometry { node_id, element }
                    if element.kind == SubGeometryKind::Edge =>
                {
                    Some(LoftProfile { node: *node_id, edge_index: element.index })
                }
                _ => None,
            })
            .collect()
    }

    fn refresh_preview(&mut self, selection: &SelectionManager) {
        self.preview.clear_previews();

        let profiles = Self::selection_snapshot(selection);
        self.preview_profiles = profiles.clone();

        if profiles.len() < 2 {
            return;
        }

        let options = self.construction_options.borrow().preview_options();
        let result = {
            let doc = self.document.lock().unwrap();
            preview_loft(&doc, &profiles, &options)
        };
        match result {
            Ok(node) => self.preview.add_preview_node(node),
            Err(e) => log::warn!("Loft preview failed: {e}"),
        }
    }

    /// Execute the loft and clean up preview state. On success sets phase = Done;
    /// on failure stays in Configuring so the user can retry with different profiles.
    fn apply(&mut self) -> anyhow::Result<()> {
        let profiles = self.preview_profiles.clone();
        let options = self.construction_options.borrow().geometry_options.clone();

        let _ = self.preview.commit();

        let mut doc = self.document.lock().unwrap();
        execute_loft(&mut doc, &profiles, &options)?;
        drop(doc);

        self.preview_profiles.clear();
        self.phase = LoftPhase::Done;
        Ok(())
    }

    /// Apply and, on success, clear the selection. Shared by the Enter key handler
    /// and the panel's Apply button.
    pub fn apply_and_clear(&mut self, selection: &mut SelectionManager) {
        if let Err(e) = self.apply() {
            log::error!("Loft failed: {e}");
        } else {
            selection.clear();
        }
    }

    /// Abort: drop the preview (profiles are construction curves, never hidden).
    pub fn cancel(&mut self) {
        self.preview.cancel();
        self.preview_profiles.clear();
        self.phase = LoftPhase::Cancelled;
    }

    /// The loft configuration panel body (ordered profile list, Apply/Cancel).
    fn render_panel(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        let mut apply_clicked = false;
        let mut cancel_clicked = false;

        // Snapshot profile names under the document lock so the body holds none.
        let profiles = Self::selection_snapshot(panel.selection);
        let entries: Vec<(SelectionItem, String)> = {
            let doc = self.document.lock().unwrap();
            profiles
                .iter()
                .map(|p| {
                    let name = doc
                        .part_for_node(p.node)
                        .and_then(|id| doc.get_part(id).map(|part| part.name.clone()))
                        .unwrap_or_else(|| "Unknown".to_owned());
                    let item = SelectionItem::SubGeometry {
                        node_id: p.node,
                        element: duck_engine_scene::resource::SubGeometryElement::new(
                            SubGeometryKind::Edge,
                            p.edge_index,
                        ),
                    };
                    (item, format!("{name} · edge {}", p.edge_index))
                })
                .collect()
        };

        ui.label("Profiles");
        if entries.is_empty() {
            ui.label("(click an edge per profile)");
        } else {
            for (idx, (item, name)) in entries.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("{}. {name}", idx + 1));
                    if ui.small_button("×").clicked() {
                        panel.selection.remove(item);
                    }
                });
            }
        }

        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Cancel").clicked() {
                cancel_clicked = true;
            }
            if ui.button("Apply  ⏎").clicked() {
                apply_clicked = true;
            }
        });

        // Act after rendering the body so we don't call &mut self methods while
        // the widgets above still borrow self.
        if apply_clicked {
            self.apply_and_clear(panel.selection);
        } else if cancel_clicked {
            self.cancel();
        }
    }
}

impl ModelingTool for LoftOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "loft", icon: icons::LOFT, shortcut: None }
    }

    fn deactivate(&mut self) {
        self.cancel();
        self.phase = LoftPhase::Configuring;
    }

    fn is_finished(&self) -> bool {
        matches!(self.phase, LoftPhase::Done | LoftPhase::Cancelled)
    }

    // Loft skins through profile edges, so select at edge granularity.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::SubGeometry(SelectionKinds::EDGE)
    }

    fn panel_title(&self) -> Option<&str> {
        Some("Loft")
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        self.render_panel(ui, panel);
    }
}

impl Operator for LoftOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::Update { .. } => {
                let profiles = Self::selection_snapshot(ctx.selection);
                if profiles != self.preview_profiles {
                    self.refresh_preview(ctx.selection);
                }
                false
            }
            DeviceEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state != ElementState::Pressed || key_event.repeat {
                    return false;
                }
                match key_event.logical_key {
                    Key::Named(NamedKey::Enter) => {
                        self.apply_and_clear(ctx.selection);
                        true
                    }
                    Key::Named(NamedKey::Escape) => {
                        self.cancel();
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Loft"
    }
}
