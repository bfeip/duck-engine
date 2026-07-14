use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_scene::NodeId;
use duck_engine_viewer::{
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, NamedKey},
    operator::{Operator, SelectionMode},
    selection::{SelectionItem, SelectionManager},
};

use crate::boolean::{execute_boolean, preview_boolean, BooleanKind};
use crate::document::Document;
use crate::notifications::Notifications;
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, PanelContext, ToolInfo};
use crate::ui::icons;
use super::ConstructionOptions;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum BooleanPhase {
    #[default]
    Configuring,
    Done,
    Cancelled,
}

pub struct BooleanOperator {
    pub kind: BooleanKind,
    phase: BooleanPhase,

    preview: PreviewSession,

    preview_target: Option<NodeId>,
    preview_tools: Vec<NodeId>,
    last_kind: BooleanKind,

    document: Arc<Mutex<Document>>,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    notifications: Notifications,
}

impl BooleanOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
        notifications: Notifications,
    ) -> Self {
        let preview = PreviewSession::new(Arc::clone(&document));
        Self {
            kind: BooleanKind::default(),
            phase: BooleanPhase::default(),
            preview,
            preview_target: None,
            preview_tools: Vec::new(),
            last_kind: BooleanKind::default(),
            document,
            construction_options,
            notifications,
        }
    }

    /// Execute the boolean operation and clean up preview state.
    /// On success sets phase = Done; on failure stays in Configuring so the user can retry.
    /// Call `selection.clear()` yourself on success.
    fn apply(&mut self) -> anyhow::Result<()> {
        let Some(target) = self.preview_target else {
            return Ok(());
        };
        let tools = self.preview_tools.clone();

        let options = self.construction_options.borrow().geometry_options.clone();

        // Remove the preview; the hidden originals are handed to execute_boolean,
        // which deletes them (target + tools), so they stay hidden meanwhile.
        let _ = self.preview.commit();

        let mut doc = self.document.lock().unwrap();
        execute_boolean(self.kind, target, &tools, &mut *doc, &options)?;
        drop(doc);

        self.preview_target = None;
        self.preview_tools.clear();
        self.phase = BooleanPhase::Done;
        Ok(())
    }

    /// Apply and, on success, clear the selection. Shared by the Enter key
    /// handler and the panel's Apply button.
    pub fn apply_and_clear(&mut self, selection: &mut SelectionManager) {
        if let Err(e) = self.apply() {
            log::error!("Boolean failed: {e}");
            self.notifications.error(format!("Boolean failed: {e}"));
        } else {
            selection.clear();
        }
    }

    /// Abort the operation, restoring the visibility of all hidden original parts.
    pub fn cancel(&mut self) {
        self.preview.cancel();
        self.preview_target = None;
        self.preview_tools.clear();
        self.phase = BooleanPhase::Cancelled;
    }

    fn selection_snapshot(selection: &SelectionManager) -> (Option<NodeId>, Vec<NodeId>) {
        let primary = selection.primary();
        let target = primary.and_then(|item| match item {
            SelectionItem::Node(id) => Some(id),
            _ => None,
        });
        let tools: Vec<_> = selection.iter()
            .filter(|&&item| Some(item) != primary)
            .filter_map(|item| match item {
                SelectionItem::Node(id) => Some(*id),
                _ => None,
            })
            .collect();
        (target, tools)
    }

    fn refresh_preview(&mut self, selection: &SelectionManager) {
        // Drop the old preview and re-show last pass's hidden sources.
        self.preview.clear_previews();

        let (target, tools) = Self::selection_snapshot(selection);
        self.preview_target = target;
        self.preview_tools = tools.clone();
        self.last_kind = self.kind;

        let Some(target_node) = target else { return };

        let options = self.construction_options.borrow().preview_options();
        let result = {
            let doc = self.document.lock().unwrap();
            preview_boolean(self.kind, target_node, &tools, &*doc, &options)
        };

        match result {
            Ok(preview) => {
                self.preview.add_preview_node(preview);
                self.preview.hide_source_node(target_node);
                for tool in tools {
                    self.preview.hide_source_node(tool);
                }
            }
            Err(e) => log::warn!("Boolean preview failed: {e}"),
        }
    }

    /// The boolean configuration panel body (operation kind, target/tool parts,
    /// Apply/Cancel).
    fn render_panel(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        let mut apply_clicked = false;
        let mut cancel_clicked = false;

        let primary = panel.selection.primary();
        let target_node = primary.and_then(|item| match item {
            SelectionItem::Node(id) => Some(id),
            _ => None,
        });
        let tool_items: Vec<SelectionItem> = panel.selection.iter()
            .filter(|&&item| Some(item) != primary)
            .copied()
            .collect();

        let (target_name, tool_entries) = {
            let doc = self.document.lock().unwrap();
            let name_for = |node: NodeId| {
                doc.part_for_node(node)
                    .and_then(|p| doc.get_part(p).map(|part| part.name.clone()))
            };
            let target_name = target_node
                .and_then(name_for)
                .unwrap_or_else(|| "(none — click a part)".to_owned());
            let tool_entries: Vec<(SelectionItem, String)> = tool_items
                .into_iter()
                .map(|item| {
                    let name = match item {
                        SelectionItem::Node(id) => name_for(id),
                        _ => None,
                    }
                    .unwrap_or_else(|| "Unknown".to_owned());
                    (item, name)
                })
                .collect();
            (target_name, tool_entries)
        };

        ui.label("Operation");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.kind, BooleanKind::Subtract, "Subtract");
            ui.selectable_value(&mut self.kind, BooleanKind::Union, "Union");
            ui.selectable_value(&mut self.kind, BooleanKind::Intersect, "Intersect");
        });

        ui.separator();

        ui.label("Target");
        ui.label(&target_name);

        ui.separator();

        ui.label("Tools");
        if tool_entries.is_empty() {
            ui.label("(shift-click parts to add tools)");
        } else {
            for (item, name) in &tool_entries {
                ui.horizontal(|ui| {
                    ui.label(name);
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

        // Act after rendering the body so we don't call &mut self methods
        // while the widgets above still borrow self.
        if apply_clicked {
            self.apply_and_clear(panel.selection);
        } else if cancel_clicked {
            self.cancel();
        }
    }
}

impl ModelingTool for BooleanOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "boolean", icon: icons::BOOLEAN, shortcut: None }
    }

    fn deactivate(&mut self) {
        self.cancel();
        self.phase = BooleanPhase::Configuring;
    }

    fn is_finished(&self) -> bool {
        matches!(self.phase, BooleanPhase::Done | BooleanPhase::Cancelled)
    }

    // Boolean operates on whole parts, so drop the always-on selection
    // operator to node granularity while active.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::Node
    }

    fn panel_title(&self) -> Option<&str> {
        Some("Boolean Operation")
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        self.render_panel(ui, panel);
    }
}

impl Operator for BooleanOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::Update { .. } => {
                let (current_target, current_tools) = Self::selection_snapshot(ctx.selection);
                let selection_changed = current_target != self.preview_target
                    || current_tools != self.preview_tools;
                let kind_changed = self.kind != self.last_kind;
                if selection_changed || kind_changed {
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
        "Boolean"
    }
}
