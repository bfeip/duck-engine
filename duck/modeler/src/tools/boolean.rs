use duck_engine_scene::cad::{tessellate_into_with_materials, CadTessellationOptions};
use duck_engine_scene::common::RgbaColor;
use duck_engine_scene::resource::{FaceMaterial, LineMaterial, NodeFlags, NodeId};
use duck_engine_viewer::{
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, MouseButton, NamedKey},
    operator::{SelectionMode},
    selection::{SelectionItem, SelectionManager},
};
use opencascade::primitives::Shape;

use crate::ops::boolean::{build_boolean, commit_boolean, preview_boolean, BooleanKind, BooleanTarget};
use crate::preview::PreviewSession;
use crate::tools::{ModelingTool, PanelContext, ToolInfo, Workspace};
use crate::ui::icons;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum BooleanPhase {
    #[default]
    Configuring,
    Done,
    Cancelled,
}

pub struct BooleanTool {
    pub kind: BooleanKind,
    phase: BooleanPhase,

    preview: PreviewSession,

    /// The parts the preview was built for.
    target: Option<BooleanTarget>,
    last_kind: BooleanKind,

    workspace: Workspace,
}

impl BooleanTool {
    pub fn new(workspace: &Workspace) -> Self {
        let preview = workspace.preview_session();
        Self {
            kind: BooleanKind::default(),
            phase: BooleanPhase::default(),
            preview,
            target: None,
            last_kind: BooleanKind::default(),
            workspace: workspace.clone(),
        }
    }

    /// Execute the boolean operation and clean up preview state.
    /// On success sets phase = Done; on failure stays in Configuring so the user can retry.
    /// Call `selection.clear()` yourself on success.
    fn apply(&mut self) -> anyhow::Result<()> {
        let Some(target) = self.target.clone() else {
            return Ok(());
        };
        let options = self.workspace.geometry_options();
        {
            let mut doc = self.workspace.document.lock().unwrap();
            let result = build_boolean(&doc, &target, self.kind)?;
            commit_boolean(&mut doc, &target, result, &options)?;
        }

        // Remove the preview. The hidden sources it hands back were the boolean's
        // inputs, already deleted by commit_boolean; on failure above the session
        // stays live so the preview and hidden sources survive for retry/cancel.
        let _ = self.preview.commit();

        self.target = None;
        self.phase = BooleanPhase::Done;
        Ok(())
    }

    /// Apply and, on success, clear the selection. Shared by the Enter key
    /// handler and the panel's Apply button.
    pub fn apply_and_clear(&mut self, selection: &mut SelectionManager) {
        if let Err(e) = self.apply() {
            self.workspace.notifications.failure("Boolean", &e);
        } else {
            selection.clear();
        }
    }

    /// Abort the operation, restoring the visibility of all hidden original parts.
    pub fn cancel(&mut self) {
        self.preview.cancel();
        self.target = None;
        self.phase = BooleanPhase::Cancelled;
    }

    /// The boolean the selection designates: the primary part as the target,
    /// and the other selected parts as its tools.
    fn selection_snapshot(selection: &SelectionManager) -> Option<BooleanTarget> {
        let primary = selection.primary();
        let Some(SelectionItem::Node(target)) = primary else { return None };
        let tools = selection
            .iter()
            .filter(|&&item| Some(item) != primary)
            .filter_map(|item| match item {
                SelectionItem::Node(id) => Some(*id),
                _ => None,
            })
            .collect();
        Some(BooleanTarget { target, tools })
    }

    fn refresh_preview(&mut self, selection: &SelectionManager) {
        // Drop the old preview and re-show last pass's hidden sources.
        self.preview.clear_previews();

        self.target = Self::selection_snapshot(selection);
        self.last_kind = self.kind;
        let Some(target) = self.target.clone() else { return };

        let result = {
            let doc = self.workspace.document.lock().unwrap();
            preview_boolean(&doc, &target, self.kind)
        };
        let preview = match result {
            Ok(preview) => preview,
            Err(e) => {
                log::warn!("Boolean preview failed: {e}");
                return;
            }
        };
        let options = self.workspace.preview_options();
        if self.preview.add_preview_from_shape(&preview.shape, &options, "Boolean preview").is_none() {
            log::warn!("Boolean preview could not be tessellated");
            return;
        }
        self.preview.hide_source_node(target.target);
        for &tool in &target.tools {
            self.preview.hide_source_node(tool);
        }
        self.show_removed(&target.tools, &preview.removed, &options);
        // No preview is a part: picks must reach the hidden parts beneath.
        self.preview.set_preview_flags(NodeFlags::DO_NOT_SELECT);
    }

    /// Show the `removed` material in translucent red.
    fn show_removed(
        &mut self,
        tools: &[NodeId],
        removed: &[Shape],
        options: &CadTessellationOptions,
    ) {
        // A subtract removes its tools whole, and their meshes already exist.
        let (ghosted, pieces): (&[NodeId], &[Shape]) =
            if self.kind == BooleanKind::Subtract { (tools, &[]) } else { (&[], removed) };
        if ghosted.is_empty() && pieces.is_empty() {
            return;
        }

        let scene = self.workspace.document.lock().unwrap().scene().clone();
        let (face, line) = {
            let mut scene = scene.lock();
            (
                scene.add_face_material(removed_face_material()),
                scene.add_line_material(removed_line_material()),
            )
        };
        for &tool in ghosted {
            self.preview.ghost_source_node(tool, &face, &line);
        }
        for piece in pieces {
            match tessellate_into_with_materials(
                piece,
                &scene,
                options,
                &face,
                &line,
                None,
                Some("Boolean removed"),
            ) {
                Ok(node) => self.preview.add_preview_node(node.id()),
                Err(e) => log::warn!("Removed material could not be tessellated: {e}"),
            }
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
            let doc = self.workspace.document.lock().unwrap();
            let name_for = |node: NodeId| doc.part_at(node).map(|part| part.name.clone());
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

/// Translucent red for the material an operation removes. Single-sided: the
/// faces it shares with the result face the other way, so culling hides them
/// rather than letting them z-fight.
fn removed_face_material() -> FaceMaterial {
    FaceMaterial::new()
        .with_base_color_factor(RgbaColor { r: 0.85, g: 0.08, b: 0.12, a: 0.4 })
        .with_roughness_factor(0.32)
}

fn removed_line_material() -> LineMaterial {
    LineMaterial::new(RgbaColor { r: 0.6, g: 0.05, b: 0.08, a: 1.0 })
}

impl ModelingTool for BooleanTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "boolean", icon: icons::BOOLEAN, shortcut: None }
    }

    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::Update { .. } => {
                let selection_changed = Self::selection_snapshot(ctx.selection) != self.target;
                let kind_changed = self.kind != self.last_kind;
                if selection_changed || kind_changed {
                    self.refresh_preview(ctx.selection);
                }
                false
            }
            // Right-click finalizes a configured operation.
            DeviceEvent::MouseClick { button: MouseButton::Right, .. } => {
                if self.target.is_none() {
                    return false;
                }
                self.apply_and_clear(ctx.selection);
                true
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

    fn deactivate(&mut self) {
        self.cancel();
        self.phase = BooleanPhase::Configuring;
    }

    /// A picked target (with or without tools) is a configured operation, so
    /// leaving the tool runs it rather than dropping the configuration.
    fn finalize(&mut self, selection: &mut SelectionManager) -> anyhow::Result<()> {
        if self.target.is_some() {
            self.apply()?;
            // The sources the boolean consumed must not stay selected.
            selection.clear();
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::Scene;
    use glam::dvec3;

    use crate::document::Document;
    use crate::testing::workspace;

    /// Each kind shows what it removes exactly once: a subtract as a ghost of
    /// its tool, an intersect as the two removed pieces, a union not at all.
    #[test]
    fn preview_shows_each_kinds_removed_material_once() {
        let ws = workspace(Document::new(Scene::default()));
        let (target, tool) = {
            let options = ws.geometry_options();
            let mut doc = ws.document.lock().unwrap();
            let target = doc.add_part("box", Shape::cube(2.0), &options).unwrap();
            let sphere = Shape::sphere(1.0).at(dvec3(2.0, 2.0, 2.0)).build();
            let tool = doc.add_part("sphere", sphere, &options).unwrap();
            (doc.node_for_part(target).unwrap(), doc.node_for_part(tool).unwrap())
        };
        let mut selection = SelectionManager::new();
        selection.add(SelectionItem::Node(target));
        selection.add(SelectionItem::Node(tool));
        let mut op = BooleanTool::new(&ws);

        for (kind, removed_nodes) in
            [(BooleanKind::Subtract, 1), (BooleanKind::Intersect, 2), (BooleanKind::Union, 0)]
        {
            op.kind = kind;
            op.refresh_preview(&selection);

            // The result node plus the removed material.
            let previews = op.preview.preview_nodes();
            assert_eq!(previews.len(), 1 + removed_nodes);
            let scene = ws.document.lock().unwrap().scene().clone();
            let scene = scene.lock();
            for &node in previews {
                assert!(scene.get_node(node).unwrap().flags().contains(NodeFlags::DO_NOT_SELECT));
            }
        }
    }
}
