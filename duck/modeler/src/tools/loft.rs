use anyhow::Context;
use duck_engine_scene::resource::SubGeometryKind;
use duck_engine_viewer::{
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, MouseButton, NamedKey},
    operator::{SelectionKinds, SelectionMode},
    selection::{SelectionItem, SelectionManager},
};

use crate::ops::loft::{build_loft, LoftProfile};
use crate::preview::PreviewSession;
use crate::tools::{ModelingTool, PanelContext, ToolInfo, Workspace};
use crate::ui::icons;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum LoftPhase {
    #[default]
    Configuring,
    Done,
    Cancelled,
}

pub struct LoftTool {
    phase: LoftPhase,

    preview: PreviewSession,
    /// Profiles the current preview was built from, so we only rebuild on change.
    preview_profiles: Vec<LoftProfile>,

    workspace: Workspace,
}

impl LoftTool {
    pub fn new(workspace: &Workspace) -> Self {
        let preview = workspace.preview_session();
        Self {
            phase: LoftPhase::default(),
            preview,
            preview_profiles: Vec::new(),
            workspace: workspace.clone(),
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

        let result = build_loft(&self.workspace.document.lock().unwrap(), &profiles);
        let options = self.workspace.preview_options();
        match result {
            Ok(loft) => {
                if self.preview.add_preview_from_shape(&loft, &options, "Loft preview").is_none() {
                    log::warn!("Loft preview could not be tessellated");
                }
            }
            Err(e) => log::warn!("Loft preview failed: {e}"),
        }
    }

    /// Add the loft as a part of its own and clean up preview state. On success
    /// sets phase = Done; on failure stays in Configuring, preview and all, so
    /// the user can retry with different profiles.
    fn apply(&mut self) -> anyhow::Result<()> {
        let options = self.workspace.geometry_options();
        {
            let mut doc = self.workspace.document.lock().unwrap();
            let loft = build_loft(&doc, &self.preview_profiles)?;
            // Tessellates atomically — if this fails, nothing changes.
            doc.undo_scope("Loft").add_numbered_part("Loft", loft, &options).context("Failed to tessellate loft")?;
        }

        let _ = self.preview.commit();
        self.preview_profiles.clear();
        self.phase = LoftPhase::Done;
        Ok(())
    }

    /// Apply and, on success, clear the selection. Shared by the Enter key handler
    /// and the panel's Apply button.
    pub fn apply_and_clear(&mut self, selection: &mut SelectionManager) {
        if let Err(e) = self.apply() {
            self.workspace.notifications.failure("Loft", &e);
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
            let doc = self.workspace.document.lock().unwrap();
            profiles
                .iter()
                .map(|p| {
                    let name =
                        doc.part_at(p.node).map_or_else(|| "Unknown".to_owned(), |part| part.name.clone());
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

impl ModelingTool for LoftTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "loft", icon: icons::LOFT, shortcut: None }
    }

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
            // Right-click finalizes once there are profiles to skin.
            DeviceEvent::MouseClick { button: MouseButton::Right, .. } => {
                if self.preview_profiles.len() < 2 {
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
        self.phase = LoftPhase::Configuring;
    }

    /// Two or more profiles are a complete loft, so leaving the tool skins them
    /// rather than dropping the picks.
    fn finalize(&mut self, selection: &mut SelectionManager) -> anyhow::Result<()> {
        if self.preview_profiles.len() >= 2 {
            self.apply()?;
            selection.clear();
        }
        Ok(())
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
