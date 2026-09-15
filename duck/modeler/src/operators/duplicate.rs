use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_scene::resource::{NodeFlags, NodeId};
use duck_engine_scene::Scene;
use duck_engine_viewer::{
    common::{decompose_matrix, Matrix4, SquareMatrix, Transform},
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, MouseButton, NamedKey},
    operator::{
        NodeTransformTarget, Operator, SelectionMode, TransformDriver, TransformFrame,
        TransformInteraction, TransformMode, TransformTarget,
    },
    selection::{SelectionItem, SelectionManager},
};

use crate::document::{Document, PartId};
use crate::duplicate::duplicate_parts;
use crate::notifications::Notifications;
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, ToolInfo};
use crate::ui::icons;
use super::ConstructionOptions;

/// Copies the selected parts and places the copies with the translate gizmo.
///
/// Selecting the tool immediately makes a copy of every selected part, sitting
/// on top of its source, and shows the translate handles. Drag a handle (or
/// press G) to carry the copies off; left-click or Enter confirms, right-click
/// or Escape reverts the drag. With no drag in progress, right-click or Enter
/// places the copies where they already are and Escape leaves without copying
/// anything. Either way the tool then cedes back to selection.
///
/// The copies are only preview geometry until they are placed, so nothing
/// reaches the document unless the user confirms — and then the whole
/// placement is a single undo step.
pub struct DuplicateTool {
    driver: TransformDriver<DuplicateTarget>,
    /// Held only to reach the scene on [`ModelingTool::deactivate`], which has
    /// no event context.
    document: Arc<Mutex<Document>>,
}

impl DuplicateTool {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
        notifications: Notifications,
    ) -> Self {
        let target = DuplicateTarget {
            nodes: NodeTransformTarget::new(),
            preview: PreviewSession::new(Arc::clone(&document)),
            sources: Vec::new(),
            pending_spawn: false,
            done: false,
            construction_options,
            document: Arc::clone(&document),
            notifications,
        };
        Self {
            driver: TransformDriver::with_target(TransformMode::Translate, target),
            document,
        }
    }

    /// Dispatches an event so long as a drag is not active.
    /// 
    /// Returns `true` if an action was taken.
    fn dispatch_idle(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        if self.driver.is_active() {
            return false;
        }
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseClick { button: MouseButton::Right, .. } => self.place(ctx),
            DeviceEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state != ElementState::Pressed || key_event.repeat {
                    return false;
                }
                match key_event.logical_key {
                    Key::Named(NamedKey::Enter) => self.place(ctx),
                    Key::Named(NamedKey::Escape) => self.abandon(),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Commit the copies at their current, undragged position.
    fn place(&mut self, ctx: &mut EventContext) -> bool {
        let target = self.driver.target_mut();
        if target.sources.is_empty() {
            return false;
        }
        target.apply(Matrix4::identity(), ctx);
        true
    }

    /// Drop the copies without adding anything to the document.
    fn abandon(&mut self) -> bool {
        let target = self.driver.target_mut();
        if target.sources.is_empty() {
            return false;
        }
        target.discard();
        target.done = true;
        true
    }
}

impl Operator for DuplicateTool {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        // The copies are built from the selection, which `activate` cannot see;
        // the frame tick is the first point with an event context.
        if let Event::Device(DeviceEvent::Update { .. }) = event {
            self.driver.target_mut().sync_sources(ctx);
        }

        self.driver.dispatch(event, ctx) || self.dispatch_idle(event, ctx)
    }

    fn name(&self) -> &str {
        "DuplicateTool"
    }
}

impl ModelingTool for DuplicateTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "duplicate", icon: icons::DUPLICATE, shortcut: Some('d') }
    }

    fn activate(&mut self) {
        self.driver.set_gizmo_enabled(true);
        self.driver.target_mut().pending_spawn = true;
    }

    fn deactivate(&mut self) {
        // Abort any in-progress drag and remove the gizmo and copy previews.
        let scene = self.document.lock().unwrap().scene().clone();
        self.driver.teardown(&scene);
        self.driver.target_mut().done = false;
    }

    /// Whole parts only — there is nothing to copy below part granularity.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::Node
    }

    fn is_finished(&self) -> bool {
        self.driver.target().done
    }

    // No `finalize`: before a placement the pending result is a copy sitting on
    // its source, and leaving the tool must not silently add a part the user
    // never asked for.
}

/// [`TransformTarget`] over preview copies of the selected parts.
///
/// The copies are `PreviewSession` ghosts, so a drag only moves scene nodes —
/// no CAD work per frame — and the real parts are cut once, on commit.
struct DuplicateTarget {
    /// Supplies the pivot and local frame for the current selection. Only ever
    /// queried; its own transform lifecycle is never begun, so the source nodes
    /// are left alone.
    nodes: NodeTransformTarget,
    /// The copies shown until they are placed.
    preview: PreviewSession,
    /// Part-backed selected nodes the copies were built from, in selection
    /// order. Doubles as the key for noticing a selection change.
    sources: Vec<NodeId>,
    /// Set by `activate`, consumed by the next `sync_sources`.
    pending_spawn: bool,
    /// Raised once the copies have been placed or abandoned: the tool cedes
    /// back to selection, and no further copies are spawned in the frames
    /// before that happens.
    done: bool,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    notifications: Notifications,
}

impl DuplicateTarget {
    /// Rebuild the copies when the tool has just been activated or the
    /// selection has changed under it.
    fn sync_sources(&mut self, ctx: &mut EventContext) {
        if self.done {
            return;
        }
        let selected = self.selected_part_nodes(ctx.selection);
        if !self.pending_spawn && selected == self.sources {
            return;
        }
        self.pending_spawn = false;
        self.respawn(selected);
    }

    /// The selected nodes that are CAD parts, in selection order.
    fn selected_part_nodes(&self, selection: &SelectionManager) -> Vec<NodeId> {
        let document = self.document.lock().unwrap();
        selection
            .iter()
            .filter_map(|item| match item {
                SelectionItem::Node(node) => document.part_for_node(*node).map(|_| *node),
                SelectionItem::SubGeometry { .. } => None,
            })
            .collect()
    }

    /// Replace the previewed copies with one per node in `sources`.
    fn respawn(&mut self, sources: Vec<NodeId>) {
        self.preview.cancel();
        self.sources = sources;

        // The preview session locks the document itself, so read everything the
        // copies need before building them.
        let options = self.construction_options.borrow().preview_options();
        let (shapes, scene) = {
            let document = self.document.lock().unwrap();
            let shapes = self
                .sources
                .iter()
                .filter_map(|&node| document.part_for_node(node))
                .filter_map(|part| document.get_part(part))
                // A shallow clone is right here: the ghost is only tessellated,
                // never modified. `duplicate_parts` deep-copies on commit.
                .map(|part| (part.name.clone(), part.shape.clone()))
                .collect::<Vec<_>>();
            (shapes, document.scene().clone())
        };

        // Until it is dragged, a copy sits exactly on its source: picking has to
        // reach the source underneath, and the copy is not a part to select yet.
        for (name, shape) in shapes {
            match self.preview.add_preview_from_shape(&shape, &options, &name) {
                Some(node) => scene.lock().set_node_flags(node, NodeFlags::DO_NOT_SELECT),
                None => log::warn!("Copy of {name} could not be tessellated"),
            }
        }
    }

    /// Cut the real parts at `placement` and select them. Ends the tool either
    /// way — the previews are gone, so there is nothing left to place.
    fn apply(&mut self, placement: Matrix4, ctx: &mut EventContext) {
        self.preview.cancel();
        let sources = std::mem::take(&mut self.sources);
        self.done = true;

        let copies = {
            let mut document = self.document.lock().unwrap();
            let parts: Vec<PartId> =
                sources.iter().filter_map(|&node| document.part_for_node(node)).collect();
            duplicate_parts(&mut document, &parts, &[placement])
        };

        match copies {
            Ok(copies) => {
                let nodes: Vec<NodeId> = {
                    let document = self.document.lock().unwrap();
                    copies.iter().filter_map(|&part| document.node_for_part(part)).collect()
                };
                // The copies are what the user is now working with.
                ctx.selection.clear();
                ctx.selection.extend(nodes.into_iter().map(SelectionItem::Node));
            }
            Err(e) => {
                log::error!("duplicate failed: {e:#}");
                self.notifications.error(format!("Duplicate failed: {e}"));
            }
        }
    }

    /// Drop the previewed copies without committing them.
    fn discard(&mut self) {
        self.preview.cancel();
        self.sources.clear();
    }
}

impl TransformTarget for DuplicateTarget {
    fn frame(&mut self, ctx: &mut EventContext) -> Option<TransformFrame> {
        self.nodes.frame(ctx)
    }

    fn begin(&mut self, _ctx: &mut EventContext) -> bool {
        // The copies already exist; there is no source state to snapshot.
        !self.sources.is_empty()
    }

    fn preview(&mut self, interaction: &TransformInteraction, ctx: &mut EventContext) {
        let camera = ctx.camera.clone();
        let delta = interaction.delta_matrix(&camera, ctx.size);
        // One delta for every copy: the whole selection moves as a rigid group,
        // and each copy was tessellated at its source's world position.
        self.preview.set_preview_transform(decompose_matrix(&delta));
    }

    fn commit(&mut self, interaction: &TransformInteraction, ctx: &mut EventContext) {
        let camera = ctx.camera.clone();
        let delta = interaction.delta_matrix(&camera, ctx.size);
        self.apply(delta, ctx);
    }

    fn cancel(&mut self, _ctx: &mut EventContext) {
        // Undo the drag but keep the copies: the gizmo stays at the original
        // pivot, ready for another attempt.
        self.preview.set_preview_transform(Transform::IDENTITY);
    }

    fn abort(&mut self, _scene: &Scene) {
        // The preview session resolves the document's current scene itself.
        self.discard();
    }
}
