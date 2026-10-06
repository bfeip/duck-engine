use duck_engine_scene::resource::{NodeFlags, NodeId};
use duck_engine_scene::Scene;
use duck_engine_viewer::{
    common::{decompose_matrix, Matrix4, SquareMatrix, Transform},
    event::{Event, EventContext},
    operator::{
        NodeTransformTarget, Operator, SelectionMode, TransformDriver, TransformFrame,
        TransformInteraction, TransformMode, TransformTarget,
    },
    selection::SelectionItem,
};

use crate::document::PartId;
use crate::ops::duplicate::duplicate_parts;
use crate::preview::PreviewSession;
use crate::tools::{Gesture, ModelingTool, ToolInfo, Workspace};
use crate::ui::icons;
use super::targets::selected_parts;

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
}

impl DuplicateTool {
    pub fn new(workspace: &Workspace) -> Self {
        let target = DuplicateTarget {
            nodes: NodeTransformTarget::new(),
            preview: workspace.preview_session(),
            sources: Vec::new(),
            pending_spawn: false,
            done: false,
            workspace: workspace.clone(),
        };
        Self { driver: TransformDriver::with_target(TransformMode::Translate, target) }
    }

    /// Commit the copies at their current, undragged position. Returns whether
    /// there were any.
    fn place(&mut self, ctx: &mut EventContext) -> bool {
        let target = self.driver.target_mut();
        if target.sources.is_empty() {
            return false;
        }
        target.apply(Matrix4::identity(), ctx);
        true
    }

    /// Drop the copies without adding anything to the document, and leave.
    fn abandon(&mut self) {
        let target = self.driver.target_mut();
        target.discard();
        target.done = true;
    }
}

impl ModelingTool for DuplicateTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "duplicate", icon: icons::DUPLICATE, shortcut: Some('d') }
    }

    /// Whole parts only — there is nothing to copy below part granularity.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::Node
    }

    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let gesture = Gesture::read(event, ctx.modifiers);
        // The copies are built from the selection, which `activate` cannot see;
        // the frame tick is the first point with an event context.
        if gesture == Some(Gesture::Frame) {
            self.driver.target_mut().sync_sources(ctx);
        }

        // A drag takes its own confirm and cancel.
        if self.driver.dispatch(event, ctx) {
            return true;
        }
        if self.driver.is_active() {
            return false;
        }
        match gesture {
            Some(Gesture::Finish) => self.place(ctx),
            Some(Gesture::Cancel) => {
                self.abandon();
                true
            }
            _ => false,
        }
    }

    fn activate(&mut self) {
        self.driver.set_gizmo_enabled(true);
        self.driver.target_mut().pending_spawn = true;
    }

    fn deactivate(&mut self) {
        // Abort any in-progress drag and remove the gizmo and copy previews.
        let scene = self.driver.target().workspace.document.lock().unwrap().scene().clone();
        self.driver.teardown(&scene);
        self.driver.target_mut().done = false;
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
    workspace: Workspace,
}

impl DuplicateTarget {
    /// Rebuild the copies when the tool has just been activated or the
    /// selection has changed under it.
    fn sync_sources(&mut self, ctx: &mut EventContext) {
        if self.done {
            return;
        }
        let selected = selected_parts(ctx.selection, &self.workspace.document.lock().unwrap());
        if !self.pending_spawn && selected == self.sources {
            return;
        }
        self.pending_spawn = false;
        self.respawn(selected);
    }

    /// Replace the previewed copies with one per node in `sources`.
    fn respawn(&mut self, sources: Vec<NodeId>) {
        self.preview.cancel();
        self.sources = sources;

        // The preview session locks the document itself, so read everything the
        // copies need before building them.
        let options = self.workspace.preview_options();
        let (shapes, scene) = {
            let document = self.workspace.document.lock().unwrap();
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

    /// Cut the real parts at `placement`, select them and end the tool. A
    /// failure keeps the copies, back where they started, to place again.
    fn apply(&mut self, placement: Matrix4, ctx: &mut EventContext) {
        let copies = {
            let mut document = self.workspace.document.lock().unwrap();
            let parts: Vec<PartId> =
                self.sources.iter().filter_map(|&node| document.part_for_node(node)).collect();
            duplicate_parts(&mut document, &parts, &[placement]).map(|copies| {
                copies.iter().filter_map(|&part| document.node_for_part(part)).collect::<Vec<_>>()
            })
        };

        match copies {
            Ok(nodes) => {
                self.discard();
                self.done = true;
                // The copies are what the user is now working with.
                ctx.selection.clear();
                ctx.selection.extend(nodes.into_iter().map(SelectionItem::Node));
            }
            Err(e) => {
                self.workspace.notifications.failure("Duplicate", &e);
                self.preview.set_preview_transform(Transform::IDENTITY);
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
        let delta = interaction.delta_matrix(ctx.camera, ctx.size);
        // One delta for every copy: the whole selection moves as a rigid group,
        // and each copy was tessellated at its source's world position.
        self.preview.set_preview_transform(decompose_matrix(&delta));
    }

    fn commit(&mut self, interaction: &TransformInteraction, ctx: &mut EventContext) {
        let delta = interaction.delta_matrix(ctx.camera, ctx.size);
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
