use duck_engine_scene::resource::{NodeFlags, NodeId};
use duck_engine_viewer::{
    common::{decompose_matrix, Matrix4, SquareMatrix, Transform},
    event::{DeviceEvent, Event, EventContext},
    operator::{Handle, HandleEvent, SelectionMode},
    selection::{SelectionItem, SelectionManager},
};

use crate::document::PartId;
use crate::ops::duplicate::duplicate_parts;
use crate::preview::PreviewSession;
use crate::tools::{Gesture, ModelingTool, PanelContext, ToolInfo, Workspace};
use crate::ui::icons;
use super::targets::selected_parts;
use super::transform::{
    parts_frame, values_ui, Frame, Input, Manipulation, Outcome, Space, TransformMode,
};

/// Copies the selected parts and places the copies with the translate gizmo.
///
/// Selecting the tool immediately makes a copy of every selected part, sitting
/// on top of its source, and shows the translate handles. Drag a handle (or
/// press G or X/Y/Z, or type a distance) to carry the copies off; left-click or
/// Enter confirms, right-click or Escape reverts the drag. With no drag in
/// progress, right-click or Enter places the copies where they already are and
/// Escape leaves without copying anything. Either way the tool then cedes back
/// to selection.
///
/// The copies are only preview geometry until they are placed, so nothing
/// reaches the document unless the user confirms — and then the whole
/// placement is a single undo step.
pub struct DuplicateTool {
    manipulation: Manipulation,
    copies: Copies,
    /// Where a drag of the copies would start, refreshed while idle.
    frame: Option<Frame>,
}

impl DuplicateTool {
    pub fn new(workspace: &Workspace) -> Self {
        let copies = Copies {
            preview: workspace.preview_session(),
            sources: Vec::new(),
            pending_spawn: false,
            done: false,
            workspace: workspace.clone(),
        };
        Self { manipulation: Manipulation::new(TransformMode::Translate), copies, frame: None }
    }

    /// Carries out what the drag asked. Returns whether the event is taken.
    fn act(&mut self, outcome: Outcome, selection: &mut SelectionManager) -> bool {
        match outcome {
            Outcome::Ignored => return false,
            Outcome::Consumed => {}
            // The copies already exist; there is nothing to lock. A drag started
            // by typing has already moved.
            Outcome::Started => match self.copies.sources.is_empty() {
                true => self.manipulation.abort(),
                false => self.preview(),
            },
            Outcome::Changed => self.preview(),
            Outcome::Confirmed(delta) => self.copies.apply(delta, selection),
            // Undo the drag but keep the copies, ready for another attempt.
            Outcome::Cancelled => self.copies.preview.set_preview_transform(Transform::IDENTITY),
        }
        true
    }

    /// Shows the copies carried by the drag so far. One delta for every copy:
    /// the selection moves as a rigid group, and each copy was tessellated at
    /// its source's world position.
    fn preview(&self) {
        self.copies.preview.set_preview_transform(decompose_matrix(&self.manipulation.delta()));
    }

    /// Commit the copies at their current, undragged position. Returns whether
    /// there were any.
    fn place(&mut self, selection: &mut SelectionManager) -> bool {
        if self.copies.sources.is_empty() {
            return false;
        }
        self.copies.apply(Matrix4::identity(), selection);
        true
    }

    /// Drop the copies without adding anything to the document, and leave.
    fn abandon(&mut self) {
        self.copies.discard();
        self.copies.done = true;
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
            self.copies.sync_sources(ctx.selection);
        }
        let refresh = matches!(
            event,
            Event::Device(DeviceEvent::Update { .. } | DeviceEvent::KeyboardInput { .. })
        );
        if refresh && !self.manipulation.is_active() {
            self.frame = self.copies.frame(ctx.selection);
        }

        // A drag takes its own confirm and cancel.
        let units = self.copies.workspace.display_units();
        let outcome = self.manipulation.dispatch(event, self.frame, &Input::of(ctx, units));
        if self.act(outcome, ctx.selection) {
            return true;
        }
        if self.manipulation.is_active() {
            return false;
        }
        match gesture {
            Some(Gesture::Finish) => self.place(ctx.selection),
            Some(Gesture::Cancel) => {
                self.abandon();
                true
            }
            _ => false,
        }
    }

    fn handles(&self) -> Vec<Handle> {
        self.manipulation.handles(self.frame.as_ref())
    }

    fn on_handle(&mut self, event: &HandleEvent, ctx: &mut EventContext) {
        let units = self.copies.workspace.display_units();
        let outcome = self.manipulation.on_handle(event, self.frame, &Input::of(ctx, units));
        self.act(outcome, ctx.selection);
    }

    /// Only while a drag is in progress, for its values.
    fn panel_title(&self) -> Option<&str> {
        self.manipulation.is_active().then_some("Duplicate")
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, _panel: &mut PanelContext) {
        if values_ui(ui, &mut self.manipulation, self.copies.workspace.display_units()) {
            self.preview();
        }
    }

    fn activate(&mut self) {
        // Carry the copies along their sources' own axes.
        self.manipulation.set_orientation(Space::Local);
        self.copies.pending_spawn = true;
    }

    fn deactivate(&mut self) {
        // Abort any in-progress drag and remove the copy previews.
        self.manipulation.abort();
        self.copies.discard();
        self.copies.done = false;
        self.frame = None;
    }

    fn is_finished(&self) -> bool {
        self.copies.done
    }

    // No `finalize`: before a placement the pending result is a copy sitting on
    // its source, and leaving the tool must not silently add a part the user
    // never asked for.
}

/// Preview copies of the selected parts.
///
/// The copies are `PreviewSession` ghosts, so a drag only moves scene nodes —
/// no CAD work per frame — and the real parts are cut once, on commit.
struct Copies {
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

impl Copies {
    /// Rebuild the copies when the tool has just been activated or the
    /// selection has changed under it.
    fn sync_sources(&mut self, selection: &SelectionManager) {
        if self.done {
            return;
        }
        let selected = selected_parts(selection, &self.workspace.document.lock().unwrap());
        if !self.pending_spawn && selected == self.sources {
            return;
        }
        self.pending_spawn = false;
        self.respawn(selected);
    }

    /// Where a drag of the copies acts: about their sources, along the primary
    /// one's axes.
    fn frame(&self, selection: &SelectionManager) -> Option<Frame> {
        let document = self.workspace.document.lock().unwrap();
        parts_frame(&self.sources, selection.primary().map(|item| item.node_id()), &document)
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
                .map(|part| (part.name.clone(), part.shape().clone()))
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
    fn apply(&mut self, placement: Matrix4, selection: &mut SelectionManager) {
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
                selection.clear();
                selection.extend(nodes.into_iter().map(SelectionItem::Node));
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

#[cfg(test)]
mod tests {
    use duck_engine_viewer::common::{InnerSpace, Point3, Real, Vector3};

    use super::*;
    use crate::testing::workspace_with_box;

    const EPSILON: Real = 1e-6;

    #[test]
    fn a_placed_copy_lands_at_the_delta_and_is_selected() {
        let (ws, node) = workspace_with_box();
        let mut tool = DuplicateTool::new(&ws);
        let mut selection = SelectionManager::new();
        selection.extend([SelectionItem::Node(node)]);
        tool.activate();
        tool.copies.sync_sources(&selection);
        assert_eq!(tool.copies.frame(&selection).map(|f| f.pivot), Some(Point3::new(0.0, 0.0, 0.0)));

        tool.copies.apply(Matrix4::from_translation(Vector3::new(5.0, 0.0, 0.0)), &mut selection);

        let copy = selection.primary().expect("the copy is selected").node_id();
        assert_ne!(copy, node);
        let document = ws.document.lock().unwrap();
        let frame = document.part_at(copy).unwrap().frame();
        assert!((frame.position - Point3::new(5.0, 0.0, 0.0)).magnitude() < EPSILON, "{frame:?}");
        assert!(tool.is_finished());
    }
}
