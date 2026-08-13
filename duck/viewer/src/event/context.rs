
use crate::input::Modifiers;
use crate::scene::{PositionedCamera, Scene};
use crate::selection::SelectionManager;

use super::Event;

/// Context passed to event callbacks, providing mutable access to application state.
///
/// All coordinates and sizes are local to the dispatching view.
pub struct EventContext<'c> {
    /// Current view size (width, height)
    pub size: (u32, u32),
    /// Current cursor position in view-local coordinates (x, y), or None if the
    /// cursor is not over the view
    pub cursor_position: &'c mut Option<(f32, f32)>,
    /// Shared scene handle. Its methods lock internally; take a guard with
    /// `scene.lock()` only for compound work.
    pub scene: Scene,
    /// The dispatching view's camera. Mutations take effect immediately.
    pub camera: &'c mut PositionedCamera,
    /// Mutable reference to the selection manager
    pub selection: &'c mut SelectionManager,
    /// Currently held keyboard modifier keys, updated by the dispatcher before each dispatch.
    // TODO: In the future we might replace this with an input state struct. Containing
    // not just modifiers but the full input state.
    pub modifiers: Modifiers,
    /// Events emitted by operators during this dispatch, awaiting re-dispatch through
    /// the operator stack. Push via [`Self::emit`]; the [`EventDispatcher`](super::EventDispatcher)
    /// drains this after the current event finishes propagating.
    //
    // NOTE: this only covers events emitted synchronously as a consequence of another
    // event, the queue lives for the duration of one dispatch. It does not support
    // autonomous/background emission; that would require a long-lived owner holding an
    // MPSC channel. `emit`'s signature is forward-compatible with that change.
    pub(crate) emit_queue: Vec<Event>,
}

impl<'c> EventContext<'c> {
    /// Emit a high-level event (or a synthesized device event) to be re-dispatched
    /// through the operator stack once the current event finishes propagating.
    ///
    /// Operators use this to signal that something happened that other operators may need
    /// to respond to (e.g. [`AppEvent::Selection`](super::AppEvent::Selection)).
    pub fn emit(&mut self, event: impl Into<Event>) {
        self.emit_queue.push(event.into());
    }

    /// Viewport aspect ratio.
    pub fn aspect(&self) -> f32 {
        self.size.0 as f32 / self.size.1 as f32
    }
}
