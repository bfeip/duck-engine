use std::sync::{Arc, Mutex, MutexGuard};

use crate::input::Modifiers;
use crate::scene::{NodePayload, PositionedCamera, Scene};
use crate::selection::SelectionManager;

use super::Event;

/// Context passed to event callbacks, providing mutable access to application state.
pub struct EventContext<'c> {
    /// Current viewport size (width, height)
    pub size: (u32, u32),
    /// Current cursor position in screen coordinates (x, y), or None if cursor is not over the window
    pub cursor_position: &'c mut Option<(f32, f32)>,
    /// Shared scene reference. Lock with `scene.lock().unwrap()` to access.
    pub scene: Arc<Mutex<Scene>>,
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

    /// Acquire the scene mutex.
    pub fn lock_scene(&self) -> MutexGuard<'_, Scene> {
        self.scene.lock().unwrap()
    }

    /// Returns a [`PositionedCamera`] for the active camera node.
    ///
    /// Combines the node's world transform with its [`CameraProjection`] payload and
    /// the current viewport aspect ratio. Panics if no active camera is set.
    pub fn camera(&self) -> PositionedCamera {
        let aspect = self.size.0 as f32 / self.size.1 as f32;
        self.scene
            .lock().unwrap()
            .active_camera_positioned(aspect)
            .expect("no active camera in scene")
    }

    /// Writes a [`PositionedCamera`] back to the active camera node.
    ///
    /// Updates both the node transform (pose) and the Camera payload (projection
    /// intrinsics + focus distance).
    pub fn set_camera(&mut self, cam: PositionedCamera) {
        let mut scene = self.scene.lock().unwrap();
        let id = scene.active_camera().expect("no active camera in scene");
        scene.set_node_transform(id, cam.to_node_transform());
        scene.set_node_payload(id, NodePayload::Camera(cam.projection()));
    }

    /// Clones the active camera, passes it to `f` for mutation, then writes it back.
    pub fn with_camera_mut(&mut self, f: impl FnOnce(&mut PositionedCamera)) {
        let mut cam = self.camera();
        f(&mut cam);
        self.set_camera(cam);
    }
}
