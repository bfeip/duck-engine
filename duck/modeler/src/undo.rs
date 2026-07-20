//! Keyboard-driven undo/redo.

use duck_engine_viewer::bindings::{InputBinding, InputMap};
use duck_engine_viewer::event::{DeviceEvent, Event, EventContext};
use duck_engine_viewer::input::{ElementState, Key, KeyEvent, Modifiers};
use duck_engine_viewer::operator::Operator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndoAction {
    Undo,
    Redo,
}

/// Last-priority operator that turns an unclaimed Ctrl+Z / Ctrl+Shift+Z /
/// Ctrl+Y press into a deferred undo/redo request, applied by the app's
/// per-frame update.
///
/// Registered behind the tool host so an active tool keeps any key it consumes.
pub struct UndoRedoOperator {
    bindings: InputMap<UndoAction>,
    pending: Option<UndoAction>,
}

impl UndoRedoOperator {
    pub fn new() -> Self {
        let ctrl = Modifiers { control: true, ..Modifiers::default() };
        let ctrl_shift = Modifiers { control: true, shift: true, ..Modifiers::default() };
        Self {
            bindings: InputMap::new()
                .bind(
                    InputBinding::Key { key: Key::Character('z'), modifiers: ctrl },
                    UndoAction::Undo,
                )
                .bind(
                    InputBinding::Key { key: Key::Character('z'), modifiers: ctrl_shift },
                    UndoAction::Redo,
                )
                .bind(
                    InputBinding::Key { key: Key::Character('y'), modifiers: ctrl },
                    UndoAction::Redo,
                ),
            pending: None,
        }
    }

    /// Records an undo/redo request if the key matches a binding.
    /// Returns `true` when the key was claimed.
    fn handle_key(&mut self, key_event: &KeyEvent, modifiers: Modifiers) -> bool {
        if key_event.state != ElementState::Pressed || key_event.repeat {
            return false;
        }
        let actions = self.bindings.actions_for_key(&key_event.logical_key, modifiers);
        let Some(&action) = actions.first() else {
            return false;
        };
        self.pending = Some(action);
        true
    }

    /// Takes the pending undo/redo request, if any.
    pub fn take_pending(&mut self) -> Option<UndoAction> {
        self.pending.take()
    }
}

impl Operator for UndoRedoOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(DeviceEvent::KeyboardInput { event: key_event, .. }) = event else {
            return false;
        };
        self.handle_key(key_event, ctx.modifiers)
    }

    fn name(&self) -> &str {
        "UndoRedo"
    }
}

#[cfg(test)]
mod tests {
    use duck_engine_viewer::input::PhysicalKey;

    use super::*;

    fn key_press(key: Key) -> KeyEvent {
        KeyEvent {
            physical_key: PhysicalKey::Unidentified,
            logical_key: key,
            state: ElementState::Pressed,
            repeat: false,
        }
    }

    const CTRL: Modifiers = Modifiers { control: true, shift: false, alt: false, super_key: false };
    const CTRL_SHIFT: Modifiers =
        Modifiers { control: true, shift: true, alt: false, super_key: false };

    #[test]
    fn matches_bound_keys() {
        let mut op = UndoRedoOperator::new();

        assert!(op.handle_key(&key_press(Key::Character('z')), CTRL));
        assert_eq!(op.take_pending(), Some(UndoAction::Undo));

        // Shift+Z arrives as an uppercase character; case is normalized.
        assert!(op.handle_key(&key_press(Key::Character('Z')), CTRL_SHIFT));
        assert_eq!(op.take_pending(), Some(UndoAction::Redo));

        assert!(op.handle_key(&key_press(Key::Character('y')), CTRL));
        assert_eq!(op.take_pending(), Some(UndoAction::Redo));
        assert_eq!(op.take_pending(), None, "take_pending must reset the request");
    }

    #[test]
    fn plain_z_is_left_for_tools() {
        let mut op = UndoRedoOperator::new();
        assert!(!op.handle_key(&key_press(Key::Character('z')), Modifiers::default()));
        assert!(op.take_pending().is_none());
    }

    #[test]
    fn ignores_repeat_and_release() {
        let mut op = UndoRedoOperator::new();

        let mut repeat = key_press(Key::Character('z'));
        repeat.repeat = true;
        assert!(!op.handle_key(&repeat, CTRL));

        let mut released = key_press(Key::Character('z'));
        released.state = ElementState::Released;
        assert!(!op.handle_key(&released, CTRL));

        assert!(op.take_pending().is_none());
    }
}
