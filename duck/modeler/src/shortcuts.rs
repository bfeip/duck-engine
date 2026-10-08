//! Keyboard shortcuts for the app's own actions and for switching tools.

use duck_engine_viewer::bindings::{InputBinding, InputMap};
use duck_engine_viewer::event::{DeviceEvent, Event, EventContext};
use duck_engine_viewer::input::{ElementState, Key, KeyEvent, Modifiers, NamedKey};
use duck_engine_viewer::operator::Operator;

use crate::tools::{ToolId, ToolInfo};
use crate::AppAction;

/// Last-priority operator turning keys nothing else claimed into
/// [`AppAction`]s, which the app takes once a frame.
///
/// Registered behind the tool host, so an active tool keeps any key it
/// consumes, such as X for a transform's axis.
pub struct Shortcuts {
    bindings: InputMap<AppAction>,
    /// Actions asked for since the last [`take`](Self::take), oldest first.
    pending: Vec<AppAction>,
}

impl Shortcuts {
    /// X or Delete deletes the selected parts, Ctrl+Z undoes, Ctrl+Shift+Z or
    /// Ctrl+Y redoes, and each of `tools`' keys switches to it.
    pub fn new(tools: &[(ToolId, ToolInfo)]) -> Self {
        let ctrl = Modifiers { control: true, ..Modifiers::default() };
        let ctrl_shift = Modifiers { shift: true, ..ctrl };
        let chord = |key, modifiers| InputBinding::Key { key, modifiers };
        let plain = |key| chord(key, Modifiers::default());
        let mut bindings = InputMap::new()
            .bind(plain(Key::Character('x')), AppAction::Delete)
            .bind(plain(Key::Named(NamedKey::Delete)), AppAction::Delete)
            .bind(chord(Key::Character('z'), ctrl), AppAction::Undo)
            .bind(chord(Key::Character('z'), ctrl_shift), AppAction::Redo)
            .bind(chord(Key::Character('y'), ctrl), AppAction::Redo);
        for &(id, info) in tools {
            if let Some(c) = info.shortcut {
                bindings.add(plain(Key::Character(c)), AppAction::SwitchTool(Some(id)));
            }
        }
        Self { bindings, pending: Vec::new() }
    }

    /// Takes the actions asked for since the last take, oldest first.
    pub fn take(&mut self) -> Vec<AppAction> {
        std::mem::take(&mut self.pending)
    }

    /// Records the action `key_event` asks for while `modifiers` are held, if
    /// any. Returns whether it asked for one.
    fn on_key(&mut self, key_event: &KeyEvent, modifiers: Modifiers) -> bool {
        if key_event.state != ElementState::Pressed || key_event.repeat {
            return false;
        }
        let Some(&action) = self.bindings.actions_for_key(&key_event.logical_key, modifiers).first()
        else {
            return false;
        };
        self.pending.push(action);
        true
    }
}

impl Operator for Shortcuts {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(DeviceEvent::KeyboardInput { event: key_event, .. }) = event else {
            return false;
        };
        self.on_key(key_event, ctx.modifiers)
    }

    fn name(&self) -> &str {
        "Shortcuts"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use duck_engine_scene::Scene;
    use duck_engine_viewer::operator::SelectionOperator;

    use super::*;
    use crate::document::Document;
    use crate::notifications::Notifications;
    use crate::testing::{key, named_key, workspace};
    use crate::tools::{DuplicateTool, ToolManager};

    const CTRL: Modifiers = Modifiers { control: true, shift: false, alt: false, super_key: false };
    const CTRL_SHIFT: Modifiers =
        Modifiers { control: true, shift: true, alt: false, super_key: false };

    #[test]
    fn delete_undo_and_redo_have_their_keys() {
        let mut shortcuts = Shortcuts::new(&[]);
        for (event, modifiers) in [
            (key('x'), Modifiers::default()),
            // Case-insensitive.
            (key('X'), Modifiers::default()),
            (named_key(NamedKey::Delete), Modifiers::default()),
            (key('z'), CTRL),
            // Shift+Z arrives as an uppercase character.
            (key('Z'), CTRL_SHIFT),
            (key('y'), CTRL),
        ] {
            assert!(shortcuts.on_key(&event, modifiers));
        }

        use AppAction::{Delete, Redo, Undo};
        assert_eq!(shortcuts.take(), [Delete, Delete, Delete, Undo, Redo, Redo]);
        assert!(shortcuts.take().is_empty(), "taking empties the queue");
    }

    #[test]
    fn a_tools_key_switches_to_it() {
        let ws = workspace(Document::new(Scene::default()));
        let selection = Arc::new(Mutex::new(SelectionOperator::new()));
        let mut tools = ToolManager::new(selection, Notifications::default());
        let duplicate = tools.register(DuplicateTool::new(&ws));
        let mut shortcuts = Shortcuts::new(&tools.palette_entries());

        assert!(shortcuts.on_key(&key('d'), Modifiers::default()));
        assert!(shortcuts.on_key(&key('D'), Modifiers::default()));
        assert_eq!(shortcuts.take(), [AppAction::SwitchTool(Some(duplicate)); 2]);
    }

    /// Plain Z is left for a transform's axis, as is any key bound to nothing.
    #[test]
    fn unbound_keys_are_left_for_others() {
        let mut shortcuts = Shortcuts::new(&[]);
        assert!(!shortcuts.on_key(&key('z'), Modifiers::default()));
        assert!(!shortcuts.on_key(&key('q'), Modifiers::default()));
        assert!(shortcuts.take().is_empty());
    }

    #[test]
    fn repeats_releases_and_other_chords_ask_for_nothing() {
        let mut shortcuts = Shortcuts::new(&[]);
        let released = KeyEvent { state: ElementState::Released, ..key('x') };
        assert!(!shortcuts.on_key(&KeyEvent { repeat: true, ..key('x') }, Modifiers::default()));
        assert!(!shortcuts.on_key(&released, Modifiers::default()));
        assert!(!shortcuts.on_key(&KeyEvent { repeat: true, ..key('z') }, CTRL));
        assert!(!shortcuts.on_key(&key('x'), CTRL));
        assert!(shortcuts.take().is_empty());
    }
}
