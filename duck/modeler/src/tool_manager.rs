use std::sync::{Arc, Mutex, MutexGuard};

use duck_engine_viewer::bindings::{InputBinding, InputMap};
use duck_engine_viewer::event::{DeviceEvent, Event, EventContext, EventDispatcher};
use duck_engine_viewer::input::{ElementState, Key, KeyEvent, Modifiers};
use duck_engine_viewer::operator::{Operator, SelectionMode, SelectionOperator};
use duck_engine_viewer::scene::Scene;

use crate::cursor::Cursor3d;
use crate::tool::{ModelingTool, ToolInfo};

/// Opaque handle to a registered tool. Minted only by [`ToolManager::register`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ToolId(usize);

/// The single dispatcher-registered operator for all modeling tools.
/// 
/// Forwards events to the active tool, if any. Registered once at startup, in front
/// of the selection/navigation operators.
struct ToolHost {
    active: Option<Arc<Mutex<dyn ModelingTool>>>,
}

impl Operator for ToolHost {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        match &self.active {
            Some(tool) => tool.lock().unwrap().dispatch(event, ctx),
            None => false,
        }
    }

    fn name(&self) -> &str {
        "ToolHost"
    }
}

/// Last-priority operator that turns unclaimed shortcut keys into deferred
/// tool-activation requests, applied by [`ToolManager::update`].
struct ToolSwitcher {
    bindings: InputMap<ToolId>,
    pending: Option<ToolId>,
}

impl ToolSwitcher {
    /// Records an activation request if the key matches a shortcut.
    /// Returns `true` when the key was claimed.
    fn handle_key(&mut self, key_event: &KeyEvent, modifiers: Modifiers) -> bool {
        if key_event.state != ElementState::Pressed || key_event.repeat {
            return false;
        }
        match self.bindings.actions_for_key(&key_event.logical_key, modifiers).first() {
            Some(&id) => {
                self.pending = Some(id);
                true
            }
            None => false,
        }
    }
}

impl Operator for ToolSwitcher {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(DeviceEvent::KeyboardInput { event: key_event, .. }) = event else {
            return false;
        };
        self.handle_key(key_event, ctx.modifiers)
    }

    fn name(&self) -> &str {
        "ToolSwitcher"
    }
}

/// Owns the registered modeling tools and everything generic about driving
/// them.
/// 
/// Handles activation/deactivation, the always-on selection operator's
/// granularity, auto-return to selection when a tool finishes, and the 3D cursor.
/// Adding a tool to the modeler is implementing
/// [`ModelingTool`] plus one [`ToolManager::register`] call.
pub struct ToolManager {
    tools: Vec<Arc<Mutex<dyn ModelingTool>>>,
    /// `None` means plain selection mode.
    active: Option<ToolId>,
    host: Arc<Mutex<ToolHost>>,
    switcher: Arc<Mutex<ToolSwitcher>>,
    sel_op: Arc<Mutex<SelectionOperator>>,
    /// The modeler-owned 3D cursor, driven each frame from the active tool.
    cursor: Cursor3d,
}

impl ToolManager {
    pub fn new(sel_op: Arc<Mutex<SelectionOperator>>) -> Self {
        Self {
            tools: Vec::new(),
            active: None,
            host: Arc::new(Mutex::new(ToolHost { active: None })),
            switcher: Arc::new(Mutex::new(ToolSwitcher {
                bindings: InputMap::new(),
                pending: None,
            })),
            sel_op,
            cursor: Cursor3d::default(),
        }
    }

    /// Registers the forwarding host with the dispatcher at highest priority
    /// and the shortcut switcher at lowest, so tool switching only claims keys
    /// no other operator consumed.
    /// Call once, after the selection/navigation operators are registered.
    pub fn install(&self, dispatcher: &mut EventDispatcher) {
        dispatcher.push_front(Arc::clone(&self.host));
        dispatcher.push_back(Arc::clone(&self.switcher));
    }

    pub fn register<T: ModelingTool>(&mut self, tool: T) -> ToolId {
        let id = ToolId(self.tools.len());
        if let Some(c) = tool.info().shortcut {
            self.switcher.lock().unwrap().bindings.add(
                InputBinding::Key { key: Key::Character(c), modifiers: Modifiers::default() },
                id,
            );
        }
        self.tools.push(Arc::new(Mutex::new(tool)));
        id
    }

    /// Switches the active tool; `None` returns to plain selection.
    /// Re-activating the already active tool is a no-op.
    pub fn activate(&mut self, id: Option<ToolId>) {
        if id == self.active {
            return;
        }

        // Locks must be taken strictly one at a time
        if let Some(old) = self.active {
            self.tools[old.0].lock().unwrap().deactivate();
        }

        self.host.lock().unwrap().active = id.map(|i| Arc::clone(&self.tools[i.0]));

        let mode = match id {
            Some(i) => {
                let mut tool = self.tools[i.0].lock().unwrap();
                tool.activate();
                tool.selection_mode()
            }
            None => SelectionMode::default(),
        };
        self.sel_op.lock().unwrap().mode = mode;

        self.active = id;
    }

    /// Per-frame update. Should be called every frame.
    pub fn update(&mut self, scene: &Arc<Mutex<Scene>>) {
        let requested = self.switcher.lock().unwrap().pending.take();
        if let Some(id) = requested {
            self.activate(Some(id));
        }

        if self.active.is_some_and(|i| self.tools[i.0].lock().unwrap().is_finished()) {
            self.activate(None);
        }

        let target = self
            .active
            .and_then(|i| self.tools[i.0].lock().unwrap().cursor_target());
        self.cursor.update(target, &mut scene.lock().unwrap());
    }

    /// Palette snapshot for the `ui` module: `(id, info)` per tool.
    /// Taken without holding any tool lock across egui rendering.
    pub fn palette_entries(&self) -> Vec<(ToolId, ToolInfo)> {
        self.tools
            .iter()
            .enumerate()
            .map(|(i, tool)| (ToolId(i), tool.lock().unwrap().info()))
            .collect()
    }

    /// The active tool's id, or `None` in plain selection mode.
    pub fn active_id(&self) -> Option<ToolId> {
        self.active
    }

    /// The active tool, locked for panel rendering, or `None` in selection mode.
    pub fn active_tool(&self) -> Option<MutexGuard<'_, dyn ModelingTool>> {
        self.active.map(|i| self.tools[i.0].lock().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use duck_engine_viewer::input::PhysicalKey;

    use super::*;

    struct MockTool {
        id: &'static str,
        shortcut: Option<char>,
        activations: Arc<AtomicUsize>,
        deactivations: Arc<AtomicUsize>,
    }

    impl MockTool {
        fn new(id: &'static str, shortcut: Option<char>) -> Self {
            Self {
                id,
                shortcut,
                activations: Arc::new(AtomicUsize::new(0)),
                deactivations: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl Operator for MockTool {
        fn dispatch(&mut self, _event: &Event, _ctx: &mut EventContext) -> bool {
            false
        }

        fn name(&self) -> &str {
            self.id
        }
    }

    impl crate::tool::ModelingTool for MockTool {
        fn info(&self) -> crate::tool::ToolInfo {
            crate::tool::ToolInfo { id: self.id, icon: ("mock", &[]), shortcut: self.shortcut }
        }

        fn activate(&mut self) {
            self.activations.fetch_add(1, Ordering::SeqCst);
        }

        fn deactivate(&mut self) {
            self.deactivations.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn key_press(c: char) -> KeyEvent {
        KeyEvent {
            physical_key: PhysicalKey::Unidentified,
            logical_key: Key::Character(c),
            state: ElementState::Pressed,
            repeat: false,
        }
    }

    fn switcher_with(bindings: &[(char, ToolId)]) -> ToolSwitcher {
        let mut map = InputMap::new();
        for &(c, id) in bindings {
            map.add(
                InputBinding::Key { key: Key::Character(c), modifiers: Modifiers::default() },
                id,
            );
        }
        ToolSwitcher { bindings: map, pending: None }
    }

    #[test]
    fn switcher_matches_bound_key() {
        let mut switcher = switcher_with(&[('g', ToolId(0))]);
        assert!(switcher.handle_key(&key_press('g'), Modifiers::default()));
        assert_eq!(switcher.pending, Some(ToolId(0)));

        // Case-insensitive via InputMap normalization.
        switcher.pending = None;
        assert!(switcher.handle_key(&key_press('G'), Modifiers::default()));
        assert_eq!(switcher.pending, Some(ToolId(0)));

        assert!(!switcher.handle_key(&key_press('q'), Modifiers::default()));
    }

    #[test]
    fn switcher_ignores_repeat_release_and_modifiers() {
        let mut switcher = switcher_with(&[('g', ToolId(0))]);

        let mut repeat = key_press('g');
        repeat.repeat = true;
        assert!(!switcher.handle_key(&repeat, Modifiers::default()));

        let mut released = key_press('g');
        released.state = ElementState::Released;
        assert!(!switcher.handle_key(&released, Modifiers::default()));

        let ctrl = Modifiers { control: true, ..Modifiers::default() };
        assert!(!switcher.handle_key(&key_press('g'), ctrl));

        assert_eq!(switcher.pending, None);
    }

    #[test]
    fn register_binds_shortcut_to_id() {
        let mut manager = ToolManager::new(Arc::new(Mutex::new(SelectionOperator::new())));
        manager.register(MockTool::new("plain", None));
        manager.register(MockTool::new("keyed", Some('g')));

        let mut switcher = manager.switcher.lock().unwrap();
        assert!(switcher.handle_key(&key_press('g'), Modifiers::default()));
        assert_eq!(switcher.pending, Some(ToolId(1)));
    }

    #[test]
    fn update_applies_pending_switch() {
        let mut manager = ToolManager::new(Arc::new(Mutex::new(SelectionOperator::new())));
        let tool = MockTool::new("keyed", Some('g'));
        let activations = Arc::clone(&tool.activations);
        manager.register(MockTool::new("plain", None));
        manager.register(tool);

        let scene = Arc::new(Mutex::new(Scene::new()));
        manager.switcher.lock().unwrap().handle_key(&key_press('g'), Modifiers::default());
        manager.update(&scene);

        assert_eq!(manager.active_tool().unwrap().info().id, "keyed");
        assert_eq!(activations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn update_same_tool_pending_is_noop() {
        let mut manager = ToolManager::new(Arc::new(Mutex::new(SelectionOperator::new())));
        let tool = MockTool::new("keyed", Some('g'));
        let deactivations = Arc::clone(&tool.deactivations);
        let id = manager.register(tool);
        manager.activate(Some(id));

        let scene = Arc::new(Mutex::new(Scene::new()));
        manager.switcher.lock().unwrap().handle_key(&key_press('g'), Modifiers::default());
        manager.update(&scene);

        assert_eq!(manager.active_tool().unwrap().info().id, "keyed");
        assert_eq!(deactivations.load(Ordering::SeqCst), 0);
    }
}
