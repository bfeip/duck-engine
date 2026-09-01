use std::sync::{Arc, Mutex, MutexGuard};

use duck_engine_viewer::bindings::{InputBinding, InputMap};
use duck_engine_viewer::scene::Scene;
use duck_engine_viewer::event::{DeviceEvent, Event, EventContext, EventDispatcher};
use duck_engine_viewer::input::{ElementState, Key, KeyEvent, Modifiers};
use duck_engine_viewer::operator::{Operator, SelectionMode, SelectionOperator};
use duck_engine_viewer::selection::SelectionManager;

use crate::cursor::Cursor3d;
use crate::notifications::Notifications;
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
    /// Reports failed implicit commits, so no tool has to.
    notifications: Notifications,
}

impl ToolManager {
    pub fn new(sel_op: Arc<Mutex<SelectionOperator>>, notifications: Notifications) -> Self {
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
            notifications,
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
    ///
    /// The outgoing tool first gets to commit a fully defined result: switching
    /// away is the user moving on, not a request to throw the operation away.
    pub fn activate(&mut self, id: Option<ToolId>, selection: &mut SelectionManager) {
        self.switch(id, Some(selection));
    }

    /// Switches back to plain selection, discarding whatever the active tool
    /// holds. For teardown the document itself demands — undo/redo, delete —
    /// where committing would add a part the user never asked for.
    pub fn discard_active(&mut self) {
        self.switch(None, None);
    }

    /// The one activation path. `finalize` carries the outgoing tool's selection
    /// when it should commit rather than discard.
    fn switch(&mut self, id: Option<ToolId>, finalize: Option<&mut SelectionManager>) {
        if id == self.active {
            return;
        }

        // Locks must be taken strictly one at a time
        if let Some(old) = self.active {
            if let Some(selection) = finalize {
                self.finalize_tool(old, selection);
            }
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

    /// Commits `id`'s pending result, reporting a failure on its behalf. The
    /// caller's `deactivate` then clears whatever a failed commit left behind.
    fn finalize_tool(&self, id: ToolId, selection: &mut SelectionManager) {
        let mut tool = self.tools[id.0].lock().unwrap();
        if let Err(e) = tool.finalize(selection) {
            let name = tool.info().id;
            log::error!("Could not finish {name}: {e:#}");
            self.notifications.error(format!("Could not finish {name}: {e}"));
        }
    }

    /// Per-frame update. Should be called every frame.
    pub fn update(&mut self, scene: &Scene, selection: &mut SelectionManager) {
        let requested = self.switcher.lock().unwrap().pending.take();
        if let Some(id) = requested {
            self.activate(Some(id), selection);
        }

        if self.active.is_some_and(|i| self.tools[i.0].lock().unwrap().is_finished()) {
            self.activate(None, selection);
        }

        let target = self
            .active
            .and_then(|i| self.tools[i.0].lock().unwrap().cursor_target());
        self.cursor.update(target, scene);
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
    use duck_engine_viewer::input::PhysicalKey;

    use super::*;

    fn manager() -> ToolManager {
        ToolManager::new(
            Arc::new(Mutex::new(SelectionOperator::new())),
            Notifications::default(),
        )
    }

    struct MockTool {
        id: &'static str,
        shortcut: Option<char>,
        /// Lifecycle calls in the order they arrived.
        calls: Arc<Mutex<Vec<&'static str>>>,
        /// Whether `finalize` reports a failed commit.
        finalize_fails: bool,
    }

    impl MockTool {
        fn new(id: &'static str, shortcut: Option<char>) -> Self {
            Self {
                id,
                shortcut,
                calls: Arc::new(Mutex::new(Vec::new())),
                finalize_fails: false,
            }
        }

        /// A tool whose pending result can't be committed.
        fn failing(id: &'static str) -> Self {
            Self { finalize_fails: true, ..Self::new(id, None) }
        }

        fn record(&self, call: &'static str) {
            self.calls.lock().unwrap().push(call);
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
            self.record("activate");
        }

        fn deactivate(&mut self) {
            self.record("deactivate");
        }

        fn finalize(&mut self, _selection: &mut SelectionManager) -> anyhow::Result<()> {
            self.record("finalize");
            if self.finalize_fails {
                anyhow::bail!("mock commit failure");
            }
            Ok(())
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
        let mut manager = manager();
        manager.register(MockTool::new("plain", None));
        manager.register(MockTool::new("keyed", Some('g')));

        let mut switcher = manager.switcher.lock().unwrap();
        assert!(switcher.handle_key(&key_press('g'), Modifiers::default()));
        assert_eq!(switcher.pending, Some(ToolId(1)));
    }

    #[test]
    fn update_applies_pending_switch() {
        let mut manager = manager();
        let tool = MockTool::new("keyed", Some('g'));
        let calls = Arc::clone(&tool.calls);
        manager.register(MockTool::new("plain", None));
        manager.register(tool);

        let scene = Scene::default();
        let mut selection = SelectionManager::new();
        manager.switcher.lock().unwrap().handle_key(&key_press('g'), Modifiers::default());
        manager.update(&scene, &mut selection);

        assert_eq!(manager.active_tool().unwrap().info().id, "keyed");
        assert_eq!(*calls.lock().unwrap(), ["activate"]);
    }

    #[test]
    fn update_same_tool_pending_is_noop() {
        let mut manager = manager();
        let tool = MockTool::new("keyed", Some('g'));
        let calls = Arc::clone(&tool.calls);
        let id = manager.register(tool);
        let mut selection = SelectionManager::new();
        manager.activate(Some(id), &mut selection);

        let scene = Scene::default();
        manager.switcher.lock().unwrap().handle_key(&key_press('g'), Modifiers::default());
        manager.update(&scene, &mut selection);

        assert_eq!(manager.active_tool().unwrap().info().id, "keyed");
        assert!(!calls.lock().unwrap().contains(&"deactivate"));
    }

    #[test]
    fn switching_tools_finalizes_before_deactivating() {
        let mut manager = manager();
        let outgoing = MockTool::new("outgoing", None);
        let calls = Arc::clone(&outgoing.calls);
        let first = manager.register(outgoing);
        let second = manager.register(MockTool::new("incoming", None));

        let mut selection = SelectionManager::new();
        manager.activate(Some(first), &mut selection);
        manager.activate(Some(second), &mut selection);

        assert_eq!(*calls.lock().unwrap(), ["activate", "finalize", "deactivate"]);
        assert_eq!(manager.active_id(), Some(second));
    }

    #[test]
    fn discard_active_skips_finalize() {
        let mut manager = manager();
        let tool = MockTool::new("tool", None);
        let calls = Arc::clone(&tool.calls);
        let id = manager.register(tool);

        let mut selection = SelectionManager::new();
        manager.activate(Some(id), &mut selection);
        manager.discard_active();

        assert_eq!(*calls.lock().unwrap(), ["activate", "deactivate"]);
        assert_eq!(manager.active_id(), None);
    }

    #[test]
    fn failed_finalize_still_deactivates_and_switches() {
        let mut manager = manager();
        let tool = MockTool::failing("failing");
        let calls = Arc::clone(&tool.calls);
        let id = manager.register(tool);

        let mut selection = SelectionManager::new();
        manager.activate(Some(id), &mut selection);
        manager.activate(None, &mut selection);

        assert_eq!(*calls.lock().unwrap(), ["activate", "finalize", "deactivate"]);
        assert_eq!(manager.active_id(), None);
    }
}
