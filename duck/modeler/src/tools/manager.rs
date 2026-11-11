use std::sync::{Arc, Mutex, MutexGuard};

use duck_engine_viewer::scene::Scene;
use duck_engine_viewer::event::{DeviceEvent, Event, EventContext, EventDispatcher};
use duck_engine_viewer::operator::{
    HandleInput, HandleOutcome, HandleSet, Operator, SelectionKinds, SelectionOperator,
};
use duck_engine_viewer::selection::SelectionManager;

use crate::cursor::Cursor3d;
use crate::labels::Label;
use crate::notifications::Notifications;
use crate::tools::{ModelingTool, ToolInfo};

/// The kinds a click selects until the user picks others: whole parts, then
/// their faces and edges.
const DEFAULT_KINDS: SelectionKinds =
    SelectionKinds::NODE.union(SelectionKinds::FACE).union(SelectionKinds::EDGE);

/// What a click may select, as the selection-kind buttons show it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectionFilter {
    /// The kinds a click selects now.
    pub kinds: SelectionKinds,
    /// The kinds the active tool acts on: those a click may be set to.
    pub allowed: SelectionKinds,
}

/// Opaque handle to a registered tool. Minted only by [`ToolManager::register`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ToolId(usize);

/// The single dispatcher-registered operator for all modeling tools.
///
/// Forwards events to the active tool, if any, and drives the handles that tool
/// asks for. Registered once at startup, in front of the selection/navigation
/// operators.
#[derive(Default)]
struct ToolHost {
    active: Option<Arc<Mutex<dyn ModelingTool>>>,
    /// Scene-side state of the active tool's handles.
    handles: HandleSet,
    /// Pointer state of a drag on one of them.
    input: HandleInput,
}

impl ToolHost {
    /// Drops any handle drag in progress, for teardown that happens outside
    /// dispatch. The handles themselves go on the next frame's sync.
    fn abort_drag(&mut self) {
        self.input.abort();
    }
}

impl Operator for ToolHost {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        // Frame tick: bring the handle set in line with what the active tool
        // wants. No tool means an empty set, which is also how a tool switch
        // tears its handles down — nothing else has to plumb a scene through.
        // Runs during a drag too, so a grip tracks the face it is moving.
        if matches!(event, Event::Device(DeviceEvent::Update { .. })) {
            let handles = match &self.active {
                Some(tool) => tool.lock().unwrap().handles(),
                None => Vec::new(),
            };
            self.handles.sync(&handles, &ctx.scene);
        }

        let Some(tool) = self.active.clone() else { return false };

        // Handles see pointer events before the tool does, so a click that
        // lands on one never reaches the tool underneath.
        match self.input.dispatch(event, &mut self.handles, ctx) {
            HandleOutcome::Ignored => {}
            HandleOutcome::Consumed => return true,
            HandleOutcome::Event(handle_event) => {
                tool.lock().unwrap().on_handle(&handle_event, ctx);
                return true;
            }
        }

        tool.lock().unwrap().dispatch(event, ctx)
    }

    fn name(&self) -> &str {
        "ToolHost"
    }
}

/// Owns the registered modeling tools and everything generic about driving
/// them.
/// 
/// Handles activation/deactivation, the kinds the always-on selection operator
/// selects, auto-return to selection when a tool finishes, and the 3D cursor.
/// Adding a tool to the modeler is implementing
/// [`ModelingTool`] plus one [`ToolManager::register`] call.
pub struct ToolManager {
    tools: Vec<Arc<Mutex<dyn ModelingTool>>>,
    /// `None` means plain selection mode.
    active: Option<ToolId>,
    host: Arc<Mutex<ToolHost>>,
    sel_op: Arc<Mutex<SelectionOperator>>,
    /// The kinds the user chose to select. A tool narrows them while active.
    sel_kinds: SelectionKinds,
    /// The modeler-owned 3D cursor, driven each frame from the active tool.
    cursor: Cursor3d,
    /// Reports failed implicit commits, so no tool has to.
    notifications: Notifications,
}

impl ToolManager {
    pub fn new(sel_op: Arc<Mutex<SelectionOperator>>, notifications: Notifications) -> Self {
        sel_op.lock().unwrap().kinds = DEFAULT_KINDS;
        Self {
            tools: Vec::new(),
            active: None,
            host: Arc::new(Mutex::new(ToolHost::default())),
            sel_op,
            sel_kinds: DEFAULT_KINDS,
            cursor: Cursor3d::default(),
            notifications,
        }
    }

    /// Registers the forwarding host with the dispatcher, ahead of every
    /// other operator.
    pub fn install(&self, dispatcher: &mut EventDispatcher) {
        dispatcher.push_front(Arc::clone(&self.host));
    }

    pub fn register<T: ModelingTool>(&mut self, tool: T) -> ToolId {
        let id = ToolId(self.tools.len());
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

        // A handle drag must not outlive the tool whose parameters it edits.
        self.host.lock().unwrap().abort_drag();

        // Locks must be taken strictly one at a time
        if let Some(old) = self.active {
            if let Some(selection) = finalize {
                self.finalize_tool(old, selection);
            }
            self.tools[old.0].lock().unwrap().deactivate();
        }

        self.host.lock().unwrap().active = id.map(|i| Arc::clone(&self.tools[i.0]));

        let selects = id.and_then(|i| {
            let mut tool = self.tools[i.0].lock().unwrap();
            tool.activate();
            tool.selects()
        });
        // Those of the user's kinds the tool acts on, or all it acts on if
        // none: any narrowing during the last run is dropped.
        let kinds = match selects {
            Some(acts_on) if self.sel_kinds.intersects(acts_on) => self.sel_kinds & acts_on,
            Some(acts_on) => acts_on,
            None => self.sel_kinds,
        };
        self.sel_op.lock().unwrap().kinds = kinds;

        self.active = id;
    }

    /// Makes `kind` the only kind a click selects, or with `extend` adds or
    /// removes it, never leaving none. While a tool that acts on some kinds
    /// only is active, the change lasts until it ends, and other kinds are
    /// refused.
    pub fn select_kind(&mut self, kind: SelectionKinds, extend: bool) {
        let acts_on = self.active_selects();
        if acts_on.is_some_and(|acts_on| !acts_on.contains(kind)) {
            return;
        }
        let mut sel_op = self.sel_op.lock().unwrap();
        let kinds = if extend { sel_op.kinds ^ kind } else { kind };
        if kinds.is_empty() {
            return;
        }
        sel_op.kinds = kinds;
        if acts_on.is_none() {
            self.sel_kinds = kinds;
        }
    }

    /// What a click selects now, and what it may be set to.
    pub fn selection_filter(&self) -> SelectionFilter {
        SelectionFilter {
            kinds: self.sel_op.lock().unwrap().kinds,
            allowed: self.active_selects().unwrap_or(SelectionKinds::all()),
        }
    }

    /// The kinds the active tool acts on, or `None` if no tool or one that
    /// keeps the user's is active.
    fn active_selects(&self) -> Option<SelectionKinds> {
        self.active.and_then(|i| self.tools[i.0].lock().unwrap().selects())
    }

    /// Commits `id`'s pending result, reporting a failure on its behalf. The
    /// caller's `deactivate` then clears whatever a failed commit left behind.
    fn finalize_tool(&self, id: ToolId, selection: &mut SelectionManager) {
        let mut tool = self.tools[id.0].lock().unwrap();
        if let Err(e) = tool.finalize(selection) {
            self.notifications.failure(tool.info().id, &e);
        }
    }

    /// Per-frame update. Should be called every frame.
    pub fn update(&mut self, scene: &Scene, selection: &mut SelectionManager) {
        if self.active.is_some_and(|i| self.tools[i.0].lock().unwrap().is_finished()) {
            self.activate(None, selection);
        }

        let target = self
            .active
            .and_then(|i| self.tools[i.0].lock().unwrap().cursor_target());
        self.cursor.update(target, scene);
    }

    /// The active tool's text over the viewport, or none in selection mode.
    pub fn labels(&self) -> Vec<Label> {
        self.active_tool().map(|tool| tool.labels()).unwrap_or_default()
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
    use super::*;

    fn manager() -> ToolManager {
        ToolManager::new(
            Arc::new(Mutex::new(SelectionOperator::new())),
            Notifications::default(),
        )
    }

    struct MockTool {
        id: &'static str,
        /// Lifecycle calls in the order they arrived.
        calls: Arc<Mutex<Vec<&'static str>>>,
        /// Whether `finalize` reports a failed commit.
        finalize_fails: bool,
        /// The kinds it acts on.
        selects: Option<SelectionKinds>,
    }

    impl MockTool {
        fn new(id: &'static str) -> Self {
            Self { id, calls: Arc::new(Mutex::new(Vec::new())), finalize_fails: false, selects: None }
        }

        /// A tool acting on `kinds` only.
        fn selecting(id: &'static str, kinds: SelectionKinds) -> Self {
            Self { selects: Some(kinds), ..Self::new(id) }
        }

        /// A tool whose pending result can't be committed.
        fn failing(id: &'static str) -> Self {
            Self { finalize_fails: true, ..Self::new(id) }
        }

        fn record(&self, call: &'static str) {
            self.calls.lock().unwrap().push(call);
        }
    }

    impl crate::tools::ModelingTool for MockTool {
        fn info(&self) -> crate::tools::ToolInfo {
            crate::tools::ToolInfo { id: self.id, icon: ("mock", &[]), shortcut: None }
        }

        fn selects(&self) -> Option<SelectionKinds> {
            self.selects
        }

        fn dispatch(&mut self, _event: &Event, _ctx: &mut EventContext) -> bool {
            false
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

    #[test]
    fn activating_the_active_tool_again_changes_nothing() {
        let mut manager = manager();
        let tool = MockTool::new("tool");
        let calls = Arc::clone(&tool.calls);
        let id = manager.register(tool);

        let mut selection = SelectionManager::new();
        manager.activate(Some(id), &mut selection);
        manager.activate(Some(id), &mut selection);

        assert_eq!(*calls.lock().unwrap(), ["activate"]);
        assert_eq!(manager.active_id(), Some(id));
    }

    #[test]
    fn switching_tools_finalizes_before_deactivating() {
        let mut manager = manager();
        let outgoing = MockTool::new("outgoing");
        let calls = Arc::clone(&outgoing.calls);
        let first = manager.register(outgoing);
        let second = manager.register(MockTool::new("incoming"));

        let mut selection = SelectionManager::new();
        manager.activate(Some(first), &mut selection);
        manager.activate(Some(second), &mut selection);

        assert_eq!(*calls.lock().unwrap(), ["activate", "finalize", "deactivate"]);
        assert_eq!(manager.active_id(), Some(second));
    }

    #[test]
    fn discard_active_skips_finalize() {
        let mut manager = manager();
        let tool = MockTool::new("tool");
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

    // ===== Selection kinds =====

    const NODE: SelectionKinds = SelectionKinds::NODE;
    const FACE: SelectionKinds = SelectionKinds::FACE;
    const EDGE: SelectionKinds = SelectionKinds::EDGE;

    fn kinds(manager: &ToolManager) -> SelectionKinds {
        manager.selection_filter().kinds
    }

    #[test]
    fn a_tool_narrows_the_users_kinds_to_those_it_acts_on() {
        let mut manager = manager();
        let id = manager.register(MockTool::selecting("move", NODE | FACE));
        let mut selection = SelectionManager::new();

        manager.activate(Some(id), &mut selection);
        assert_eq!(kinds(&manager), NODE | FACE);

        manager.activate(None, &mut selection);
        manager.select_kind(FACE, false);
        manager.activate(Some(id), &mut selection);
        assert_eq!(kinds(&manager), FACE);
    }

    #[test]
    fn a_tool_acting_on_none_of_the_users_kinds_uses_its_own() {
        let mut manager = manager();
        let id = manager.register(MockTool::selecting("fillet", EDGE));
        let mut selection = SelectionManager::new();
        manager.select_kind(NODE, false);

        manager.activate(Some(id), &mut selection);
        assert_eq!(manager.selection_filter(), SelectionFilter { kinds: EDGE, allowed: EDGE });
    }

    #[test]
    fn a_tool_with_no_preference_keeps_the_users_kinds() {
        let mut manager = manager();
        let id = manager.register(MockTool::new("box"));
        let mut selection = SelectionManager::new();
        manager.select_kind(FACE, false);

        manager.activate(Some(id), &mut selection);
        assert_eq!(manager.selection_filter(), SelectionFilter { kinds: FACE, allowed: SelectionKinds::all() });
    }

    /// Kinds picked while a tool is active last until it ends.
    #[test]
    fn leaving_a_tool_restores_the_users_kinds() {
        let mut manager = manager();
        let id = manager.register(MockTool::selecting("move", NODE | FACE));
        let mut selection = SelectionManager::new();
        let chosen = kinds(&manager);

        manager.activate(Some(id), &mut selection);
        manager.select_kind(FACE, false);
        assert_eq!(kinds(&manager), FACE);
        manager.discard_active();
        assert_eq!(kinds(&manager), chosen);

        manager.activate(Some(id), &mut selection);
        assert_eq!(kinds(&manager), NODE | FACE);
    }

    #[test]
    fn a_plain_pick_takes_one_kind_and_a_shift_pick_toggles() {
        let mut manager = manager();
        manager.select_kind(FACE, false);
        assert_eq!(kinds(&manager), FACE);
        manager.select_kind(EDGE, true);
        assert_eq!(kinds(&manager), FACE | EDGE);
        manager.select_kind(FACE, true);
        assert_eq!(kinds(&manager), EDGE);
        // The last kind stays.
        manager.select_kind(EDGE, true);
        assert_eq!(kinds(&manager), EDGE);
    }

    #[test]
    fn a_tool_refuses_kinds_it_does_not_act_on() {
        let mut manager = manager();
        let id = manager.register(MockTool::selecting("extrude", FACE | EDGE));
        let mut selection = SelectionManager::new();
        manager.activate(Some(id), &mut selection);

        manager.select_kind(NODE, false);
        manager.select_kind(NODE, true);
        assert_eq!(kinds(&manager), FACE | EDGE);
    }

    #[test]
    fn picks_under_a_tool_with_no_preference_are_the_users() {
        let mut manager = manager();
        let id = manager.register(MockTool::new("box"));
        let mut selection = SelectionManager::new();
        manager.activate(Some(id), &mut selection);

        manager.select_kind(FACE, false);
        manager.activate(None, &mut selection);
        assert_eq!(kinds(&manager), FACE);
    }
}
