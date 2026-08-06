use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use web_time::Instant;

use crate::input::{
    ElementState, Key, KeyEvent, Modifiers, MouseButton, MouseScrollDelta, NamedKey,
    TouchId, TouchPhase,
};
use crate::operator::Operator;

use super::{ArcOperator, DeviceEvent, Event, EventContext};

/// Movement threshold in pixels before a mouse button hold becomes a drag.
const DRAG_THRESHOLD_PIXELS: f32 = 4.0;

/// Maximum time in milliseconds for a button press/release to be considered a click.
const CLICK_TIME_THRESHOLD_MS: u64 = 300;

/// Current touch interaction mode for synthesis.
#[derive(Debug, Clone)]
enum TouchMode {
    /// No active touch interaction
    None,
    /// Single finger is active
    SingleFinger { id: TouchId },
    /// Two fingers are active
    TwoFinger { ids: [TouchId; 2] },
}

/// State for synthesizing mouse events from touch input.
#[derive(Debug, Clone)]
struct TouchSynthState {
    /// Active touch points: id → last known position
    active_touches: HashMap<TouchId, (f64, f64)>,
    /// Current interaction mode
    mode: TouchMode,
    /// Previous pinch distance for delta computation
    prev_pinch_distance: Option<f64>,
}

impl TouchSynthState {
    fn new() -> Self {
        Self {
            active_touches: HashMap::new(),
            mode: TouchMode::None,
            prev_pinch_distance: None,
        }
    }
}

/// State tracking for a mouse button that is currently pressed.
#[derive(Debug, Clone)]
struct ButtonState {
    /// Position where the button was initially pressed (in physical pixels)
    down_position: (f32, f32),
    /// Time when the button was pressed
    down_time: Instant,
    /// Whether this button press has transitioned into a drag
    is_dragging: bool,
    /// Total distance dragged since button down (in pixels)
    distance_dragged: f32,
}

/// Manages operators and synthesizes high-level input events.
///
/// The dispatcher holds an ordered list of operators as [`Arc<Mutex<T>>`] values, type-erased
/// via the internal [`ArcOperator`] trait. On each call to [`dispatch`](EventDispatcher::dispatch) it:
///
/// 1. Snapshots the current modifier state into [`EventContext::modifiers`].
/// 2. Updates internal state and synthesizes derived events (drag start/end,
///    click, touch-to-mouse mapping).
/// 3. Calls each operator's dispatch method in priority order until one returns
///    `true` (consumes the event) or all have been called.
///
/// Operators at the front of the list have the highest priority.
///
/// Register operators with [`push_front`](Self::push_front) or [`push_back`](Self::push_back),
/// which accept `Arc<Mutex<T>>` for any `T: Operator`. The caller retains a clone of the `Arc`
/// for direct typed access (locking) without involving the dispatcher.
pub struct EventDispatcher {
    operators: Vec<Arc<dyn ArcOperator>>,
    /// State for each currently-pressed mouse button
    button_states: HashMap<MouseButton, ButtonState>,
    /// Current cursor position in physical pixels (from CursorMoved events)
    current_cursor_position: Option<(f32, f32)>,
    /// State for synthesizing mouse events from touch input
    touch_state: TouchSynthState,
    /// Currently held keyboard modifier keys (Shift, Ctrl, Alt, Super)
    current_modifiers: Modifiers,
}

impl EventDispatcher {
    /// Creates a new dispatcher with no operators.
    pub(crate) fn new() -> Self {
        Self {
            operators: Vec::new(),
            button_states: HashMap::new(),
            current_cursor_position: None,
            touch_state: TouchSynthState::new(),
            current_modifiers: Modifiers::default(),
        }
    }

    // ── Operator management ──────────────────────────────────────────────────

    /// Adds an operator to the front of the stack (highest priority).
    ///
    /// The caller should retain a clone of `op` for direct typed access after registration.
    pub fn push_front<T: Operator>(&mut self, op: Arc<Mutex<T>>) {
        self.operators.insert(0, op);
    }

    /// Adds an operator to the back of the stack (lowest priority).
    ///
    /// The caller should retain a clone of `op` for direct typed access after registration.
    pub fn push_back<T: Operator>(&mut self, op: Arc<Mutex<T>>) {
        self.operators.push(op);
    }

    /// Removes the operator identified by `op`.
    ///
    /// Uses pointer equality: `op` must be a clone of the `Arc` passed to [`push_front`](Self::push_front)
    /// or [`push_back`](Self::push_back). Returns `true` if found and removed.
    pub fn remove<T: Operator>(&mut self, op: &Arc<Mutex<T>>) -> bool {
        if let Some(pos) = self.find_pos(op) {
            self.operators.remove(pos);
            true
        } else {
            false
        }
    }

    /// Moves the operator identified by `op` to the front (highest priority).
    ///
    /// Returns `true` if the operator was found.
    pub fn move_to_front<T: Operator>(&mut self, op: &Arc<Mutex<T>>) -> bool {
        if let Some(pos) = self.find_pos(op) {
            if pos > 0 {
                let entry = self.operators.remove(pos);
                self.operators.insert(0, entry);
            }
            true
        } else {
            false
        }
    }

    /// Moves the operator identified by `op` to the back (lowest priority).
    ///
    /// Returns `true` if the operator was found.
    pub fn move_to_back<T: Operator>(&mut self, op: &Arc<Mutex<T>>) -> bool {
        if let Some(pos) = self.find_pos(op) {
            if pos < self.operators.len() - 1 {
                let entry = self.operators.remove(pos);
                self.operators.push(entry);
            }
            true
        } else {
            false
        }
    }

    /// Swaps the positions of two operators.
    ///
    /// Returns `true` if both operators were found. Returns `true` without
    /// changing anything if both arcs point to the same operator.
    pub fn swap<A: Operator, B: Operator>(&mut self, a: &Arc<Mutex<A>>, b: &Arc<Mutex<B>>) -> bool {
        let pos_a = self.find_pos(a);
        let pos_b = self.find_pos(b);
        match (pos_a, pos_b) {
            (Some(p1), Some(p2)) if p1 != p2 => {
                self.operators.swap(p1, p2);
                true
            }
            (Some(_), Some(_)) => true,
            _ => false,
        }
    }

    /// Returns the number of operators.
    pub fn len(&self) -> usize {
        self.operators.len()
    }

    /// Returns `true` if there are no operators.
    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }

    /// Returns the names of all operators in priority order (front to back).
    pub fn iter_names(&self) -> impl Iterator<Item = String> + '_ {
        self.operators.iter().map(|op| op.name())
    }

    /// Returns the position of an operator. Position 0 is highest priority.
    pub fn position<T: Operator>(&self, op: &Arc<Mutex<T>>) -> Option<usize> {
        self.find_pos(op)
    }

    fn find_pos<T: Operator>(&self, op: &Arc<Mutex<T>>) -> Option<usize> {
        let target = Arc::as_ptr(op) as *const ();
        self.operators.iter().position(|stored| stored.as_ptr() == target)
    }

    // ── Dispatch ─────────────────────────────────────────────────────────────

    /// Dispatches an event to all operators in priority order.
    ///
    /// Before dispatching, modifier state is snapshotted into `ctx.modifiers` and
    /// internal synthesis state is updated (drag, click, touch-to-mouse mapping).
    /// Synthesized events are dispatched to operators before the original event.
    ///
    /// Returns `true` if any operator consumed the event.
    pub fn dispatch<'c>(&mut self, event: &Event, ctx: &mut EventContext<'c>) -> bool {
        // Snapshot modifier state before processing so both synthesized events (dispatched
        // inside process_event) and the original event see the same modifier state.
        ctx.modifiers = self.current_modifiers;

        // Update state and synthesize high-level events (drag, click, etc.). Only
        // device events drive synthesis; app events flow straight to the operators.
        if let Event::Device(device_event) = event {
            self.process_event(device_event, ctx);
        }

        // Dispatch the original event, then re-dispatch anything operators emitted.
        let consumed = self.dispatch_to_operators(event, ctx);
        self.drain_emitted(ctx);
        consumed
    }

    /// Re-dispatches events emitted via [`EventContext::emit`] back through the operator
    /// stack, draining the queue until empty so cascaded emits (an emitted event causing
    /// further emits) are also delivered.
    fn drain_emitted<'c>(&mut self, ctx: &mut EventContext<'c>) {
        loop {
            let batch = std::mem::take(&mut ctx.emit_queue);
            if batch.is_empty() {
                break;
            }
            for event in &batch {
                self.dispatch_to_operators(event, ctx);
            }
        }
    }

    /// Dispatches an event to all operators without updating synthesis state.
    fn dispatch_to_operators<'c>(&mut self, event: &Event, ctx: &mut EventContext<'c>) -> bool {
        for op in &self.operators {
            if op.dispatch(event, ctx) {
                return true;
            }
        }
        false
    }

    // ── Synthesis ────────────────────────────────────────────────────────────

    /// Processes a device event to update internal state and synthesize high-level events.
    fn process_event<'c>(&mut self, event: &DeviceEvent, ctx: &mut EventContext<'c>) {
        match event {
            DeviceEvent::CursorMoved { position } => {
                self.process_cursor_moved(*position);
            }
            DeviceEvent::MouseInput { state, button } => {
                self.process_mouse_input(*state, *button, ctx);
            }
            DeviceEvent::MouseMotion { delta } => {
                self.process_mouse_motion(*delta, ctx);
            }
            DeviceEvent::Touch { id, phase, position } => {
                self.process_touch(*id, *phase, *position, ctx);
            }
            DeviceEvent::KeyboardInput { event: key_event, .. } => {
                self.process_modifier_key(key_event);
            }
            _ => {}
        }
    }

    /// Tracks modifier key (Shift, Ctrl, Alt, Super) press/release state.
    fn process_modifier_key(&mut self, key_event: &KeyEvent) {
        let pressed = key_event.state == ElementState::Pressed;
        match &key_event.logical_key {
            Key::Named(NamedKey::Shift) => self.current_modifiers.shift = pressed,
            Key::Named(NamedKey::Control) => self.current_modifiers.control = pressed,
            Key::Named(NamedKey::Alt) => self.current_modifiers.alt = pressed,
            Key::Named(NamedKey::Super) => self.current_modifiers.super_key = pressed,
            _ => {}
        }
    }

    /// Processes CursorMoved events to update cursor position tracking.
    fn process_cursor_moved(&mut self, position: (f64, f64)) {
        self.current_cursor_position = Some((position.0 as f32, position.1 as f32));
    }

    /// Processes MouseInput events to track button press/release and synthesize click events.
    fn process_mouse_input<'c>(
        &mut self,
        state: ElementState,
        button: MouseButton,
        ctx: &mut EventContext<'c>,
    ) {
        match state {
            ElementState::Pressed => {
                // Button pressed - start tracking
                if let Some(pos) = self.current_cursor_position {
                    self.button_states.insert(
                        button,
                        ButtonState {
                            down_position: pos,
                            down_time: Instant::now(),
                            is_dragging: false,
                            distance_dragged: 0.0,
                        },
                    );
                }
            }
            ElementState::Released => {
                // Button released - check if it was a click or drag
                if let Some(button_state) = self.button_states.remove(&button) {
                    if button_state.is_dragging {
                        self.synthesize_drag_end(button, button_state, ctx);
                    } else {
                        self.synthesize_click_if_quick(button, button_state, ctx);
                    }
                }
            }
        }
    }

    /// Synthesizes a MouseDragEnd event when a dragged button is released.
    fn synthesize_drag_end<'c>(
        &mut self,
        button: MouseButton,
        button_state: ButtonState,
        ctx: &mut EventContext<'c>,
    ) {
        if let Some(end_pos) = self.current_cursor_position {
            let drag_end = DeviceEvent::MouseDragEnd {
                button,
                start_pos: button_state.down_position,
                end_pos,
            };
            self.dispatch_to_operators(&Event::Device(drag_end), ctx);
        }
    }

    /// Synthesizes a MouseClick event if the button was released quickly.
    fn synthesize_click_if_quick<'c>(
        &mut self,
        button: MouseButton,
        button_state: ButtonState,
        ctx: &mut EventContext<'c>,
    ) {
        let duration = button_state.down_time.elapsed();
        if duration.as_millis() as u64 <= CLICK_TIME_THRESHOLD_MS {
            let click = DeviceEvent::MouseClick {
                button,
                position: button_state.down_position,
                duration_ms: duration.as_millis() as u64,
            };
            self.dispatch_to_operators(&Event::Device(click), ctx);
        }
    }

    /// Processes a touch event and synthesizes corresponding mouse events.
    ///
    /// Touch gestures are mapped to mouse events so existing operators work unchanged:
    /// - Single-finger drag → left mouse drag (orbit)
    /// - Two-finger drag → right mouse drag (pan)
    /// - Two-finger pinch → mouse wheel (zoom)
    /// - Single tap → left mouse click (selection)
    fn process_touch<'c>(
        &mut self,
        id: TouchId,
        phase: TouchPhase,
        position: (f64, f64),
        ctx: &mut EventContext<'c>,
    ) {
        match phase {
            TouchPhase::Started => self.process_touch_started(id, position, ctx),
            TouchPhase::Moved => self.process_touch_moved(id, position, ctx),
            TouchPhase::Ended | TouchPhase::Cancelled => self.process_touch_ended(id, ctx),
        }
    }

    /// Handles a new finger touching the screen.
    fn process_touch_started<'c>(
        &mut self,
        id: TouchId,
        position: (f64, f64),
        ctx: &mut EventContext<'c>,
    ) {
        self.touch_state.active_touches.insert(id, position);

        match &self.touch_state.mode {
            TouchMode::None => {
                // First finger: start single-finger mode (maps to left mouse)
                self.touch_state.mode = TouchMode::SingleFinger { id };
                self.dispatch_synthetic(
                    &DeviceEvent::CursorMoved { position },
                    ctx,
                );
                self.dispatch_synthetic(
                    &DeviceEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                    },
                    ctx,
                );
            }
            TouchMode::SingleFinger { id: first_id } => {
                // Second finger: transition to two-finger mode
                let first_id = *first_id;

                // Release the left button to cancel any single-finger drag
                self.dispatch_synthetic(
                    &DeviceEvent::MouseInput {
                        state: ElementState::Released,
                        button: MouseButton::Left,
                    },
                    ctx,
                );

                // Calculate initial pinch distance and center
                if let (Some(&first_pos), Some(&second_pos)) = (
                    self.touch_state.active_touches.get(&first_id),
                    self.touch_state.active_touches.get(&id),
                ) {
                    let center = touch_center(first_pos, second_pos);
                    let distance = touch_distance(first_pos, second_pos);
                    self.touch_state.prev_pinch_distance = Some(distance);
                    self.touch_state.mode = TouchMode::TwoFinger {
                        ids: [first_id, id],
                    };

                    // Start right-button press at center for pan
                    self.dispatch_synthetic(
                        &DeviceEvent::CursorMoved { position: center },
                        ctx,
                    );
                    self.dispatch_synthetic(
                        &DeviceEvent::MouseInput {
                            state: ElementState::Pressed,
                            button: MouseButton::Right,
                        },
                        ctx,
                    );
                }
            }
            TouchMode::TwoFinger { .. } => {
                // Third+ finger: ignore
            }
        }
    }

    /// Handles a finger moving on the screen.
    fn process_touch_moved<'c>(
        &mut self,
        id: TouchId,
        position: (f64, f64),
        ctx: &mut EventContext<'c>,
    ) {
        let old_position = self.touch_state.active_touches.get(&id).copied();
        self.touch_state.active_touches.insert(id, position);

        match &self.touch_state.mode {
            TouchMode::None => {}
            TouchMode::SingleFinger { id: finger_id } => {
                if id != *finger_id {
                    return;
                }
                if let Some(old_pos) = old_position {
                    let delta = (position.0 - old_pos.0, position.1 - old_pos.1);
                    self.dispatch_synthetic(
                        &DeviceEvent::CursorMoved { position },
                        ctx,
                    );
                    self.dispatch_synthetic(
                        &DeviceEvent::MouseMotion { delta },
                        ctx,
                    );
                }
            }
            TouchMode::TwoFinger { ids } => {
                let [id0, id1] = *ids;
                if id != id0 && id != id1 {
                    return;
                }
                if let (Some(&pos0), Some(&pos1)) = (
                    self.touch_state.active_touches.get(&id0),
                    self.touch_state.active_touches.get(&id1),
                ) {
                    let center = touch_center(pos0, pos1);
                    let distance = touch_distance(pos0, pos1);

                    // Emit pan as right-drag (center movement)
                    if let Some(old_pos) = old_position {
                        // Compute how the center moved due to this finger's movement
                        let half_delta = (
                            (position.0 - old_pos.0) * 0.5,
                            (position.1 - old_pos.1) * 0.5,
                        );
                        self.dispatch_synthetic(
                            &DeviceEvent::CursorMoved { position: center },
                            ctx,
                        );
                        self.dispatch_synthetic(
                            &DeviceEvent::MouseMotion { delta: half_delta },
                            ctx,
                        );
                    }

                    // Emit zoom as mouse wheel (pinch distance change)
                    if let Some(prev_distance) = self.touch_state.prev_pinch_distance
                        && prev_distance > 0.0 {
                            let delta = distance - prev_distance;
                            // Scale: positive delta = fingers moving apart = zoom in
                            if delta.abs() > 0.5 {
                                self.dispatch_synthetic(
                                    &DeviceEvent::MouseWheel {
                                        delta: MouseScrollDelta::PixelDelta(0.0, delta as f32),
                                    },
                                    ctx,
                                );
                            }
                        }
                    self.touch_state.prev_pinch_distance = Some(distance);
                }
            }
        }
    }

    /// Handles a finger leaving the screen.
    fn process_touch_ended<'c>(
        &mut self,
        id: TouchId,
        ctx: &mut EventContext<'c>,
    ) {
        self.touch_state.active_touches.remove(&id);

        match &self.touch_state.mode {
            TouchMode::None => {}
            TouchMode::SingleFinger { id: finger_id } => {
                if id != *finger_id {
                    return;
                }
                // Release left button
                self.dispatch_synthetic(
                    &DeviceEvent::MouseInput {
                        state: ElementState::Released,
                        button: MouseButton::Left,
                    },
                    ctx,
                );
                self.touch_state.mode = TouchMode::None;
            }
            TouchMode::TwoFinger { ids } => {
                let [id0, id1] = *ids;
                if id != id0 && id != id1 {
                    return;
                }

                // Release right button (end pan)
                self.dispatch_synthetic(
                    &DeviceEvent::MouseInput {
                        state: ElementState::Released,
                        button: MouseButton::Right,
                    },
                    ctx,
                );
                self.touch_state.prev_pinch_distance = None;

                // Find remaining finger and transition to single-finger mode
                let remaining_id = if id == id0 { id1 } else { id0 };
                if let Some(&remaining_pos) = self.touch_state.active_touches.get(&remaining_id) {
                    self.touch_state.mode = TouchMode::SingleFinger { id: remaining_id };
                    self.dispatch_synthetic(
                        &DeviceEvent::CursorMoved {
                            position: remaining_pos,
                        },
                        ctx,
                    );
                    self.dispatch_synthetic(
                        &DeviceEvent::MouseInput {
                            state: ElementState::Pressed,
                            button: MouseButton::Left,
                        },
                        ctx,
                    );
                } else {
                    self.touch_state.mode = TouchMode::None;
                }
            }
        }
    }

    /// Dispatches a synthesized event through the full processing pipeline.
    ///
    /// This calls `process_event` so the synthesized event updates button/cursor
    /// state and triggers further synthesis (e.g., drag and click events).
    fn dispatch_synthetic<'c>(
        &mut self,
        event: &DeviceEvent,
        ctx: &mut EventContext<'c>,
    ) {
        self.process_event(event, ctx);
        self.dispatch_to_operators(&Event::Device(event.clone()), ctx);
    }

    /// Processes MouseMotion events to track drag distance and synthesize drag events.
    fn process_mouse_motion<'c>(
        &mut self,
        delta: (f64, f64),
        ctx: &mut EventContext<'c>,
    ) {
        let delta_magnitude = ((delta.0 * delta.0 + delta.1 * delta.1) as f32).sqrt();

        // Collect buttons that need drag events (to avoid borrow issues)
        let mut drag_events = Vec::new();

        for (button, button_state) in &mut self.button_states {
            button_state.distance_dragged += delta_magnitude;

            if !button_state.is_dragging && button_state.distance_dragged > DRAG_THRESHOLD_PIXELS {
                // Transition to dragging state
                button_state.is_dragging = true;
                if let Some(current_pos) = self.current_cursor_position {
                    drag_events.push(DeviceEvent::MouseDragStart {
                        button: *button,
                        start_pos: button_state.down_position,
                        current_pos,
                    });
                }
            } else if button_state.is_dragging {
                // Already dragging - queue MouseDrag event
                if let Some(current_pos) = self.current_cursor_position {
                    drag_events.push(DeviceEvent::MouseDrag {
                        button: *button,
                        start_pos: button_state.down_position,
                        current_pos,
                        delta: (delta.0 as f32, delta.1 as f32),
                    });
                }
            }
        }

        // Dispatch all queued drag events
        for drag_event in drag_events {
            self.dispatch_to_operators(&Event::Device(drag_event), ctx);
        }
    }
}

/// Compute the center point between two touch positions.
fn touch_center(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
}

/// Compute the Euclidean distance between two touch positions.
fn touch_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::AppEvent;
    use crate::input::{ElementState, MouseButton};
    use crate::scene::resource::NodePayload;
    use crate::scene::{Scene, SceneData};
    use crate::selection::SelectionManager;
    use duck_engine_scene::resource::NodeFlags;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn create_mock_context_parts() -> (Option<(f32, f32)>, Scene, SelectionManager) {
        use crate::scene::PositionedCamera;
        use duck_engine_common::Vector3;
        let camera = PositionedCamera {
            eye: (0.0, 0.0, 1.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: Vector3::unit_y(),
            aspect: 800.0 / 600.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        };
        let mut scene = SceneData::new();
        let cam_id = scene.add_node(
            None, None, camera.to_node_transform(), NodeFlags::NONE
        ).unwrap().id();
        scene.set_node_payload(cam_id, NodePayload::Camera(camera.projection()));
        scene.set_active_camera(Some(cam_id));
        (None, Scene::new(scene), SelectionManager::new())
    }

    fn make_context(parts: &mut (Option<(f32, f32)>, Scene, SelectionManager)) -> EventContext<'_> {
        EventContext {
            size: (800, 600),
            cursor_position: &mut parts.0,
            scene: parts.1.clone(),
            selection: &mut parts.2,
            modifiers: Default::default(),
            emit_queue: Vec::new(),
        }
    }

    fn arc_op<T: Operator>(op: T) -> Arc<Mutex<T>> {
        Arc::new(Mutex::new(op))
    }

    // Simple counter operator: increments counter on every event, optionally consumes.
    struct CounterOp {
        counter: Rc<Cell<u32>>,
        consumes: bool,
    }
    impl Operator for CounterOp {
        fn dispatch(&mut self, _: &Event, _: &mut EventContext) -> bool {
            self.counter.set(self.counter.get() + 1);
            self.consumes
        }
        fn name(&self) -> &str { "CounterOp" }
    }

    struct NamedOp(&'static str);
    impl Operator for NamedOp {
        fn dispatch(&mut self, _: &Event, _: &mut EventContext) -> bool { false }
        fn name(&self) -> &str { self.0 }
    }

    struct OtherOp;
    impl Operator for OtherOp {
        fn dispatch(&mut self, _: &Event, _: &mut EventContext) -> bool { false }
        fn name(&self) -> &str { "OtherOp" }
    }

    // ── Operator management ──────────────────────────────────────────────────

    #[test]
    fn new_has_no_operators() {
        assert!(EventDispatcher::new().is_empty());
    }

    #[test]
    fn push_back_ordering() {
        let mut d = EventDispatcher::new();
        d.push_back(arc_op(NamedOp("A")));
        d.push_back(arc_op(OtherOp));
        let names: Vec<String> = d.iter_names().collect();
        assert_eq!(names, ["A", "OtherOp"]);
    }

    #[test]
    fn push_front_ordering() {
        let mut d = EventDispatcher::new();
        d.push_back(arc_op(NamedOp("A")));
        d.push_front(arc_op(OtherOp));
        let names: Vec<String> = d.iter_names().collect();
        assert_eq!(names, ["OtherOp", "A"]);
    }

    #[test]
    fn remove_by_arc() {
        let mut d = EventDispatcher::new();
        let a = arc_op(NamedOp("A"));
        d.push_back(a.clone());
        d.push_back(arc_op(OtherOp));
        assert!(d.remove(&a));
        assert_eq!(d.len(), 1);
        assert_eq!(d.iter_names().next().unwrap(), "OtherOp");
    }

    #[test]
    fn remove_nonexistent_returns_false() {
        let mut d = EventDispatcher::new();
        d.push_back(arc_op(NamedOp("A")));
        let unregistered = arc_op(OtherOp);
        assert!(!d.remove(&unregistered));
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn move_to_front() {
        let mut d = EventDispatcher::new();
        let a = arc_op(NamedOp("A"));
        d.push_back(a.clone());
        d.push_back(arc_op(OtherOp));
        assert!(d.move_to_back(&a));
        let names: Vec<String> = d.iter_names().collect();
        assert_eq!(names, ["OtherOp", "A"]);
    }

    #[test]
    fn move_to_back() {
        let mut d = EventDispatcher::new();
        let other = arc_op(OtherOp);
        d.push_back(arc_op(NamedOp("A")));
        d.push_back(other.clone());
        assert!(d.move_to_front(&other));
        let names: Vec<String> = d.iter_names().collect();
        assert_eq!(names, ["OtherOp", "A"]);
    }

    #[test]
    fn swap_operators() {
        let mut d = EventDispatcher::new();
        let a = arc_op(NamedOp("A"));
        let other = arc_op(OtherOp);
        d.push_back(a.clone());
        d.push_back(other.clone());
        assert!(d.swap(&a, &other));
        let names: Vec<String> = d.iter_names().collect();
        assert_eq!(names, ["OtherOp", "A"]);
    }

    #[test]
    fn multiple_instances_of_same_type() {
        let mut d = EventDispatcher::new();
        d.push_back(arc_op(NamedOp("first")));
        d.push_back(arc_op(NamedOp("second")));
        assert_eq!(d.len(), 2);
        let names: Vec<String> = d.iter_names().collect();
        assert_eq!(names, ["first", "second"]);
    }

    #[test]
    fn mutation_via_retained_arc() {
        let mut d = EventDispatcher::new();
        let op = arc_op(NamedOp("hello"));
        d.push_back(op.clone());
        op.lock().unwrap().0 = "world";
        assert_eq!(d.iter_names().next().unwrap(), "world");
    }

    // ── Dispatch propagation ─────────────────────────────────────────────────

    #[test]
    fn dispatch_calls_all_non_consuming_operators() {
        let mut d = EventDispatcher::new();
        let c1 = Rc::new(Cell::new(0u32));
        let c2 = Rc::new(Cell::new(0u32));
        d.push_back(arc_op(CounterOp { counter: c1.clone(), consumes: false }));
        d.push_back(arc_op(CounterOp { counter: c2.clone(), consumes: false }));

        let mut parts = create_mock_context_parts();
        let mut ctx = make_context(&mut parts);
        let consumed = d.dispatch(&Event::Device(DeviceEvent::Test), &mut ctx);

        assert_eq!(c1.get(), 1);
        assert_eq!(c2.get(), 1);
        assert!(!consumed);
    }

    #[test]
    fn dispatch_stops_at_consuming_operator() {
        let mut d = EventDispatcher::new();
        let c1 = Rc::new(Cell::new(0u32));
        let c2 = Rc::new(Cell::new(0u32));
        d.push_back(arc_op(CounterOp { counter: c1.clone(), consumes: true }));
        d.push_back(arc_op(CounterOp { counter: c2.clone(), consumes: false }));

        let mut parts = create_mock_context_parts();
        let mut ctx = make_context(&mut parts);
        let consumed = d.dispatch(&Event::Device(DeviceEvent::Test), &mut ctx);

        assert_eq!(c1.get(), 1);
        assert_eq!(c2.get(), 0); // never reached
        assert!(consumed);
    }

    #[test]
    fn dispatch_priority_order_front_to_back() {
        let order = Rc::new(Cell::new(0u32));

        struct RecordOp(Rc<Cell<u32>>, u32);
        impl Operator for RecordOp {
            fn dispatch(&mut self, _: &Event, _: &mut EventContext) -> bool {
                self.0.set(self.0.get() * 10 + self.1);
                false
            }
            fn name(&self) -> &str { "RecordOp" }
        }

        let mut d = EventDispatcher::new();
        d.push_back(arc_op(RecordOp(order.clone(), 1)));
        d.push_back(arc_op(RecordOp(order.clone(), 2)));

        let mut parts = create_mock_context_parts();
        let mut ctx = make_context(&mut parts);
        d.dispatch(&Event::Device(DeviceEvent::Test), &mut ctx);

        assert_eq!(order.get(), 12); // 1 fired first, then 2
    }

    #[test]
    fn dispatch_no_operators_returns_false() {
        let mut d = EventDispatcher::new();
        let mut parts = create_mock_context_parts();
        let mut ctx = make_context(&mut parts);
        assert!(!d.dispatch(&Event::Device(DeviceEvent::Test), &mut ctx));
    }

    // ── Click synthesis ──────────────────────────────────────────────────────

    #[test]
    fn quick_press_release_synthesizes_click() {
        let mut d = EventDispatcher::new();
        let click_count = Rc::new(Cell::new(0u32));

        struct ClickCounter(Rc<Cell<u32>>);
        impl Operator for ClickCounter {
            fn dispatch(&mut self, event: &Event, _: &mut EventContext) -> bool {
                if matches!(event, Event::Device(DeviceEvent::MouseClick { .. })) {
                    self.0.set(self.0.get() + 1);
                }
                false
            }
            fn name(&self) -> &str { "ClickCounter" }
        }

        d.push_back(arc_op(ClickCounter(click_count.clone())));

        let mut parts = create_mock_context_parts();

        // Move cursor so position is known
        let cursor = Event::Device(DeviceEvent::CursorMoved { position: (100.0, 100.0) });
        d.dispatch(&cursor, &mut make_context(&mut parts));

        // Press
        let press = Event::Device(DeviceEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left });
        d.dispatch(&press, &mut make_context(&mut parts));

        // Release immediately (no motion → click)
        let release = Event::Device(DeviceEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left });
        d.dispatch(&release, &mut make_context(&mut parts));

        assert_eq!(click_count.get(), 1);
    }

    #[test]
    fn emitted_events_are_redispatched_to_operators() {
        // Emits a Custom app event the first time it sees the Test device event.
        struct EmitOp {
            emitted: Rc<Cell<bool>>,
        }
        impl Operator for EmitOp {
            fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
                if matches!(event, Event::Device(DeviceEvent::Test)) && !self.emitted.get() {
                    self.emitted.set(true);
                    ctx.emit(AppEvent::Custom(Box::new(7u32)));
                }
                false
            }
            fn name(&self) -> &str {
                "EmitOp"
            }
        }
        // Records the payload of any Custom app event it receives.
        struct ObserveOp {
            seen: Rc<Cell<Option<u32>>>,
        }
        impl Operator for ObserveOp {
            fn dispatch(&mut self, event: &Event, _: &mut EventContext) -> bool {
                if let Event::App(AppEvent::Custom(payload)) = event
                    && let Some(v) = payload.downcast_ref::<u32>()
                {
                    self.seen.set(Some(*v));
                }
                false
            }
            fn name(&self) -> &str {
                "ObserveOp"
            }
        }

        let emitted = Rc::new(Cell::new(false));
        let seen = Rc::new(Cell::new(None));
        let mut d = EventDispatcher::new();
        d.push_back(arc_op(EmitOp { emitted: emitted.clone() }));
        d.push_back(arc_op(ObserveOp { seen: seen.clone() }));

        let mut parts = create_mock_context_parts();
        d.dispatch(&Event::Device(DeviceEvent::Test), &mut make_context(&mut parts));

        assert_eq!(seen.get(), Some(7));
    }

    #[test]
    fn emitted_events_cascade() {
        // On Test, emits Custom(3); in response to Custom(n) emits Custom(n-1) until 0.
        struct CascadeOp {
            count: Rc<Cell<u32>>,
        }
        impl Operator for CascadeOp {
            fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
                match event {
                    Event::Device(DeviceEvent::Test) => ctx.emit(AppEvent::Custom(Box::new(3u32))),
                    Event::App(AppEvent::Custom(payload)) => {
                        if let Some(&n) = payload.downcast_ref::<u32>() {
                            self.count.set(self.count.get() + 1);
                            if n > 0 {
                                ctx.emit(AppEvent::Custom(Box::new(n - 1)));
                            }
                        }
                    }
                    _ => {}
                }
                false
            }
            fn name(&self) -> &str {
                "CascadeOp"
            }
        }

        let count = Rc::new(Cell::new(0u32));
        let mut d = EventDispatcher::new();
        d.push_back(arc_op(CascadeOp { count: count.clone() }));

        let mut parts = create_mock_context_parts();
        d.dispatch(&Event::Device(DeviceEvent::Test), &mut make_context(&mut parts));

        // Custom(3) → (2) → (1) → (0): four deliveries.
        assert_eq!(count.get(), 4);
    }
}
