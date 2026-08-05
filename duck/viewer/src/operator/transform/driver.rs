//! The shared interactive core of every transform-style operator.
//!
//! [`TransformDriver`] owns the pieces common to all transform operations —
//! the [`TransformInteraction`] state machine, the gizmo visuals, and the full
//! input event loop — and delegates everything target-specific (what is being
//! transformed, how it is previewed, and what commit means) to a
//! [`TransformTarget`] implementation.


use duck_engine_common::{Point3, Quaternion};

use super::gizmo::{GizmoHandleId, GizmoState};
use super::interaction::{TransformAction, TransformInteraction, TransformMode};
use crate::event::{DeviceEvent, Event, EventContext};
use crate::input::{ElementState, Modifiers};
use crate::operator::Operator;
use crate::scene::Scene;

/// World-space anchor of a transform target: where the gizmo sits and what
/// orients local-axis constraints.
pub struct TransformFrame {
    pub pivot: Point3,
    /// Orients local-axis constraints; `None` keeps the interaction's
    /// previous frame.
    pub frame_rotation: Option<Quaternion>,
}

/// What interactions a target supports.
///
/// Only queried while a transform is active (after a successful
/// [`TransformTarget::begin`]), so targets that route between sub-targets may
/// answer for the locked one.
#[derive(Debug, Clone, Copy, Default)]
pub struct TransformCaps {
    /// Scale must stay uniform: axis/plane keys are accepted but don't
    /// constrain in Scale mode, and a grabbed scale handle behaves like the
    /// center ball.
    pub uniform_scale_only: bool,
}

/// The target-specific half of a transform operator: what is transformed, how
/// the drag is previewed, and what confirm/cancel do.
///
/// Locking discipline for all methods: the driver never holds the scene lock
/// across a call, and implementations must precompute camera-dependent values
/// via [`EventContext::camera`] *before* locking the scene themselves —
/// `camera()` acquires the scene lock internally.
pub trait TransformTarget: 'static {
    /// Pivot/frame for what would be transformed right now. Drives idle gizmo
    /// placement and seeds the interaction on start; `None` hides the gizmo
    /// and refuses starts. Never called while a transform is active.
    fn frame(&mut self, ctx: &mut EventContext) -> Option<TransformFrame>;

    /// See [`TransformCaps`]. Only queried while a transform is active.
    fn caps(&self) -> TransformCaps {
        TransformCaps::default()
    }

    /// Lock onto the current target and snapshot whatever `preview`/`cancel`
    /// need. Returning `false` refuses the start.
    fn begin(&mut self, ctx: &mut EventContext) -> bool;

    /// Show the interaction's current delta on the target.
    fn preview(&mut self, interaction: &TransformInteraction, ctx: &mut EventContext);

    /// Apply the final delta and release the target begun by `begin`. Called
    /// while the interaction is still active so the final delta is readable.
    fn commit(&mut self, interaction: &TransformInteraction, ctx: &mut EventContext);

    /// Revert the preview and release the target begun by `begin`.
    fn cancel(&mut self, ctx: &mut EventContext);

    /// Abort outside dispatch (e.g. tool switch): revert any in-progress
    /// preview and drop target state. Must be idempotent. Receives the scene
    /// handle, not a guard, because implementations lock scene and/or other
    /// state in their own order.
    fn abort(&mut self, scene: &Scene);
}

/// Operator driving a single transform operation (translate, rotate, or
/// scale) on a [`TransformTarget`].
///
/// The driver handles input bindings, gizmo display/picking/highlighting,
/// axis-constraint cycling, and confirm/cancel flow; the target supplies the
/// pivot and the preview/commit behavior.
pub struct TransformDriver<T: TransformTarget> {
    /// The interactive drag/constraint state machine.
    interaction: TransformInteraction,

    /// Gizmo handle state (3D visual handles for this operator's mode).
    gizmo: GizmoState,

    /// Whether the persistent gizmo should be shown (when the target reports
    /// a frame). The handle set is always [`TransformMode::gizmo_type`] for
    /// this driver's mode.
    gizmo_enabled: bool,

    target: T,
}

impl<T: TransformTarget> TransformDriver<T> {
    /// Creates a driver locked to the given mode, transforming `target`.
    pub fn with_target(mode: TransformMode, target: T) -> Self {
        Self {
            interaction: TransformInteraction::new(mode),
            gizmo: GizmoState::new(),
            gizmo_enabled: false,
            target,
        }
    }

    /// Returns true if a transform operation is currently active.
    pub fn is_active(&self) -> bool {
        self.interaction.is_active()
    }

    /// The target this driver transforms.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// Show or hide the persistent gizmo handles. When enabled, the handles
    /// for this driver's mode appear whenever the target reports a frame.
    pub fn set_gizmo_enabled(&mut self, on: bool) {
        self.gizmo_enabled = on;
    }

    /// Tears down all scene-side visuals owned by this driver (gizmo handles)
    /// and aborts any in-progress transform, reverting the target to its
    /// pre-transform state.
    pub fn teardown(&mut self, scene: &Scene) {
        self.target.abort(scene);
        self.gizmo.hide(scene);
        self.gizmo_enabled = false;
        self.interaction.finish();
    }

    /// Whether the active transform's constraints must be kept uniform-scale.
    fn uniform_scale_only(&self) -> bool {
        self.target.caps().uniform_scale_only && self.interaction.mode() == TransformMode::Scale
    }

    /// Handle an axis key (X/Y/Z): start the transform constrained to that
    /// axis if idle, otherwise cycle the existing constraint. Returns whether
    /// the key was acted on (false only when starting failed).
    fn constrain_axis(&mut self, axis: char, ctx: &mut EventContext) -> bool {
        if !self.is_active() {
            self.start_transform(*ctx.cursor_position, ctx);
            if !self.is_active() {
                return false;
            }
        }
        if self.uniform_scale_only() {
            return true;
        }
        self.interaction.cycle_axis_constraint(axis);
        self.after_constraint_changed(ctx);
        true
    }

    /// Handle a plane key (Shift+X/Y/Z): start the transform constrained to
    /// that plane if idle, otherwise cycle the existing plane constraint.
    /// Returns whether the key was acted on (false only when starting failed).
    fn constrain_plane(&mut self, axis: char, ctx: &mut EventContext) -> bool {
        if !self.is_active() {
            self.start_transform(*ctx.cursor_position, ctx);
            if !self.is_active() {
                return false;
            }
        }
        if self.uniform_scale_only() {
            return true;
        }
        self.interaction.cycle_plane_constraint(axis);
        self.after_constraint_changed(ctx);
        true
    }

    /// Re-apply the preview and refresh the gizmo highlight after a keyboard
    /// constraint change.
    fn after_constraint_changed(&mut self, ctx: &mut EventContext) {
        // Translation and scale re-solve from the drag anchor, but a swept
        // rotation angle belongs to the axis it was swept about.
        self.interaction.reset_rotation();
        self.apply_preview(ctx);
        let highlight = self.interaction.highlight_handle();
        self.gizmo.set_highlight(highlight, &ctx.scene);
    }

    /// Show the current delta on the target, if a transform is active.
    fn apply_preview(&mut self, ctx: &mut EventContext) {
        if self.is_active() {
            self.target.preview(&self.interaction, ctx);
        }
    }

    /// Show, reposition, or hide the gizmo based on `gizmo_enabled` and the
    /// target's current frame.
    fn sync_gizmo(&mut self, ctx: &mut EventContext) {
        let frame = if self.gizmo_enabled { self.target.frame(ctx) } else { None };

        // Hide the gizmo whenever it is disabled or there is nothing to
        // transform.
        let Some(frame) = frame else {
            self.gizmo.hide(&ctx.scene);
            return;
        };

        let gizmo_type = self.interaction.mode().gizmo_type();
        if self.gizmo.current_type() == Some(gizmo_type) {
            self.gizmo.update_position(frame.pivot, &ctx.scene);
        } else {
            self.gizmo.show(gizmo_type, frame.pivot, &ctx.scene);
        }
    }

    /// Start a transform operation in this driver's mode.
    ///
    /// `anchor` is the pixel the drag is measured from — the grabbed handle for
    /// a gizmo drag, the cursor for a keyboard start. With no cursor to anchor
    /// to, the pivot's own projection stands in.
    fn start_transform(&mut self, anchor: Option<(f32, f32)>, ctx: &mut EventContext) {
        let Some(frame) = self.target.frame(ctx) else { return };
        if !self.target.begin(ctx) {
            return;
        }

        let anchor = anchor.unwrap_or_else(|| {
            let projected = ctx.camera().project_point_screen(frame.pivot, ctx.size.0, ctx.size.1);
            (projected.x, projected.y)
        });
        self.interaction.start(frame.pivot, frame.frame_rotation, anchor);
    }

    /// Confirm the transform: the target applies the final delta.
    ///
    /// The gizmo remains visible at the new position so the user can
    /// start another transform immediately.
    fn confirm_transform(&mut self, ctx: &mut EventContext) {
        // The interaction is still active here so the target can read the
        // final delta.
        self.target.commit(&self.interaction, ctx);
        self.gizmo.set_highlight(None, &ctx.scene);
        self.interaction.finish();
        self.sync_gizmo(ctx);
    }

    /// Cancel the transform: the target reverts to its original state.
    ///
    /// The gizmo remains visible at the original position.
    fn cancel_transform(&mut self, ctx: &mut EventContext) {
        self.target.cancel(ctx);
        self.gizmo.set_highlight(None, &ctx.scene);
        self.interaction.finish();
        self.sync_gizmo(ctx);
    }
}

impl<T: TransformTarget> Operator for TransformDriver<T> {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state != ElementState::Pressed || key_event.repeat {
                    return false;
                }
                let actions = self.interaction.bindings
                    .actions_for_key(&key_event.logical_key, ctx.modifiers)
                    .to_vec();
                for action in actions {
                    match action {
                        TransformAction::StartTransform if !self.is_active() => {
                            self.start_transform(*ctx.cursor_position, ctx);
                            return self.is_active();
                        }
                        TransformAction::ConstrainX => {
                            if self.constrain_axis('x', ctx) {
                                return true;
                            }
                        }
                        TransformAction::ConstrainY => {
                            if self.constrain_axis('y', ctx) {
                                return true;
                            }
                        }
                        TransformAction::ConstrainZ => {
                            if self.constrain_axis('z', ctx) {
                                return true;
                            }
                        }
                        TransformAction::ConstrainPlaneX => {
                            if self.constrain_plane('x', ctx) {
                                return true;
                            }
                        }
                        TransformAction::ConstrainPlaneY => {
                            if self.constrain_plane('y', ctx) {
                                return true;
                            }
                        }
                        TransformAction::ConstrainPlaneZ => {
                            if self.constrain_plane('z', ctx) {
                                return true;
                            }
                        }
                        TransformAction::KeyConfirm if self.is_active() => {
                            self.confirm_transform(ctx);
                            return true;
                        }
                        TransformAction::KeyCancel if self.is_active() => {
                            self.cancel_transform(ctx);
                            return true;
                        }
                        _ => {}
                    }
                }
                false
            }

            DeviceEvent::MouseMotion { delta } => {
                if self.is_active() {
                    let camera = ctx.camera();
                    self.interaction.accumulate(
                        delta.0 as f32, delta.1 as f32, &camera, ctx.size,
                    );
                    self.apply_preview(ctx);
                    true
                } else {
                    false
                }
            }

            DeviceEvent::MouseClick { button, .. } => {
                if !self.is_active() {
                    return false;
                }
                let actions =
                    self.interaction.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                for action in actions {
                    match action {
                        TransformAction::MouseConfirm => {
                            self.confirm_transform(ctx);
                            return true;
                        }
                        TransformAction::MouseCancel => {
                            self.cancel_transform(ctx);
                            return true;
                        }
                        _ => {}
                    }
                }
                false
            }

            DeviceEvent::MouseDragStart { button, start_pos, .. } => {
                if !self.interaction.bindings
                    .actions_for_drag_start(*button, ctx.modifiers)
                    .contains(&TransformAction::GizmoDrag)
                {
                    return false;
                }
                if self.is_active() || !self.gizmo.has_gizmo() {
                    return false;
                }
                let picked = {
                    let camera = ctx.camera();
                    let ray = camera.ray_from_screen_point(
                        start_pos.0, start_pos.1, ctx.size.0, ctx.size.1,
                    );
                    self.gizmo.pick_handle(ray, &ctx.scene, &camera, ctx.size)
                };
                if let Some(handle) = picked {
                    // Anchor at the pixel the handle was picked at, not the
                    // dispatcher's current cursor: `MouseDragStart` is
                    // synthesized from inside `process_mouse_motion`, so the
                    // current cursor already includes the motion that is about
                    // to be accumulated from the original `MouseMotion`.
                    self.start_transform(Some(*start_pos), ctx);
                    if !self.is_active() {
                        return false;
                    }
                    // Uniform-only scale: treat any handle like the center
                    // ball so the grab only begins a directional drag. Other
                    // modes adopt the handle's axis/plane constraint.
                    let constraint_handle = if self.uniform_scale_only() {
                        GizmoHandleId::Ball
                    } else {
                        handle
                    };
                    self.interaction.constrain_to_handle(constraint_handle);
                    self.gizmo.set_highlight(Some(handle), &ctx.scene);
                    return true;
                }
                false
            }

            // Consumed but deliberately not accumulated: the dispatcher
            // synthesizes `MouseDrag` from a `MouseMotion` and dispatches it
            // first, then dispatches the original, so accumulating both would
            // double-count every drag.
            DeviceEvent::MouseDrag { button, .. } => {
                if !self.interaction.bindings
                    .actions_for_drag(*button, ctx.modifiers)
                    .contains(&TransformAction::GizmoDrag)
                {
                    return false;
                }
                self.is_active()
            }

            DeviceEvent::MouseDragEnd { button, .. } => {
                if !self.interaction.bindings
                    .actions_for_drag_end(*button, Modifiers::default())
                    .contains(&TransformAction::GizmoDrag)
                {
                    return false;
                }
                if self.is_active() {
                    self.confirm_transform(ctx);
                    return true;
                }
                false
            }

            DeviceEvent::CursorMoved { position } => {
                // While dragging, the absolute cursor is what the grabbed
                // geometry must line up with — `MouseMotion` reports raw device
                // movement, which pointer acceleration makes a different
                // quantity from on-screen pixels.
                if self.is_active() {
                    let camera = ctx.camera();
                    self.interaction.set_cursor(
                        (position.0 as f32, position.1 as f32), &camera, ctx.size,
                    );
                    self.apply_preview(ctx);
                    return true;
                }

                // Hover highlight on gizmo handles when the gizmo is visible
                // but no transform is active.
                if self.gizmo.has_gizmo() {
                    let camera = ctx.camera();
                    let ray = camera.ray_from_screen_point(
                        position.0 as f32, position.1 as f32, ctx.size.0, ctx.size.1,
                    );
                    let handle = self.gizmo.pick_handle(ray, &ctx.scene, &camera, ctx.size);
                    self.gizmo.set_highlight(handle, &ctx.scene);
                }
                false
            }

            DeviceEvent::Update { .. } => {
                if self.gizmo_enabled && !self.is_active() {
                    self.sync_gizmo(ctx);
                }
                false
            }

            _ => false,
        }
    }

    fn name(&self) -> &str {
        match self.interaction.mode() {
            TransformMode::Translate => "Translate",
            TransformMode::Rotate => "Rotate",
            TransformMode::Scale => "Scale",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::AppEvent;
    use crate::input::MouseButton;
    use super::super::interaction::{AxisConstraint, ConstraintSpace};
    use crate::common::Axis;
    use crate::scene::{NodePayload, PositionedCamera, Scene, SceneData};
    use crate::selection::SelectionManager;
    use duck_engine_common::{InnerSpace, Vector3};
    use duck_engine_scene::NodeFlags;
    use std::cell::RefCell;
    use std::rc::Rc;

    type ContextParts = (Option<(f32, f32)>, Scene, SelectionManager);

    fn context_parts() -> ContextParts {
        context_parts_viewed_from((0.0, 0.0, 4.0))
    }

    fn context_parts_viewed_from(eye: (f32, f32, f32)) -> ContextParts {
        let camera = PositionedCamera {
            eye: eye.into(),
            target: (0.0, 0.0, 0.0).into(),
            up: Vector3::unit_y(),
            aspect: 800.0 / 600.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        };
        let mut scene = SceneData::new();
        let cam_id = scene
            .add_node(None, None, camera.to_node_transform(), NodeFlags::NONE)
            .unwrap()
            .id();
        scene.set_node_payload(cam_id, NodePayload::Camera(camera.projection()));
        scene.set_active_camera(Some(cam_id));
        (Some((400.0, 300.0)), Scene::new(scene), SelectionManager::new())
    }

    fn make_context(parts: &mut ContextParts) -> EventContext<'_> {
        EventContext {
            size: (800, 600),
            cursor_position: &mut parts.0,
            scene: parts.1.clone(),
            selection: &mut parts.2,
            modifiers: Default::default(),
            emit_queue: Vec::new(),
        }
    }

    /// Records every translation the driver previews, so a test can count how
    /// many times a given event stream moved the target.
    struct RecordingTarget {
        previews: Rc<RefCell<Vec<Vector3>>>,
    }

    impl TransformTarget for RecordingTarget {
        fn frame(&mut self, _ctx: &mut EventContext) -> Option<TransformFrame> {
            Some(TransformFrame { pivot: Point3::new(0.0, 0.0, 0.0), frame_rotation: None })
        }
        fn begin(&mut self, _ctx: &mut EventContext) -> bool {
            true
        }
        fn preview(&mut self, interaction: &TransformInteraction, ctx: &mut EventContext) {
            let camera = ctx.camera();
            self.previews.borrow_mut().push(interaction.translation(&camera, ctx.size));
        }
        fn commit(&mut self, _interaction: &TransformInteraction, _ctx: &mut EventContext) {}
        fn cancel(&mut self, _ctx: &mut EventContext) {}
        fn abort(&mut self, _scene: &Scene) {}
    }

    fn motion(delta: (f64, f64)) -> Event {
        Event::Device(DeviceEvent::MouseMotion { delta })
    }

    fn cursor_moved(position: (f32, f32)) -> Event {
        Event::Device(DeviceEvent::CursorMoved {
            position: (position.0 as f64, position.1 as f64),
        })
    }

    fn drag(delta: (f32, f32)) -> Event {
        Event::Device(DeviceEvent::MouseDrag {
            button: MouseButton::Left,
            start_pos: (400.0, 300.0),
            current_pos: (400.0 + delta.0, 300.0 + delta.1),
            delta,
        })
    }

    #[test]
    fn mouse_drag_is_swallowed_without_accumulating() {
        // The dispatcher synthesizes `MouseDrag` from a `MouseMotion` and
        // dispatches both, so accumulating each would double-count the drag. The
        // arm must still consume: `NavigationOperator` binds left-button
        // `MouseDrag` to orbit and sits behind the transform drivers.
        let previews = Rc::new(RefCell::new(Vec::new()));
        let mut driver = TransformDriver::with_target(
            TransformMode::Translate,
            RecordingTarget { previews: Rc::clone(&previews) },
        );

        let mut parts = context_parts();
        {
            let mut ctx = make_context(&mut parts);
            driver.start_transform(Some((400.0, 300.0)), &mut ctx);
        }
        assert!(driver.is_active());

        let mut ctx = make_context(&mut parts);
        assert!(driver.dispatch(&motion((30.0, 0.0)), &mut ctx));
        assert_eq!(previews.borrow().len(), 1, "motion should preview once");
        let after_motion = previews.borrow()[0];
        assert!(after_motion.magnitude() > 0.0, "motion should translate");

        assert!(
            driver.dispatch(&drag((30.0, 0.0)), &mut ctx),
            "MouseDrag must be consumed so the camera does not orbit mid-drag"
        );
        assert_eq!(
            previews.borrow().len(),
            1,
            "MouseDrag must not accumulate on top of the MouseMotion it was synthesized from"
        );
    }

    #[test]
    fn mouse_drag_is_ignored_when_no_transform_is_active() {
        // Idle drivers must let drags through, or navigation could never orbit.
        let previews = Rc::new(RefCell::new(Vec::new()));
        let mut driver = TransformDriver::with_target(
            TransformMode::Translate,
            RecordingTarget { previews: Rc::clone(&previews) },
        );

        let mut parts = context_parts();
        let mut ctx = make_context(&mut parts);
        assert!(!driver.dispatch(&drag((10.0, 5.0)), &mut ctx));
        assert!(!driver.dispatch(&motion((10.0, 5.0)), &mut ctx));
        assert!(previews.borrow().is_empty());
    }

    #[test]
    fn keyboard_start_anchors_at_the_cursor() {
        // A keyboard-started transform has no grabbed handle, so the cursor is
        // the anchor. With the cursor exactly on the pivot's projection the two
        // agree, so compare against an explicit off-pivot cursor instead.
        let previews = Rc::new(RefCell::new(Vec::new()));
        let mut driver = TransformDriver::with_target(
            TransformMode::Scale,
            RecordingTarget { previews: Rc::clone(&previews) },
        );

        let mut parts = context_parts();
        parts.0 = Some((520.0, 300.0));
        {
            let mut ctx = make_context(&mut parts);
            let anchor = *ctx.cursor_position;
            driver.start_transform(anchor, &mut ctx);
        }
        assert!(driver.is_active());

        // Anchored 120 px right of the pivot and dragged another 120 px out, the
        // grabbed point is twice as far from the pivot: a 2x scale.
        let mut ctx = make_context(&mut parts);
        assert!(driver.dispatch(&motion((120.0, 0.0)), &mut ctx));
        let camera = ctx.camera();
        let scale = driver.interaction.scale(&camera, ctx.size);
        assert!((scale.x - 2.0).abs() < 1e-3, "{scale:?}");
    }

    #[test]
    fn ground_plane_drag_follows_the_cursor_not_the_device_delta() {
        // End-to-end regression for the reported symptom: a sphere dragged on the
        // XZ plane tracked the right direction but fell short of the cursor, and
        // further short the further it was dragged. Raw `MouseMotion` deltas are
        // unaccelerated, so they cannot place the grabbed point under the pointer;
        // the absolute `CursorMoved` position has to win.
        let previews = Rc::new(RefCell::new(Vec::new()));
        let mut driver = TransformDriver::with_target(
            TransformMode::Translate,
            RecordingTarget { previews: Rc::clone(&previews) },
        );

        // Camera above the ground plane, looking down at the pivot.
        let mut parts = context_parts_viewed_from((0.0, 4.0, 9.0));
        let pivot = Point3::new(0.0, 0.0, 0.0);
        let anchor = {
            let mut ctx = make_context(&mut parts);
            let projected = ctx.camera().project_point_screen(pivot, ctx.size.0, ctx.size.1);
            let anchor = (projected.x, projected.y);
            driver.start_transform(Some(anchor), &mut ctx);
            driver.interaction.set_axis_constraint(AxisConstraint::Plane(
                Axis::Y,
                ConstraintSpace::World,
            ));
            anchor
        };
        assert!(driver.is_active());

        // Walk the pointer up and to the left, with the device under-reporting
        // every step by a quarter — the pointer-acceleration mismatch.
        let mut ctx = make_context(&mut parts);
        let mut cursor = anchor;
        for _ in 0..10 {
            cursor = (cursor.0 - 14.0, cursor.1 - 22.0);
            assert!(driver.dispatch(&cursor_moved(cursor), &mut ctx));
            driver.dispatch(&motion((-10.5, -16.5)), &mut ctx);
        }

        let camera = ctx.camera();
        let translation = *previews.borrow().last().unwrap();
        let landed = camera.project_point_screen(pivot + translation, ctx.size.0, ctx.size.1);
        assert!(
            (landed.x - cursor.0).abs() < 0.05 && (landed.y - cursor.1).abs() < 0.05,
            "grabbed point at {:?} should be under the cursor {cursor:?}",
            (landed.x, landed.y)
        );
        // Stayed on the constraint plane throughout.
        assert!(translation.y.abs() < 1e-5, "{translation:?}");
    }

    #[test]
    fn app_events_are_not_consumed() {
        let previews = Rc::new(RefCell::new(Vec::new()));
        let mut driver = TransformDriver::with_target(
            TransformMode::Translate,
            RecordingTarget { previews },
        );
        let mut parts = context_parts();
        let mut ctx = make_context(&mut parts);
        assert!(!driver.dispatch(&Event::App(AppEvent::CameraInteractionStart), &mut ctx));
    }
}
