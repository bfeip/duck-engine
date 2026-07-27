//! The shared interactive core of every transform-style operator.
//!
//! [`TransformDriver`] owns the pieces common to all transform operations —
//! the [`TransformInteraction`] state machine, the gizmo visuals, and the full
//! input event loop — and delegates everything target-specific (what is being
//! transformed, how it is previewed, and what commit means) to a
//! [`TransformTarget`] implementation.

use std::sync::{Arc, Mutex};

use duck_engine_common::{Point3, Quaternion};

use super::gizmo::{GizmoHandleId, GizmoState};
use super::interaction::{TransformAction, TransformInteraction, TransformMode};
use crate::event::{DeviceEvent, Event, EventContext};
use crate::input::{ElementState, Modifiers};
use crate::operator::Operator;
use crate::scene::Scene;
use crate::scene_scale;

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
    fn abort(&mut self, scene: &Arc<Mutex<Scene>>);
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
    pub fn teardown(&mut self, scene: &Arc<Mutex<Scene>>) {
        self.target.abort(scene);
        self.gizmo.hide(&mut scene.lock().unwrap());
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
            self.start_transform(ctx);
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
            self.start_transform(ctx);
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
        self.apply_preview(ctx);
        let highlight = self.interaction.highlight_handle();
        let mut scene = ctx.scene.lock().unwrap();
        self.gizmo.set_highlight(highlight, &mut scene);
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
            let mut scene = ctx.scene.lock().unwrap();
            self.gizmo.hide(&mut scene);
            return;
        };

        let gizmo_type = self.interaction.mode().gizmo_type();
        let mut scene = ctx.scene.lock().unwrap();
        if self.gizmo.current_type() == Some(gizmo_type) {
            self.gizmo.update_position(frame.pivot, &mut scene);
        } else {
            self.gizmo.show(gizmo_type, frame.pivot, &mut scene);
        }
    }

    /// Start a transform operation in this driver's mode.
    fn start_transform(&mut self, ctx: &mut EventContext) {
        let Some(frame) = self.target.frame(ctx) else { return };
        if !self.target.begin(ctx) {
            return;
        }

        // Model radius scales drag sensitivity.
        let model_radius = {
            let scene = ctx.scene.lock().unwrap();
            scene_scale::model_radius_from_bounds(scene.bounding().bounds.as_ref())
        };
        self.interaction.start(frame.pivot, frame.frame_rotation, model_radius);
    }

    /// Confirm the transform: the target applies the final delta.
    ///
    /// The gizmo remains visible at the new position so the user can
    /// start another transform immediately.
    fn confirm_transform(&mut self, ctx: &mut EventContext) {
        // The interaction is still active here so the target can read the
        // final delta.
        self.target.commit(&self.interaction, ctx);
        {
            let mut scene = ctx.scene.lock().unwrap();
            self.gizmo.set_highlight(None, &mut scene);
        }
        self.interaction.finish();
        self.sync_gizmo(ctx);
    }

    /// Cancel the transform: the target reverts to its original state.
    ///
    /// The gizmo remains visible at the original position.
    fn cancel_transform(&mut self, ctx: &mut EventContext) {
        self.target.cancel(ctx);
        {
            let mut scene = ctx.scene.lock().unwrap();
            self.gizmo.set_highlight(None, &mut scene);
        }
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
                            self.start_transform(ctx);
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
                    self.interaction.accumulate(delta.0 as f32, delta.1 as f32);
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
                    let scene = ctx.scene.lock().unwrap();
                    self.gizmo.pick_handle(ray, &scene, &camera, ctx.size)
                };
                if let Some(handle) = picked {
                    self.start_transform(ctx);
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
                    self.interaction.constrain_to_handle(constraint_handle, *start_pos);
                    let mut scene = ctx.scene.lock().unwrap();
                    self.gizmo.set_highlight(Some(handle), &mut scene);
                    return true;
                }
                false
            }

            DeviceEvent::MouseDrag { button, delta, .. } => {
                if !self.interaction.bindings
                    .actions_for_drag(*button, ctx.modifiers)
                    .contains(&TransformAction::GizmoDrag)
                {
                    return false;
                }
                if self.is_active() {
                    self.interaction.accumulate(delta.0, delta.1);
                    self.apply_preview(ctx);
                    return true;
                }
                false
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
                // Hover highlight on gizmo handles when the gizmo is visible
                // but no transform is active.
                if self.gizmo.has_gizmo() && !self.is_active() {
                    let camera = ctx.camera();
                    let ray = camera.ray_from_screen_point(
                        position.0 as f32, position.1 as f32, ctx.size.0, ctx.size.1,
                    );
                    let mut scene = ctx.scene.lock().unwrap();
                    let handle = self.gizmo.pick_handle(ray, &scene, &camera, ctx.size);
                    self.gizmo.set_highlight(handle, &mut scene);
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
