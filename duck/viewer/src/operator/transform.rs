//! Single-mode transform operator (translate, rotate, or scale) for scene nodes.
//!
//! Each operator instance is locked to one [`TransformMode`] chosen at
//! construction; its gizmo handle reflects that mode. Separate transform
//! operations are separate operators.
//!
//! The interactive core — input bindings, axis-constraint cycling, and the
//! mouse-delta → transform math — lives in [`TransformInteraction`]; this
//! operator supplies the node-specific parts: pivot from the selected nodes,
//! live preview by mutating node transforms, restore-on-cancel, and the
//! [`AppEvent::TransformCommitted`] notification.

mod interaction;

pub use interaction::{
    axis_from_constraint, AxisConstraint, TransformAction, TransformInteraction, TransformMode,
};

use duck_engine_common::{
    EuclideanSpace, Matrix4, Point3, Quaternion, SquareMatrix, Vector3,
};

use crate::common::{
    apply_scale, centroid_of_slice, compose_rotation, decompose_matrix,
    rotate_position_about_pivot, scale_position_about_pivot_local, scale_position_about_pivot_world,
    Transform,
};

use crate::event::{AppEvent, DeviceEvent, Event, EventContext};
use crate::gizmo::GizmoState;
use crate::input::{ElementState, Modifiers};
use crate::operator::Operator;
use crate::scene::{NodeId, Scene};
use crate::scene_scale;

/// Original transform state for a node (used for cancel/restore).
#[derive(Debug, Clone)]
struct OriginalTransform {
    node_id: NodeId,
    local_transform: Transform,
    world_transform: Transform,
    parent_world_transform: Transform,
}

/// Operator for a single transform operation (translate, rotate, or scale).
pub struct TransformOperator {
    /// The interactive drag/constraint state machine.
    interaction: TransformInteraction,

    /// Original transforms of selected nodes (for cancel/restore).
    original_transforms: Vec<OriginalTransform>,

    /// Gizmo handle state (3D visual handles for this operator's mode).
    gizmo: GizmoState,

    /// Whether the persistent gizmo should be shown (when a selection exists).
    /// The handle set is always [`TransformMode::gizmo_type`] for this mode.
    gizmo_enabled: bool,
}

impl TransformOperator {
    /// Creates a new transform operator locked to the given mode.
    pub fn new(mode: TransformMode) -> Self {
        Self {
            interaction: TransformInteraction::new(mode),
            original_transforms: Vec::new(),
            gizmo: GizmoState::new(),
            gizmo_enabled: false,
        }
    }

    /// Returns true if a transform operation is currently active.
    pub fn is_active(&self) -> bool {
        self.interaction.is_active()
    }

    /// Show or hide the persistent gizmo handles. When enabled, the handles for
    /// this operator's mode appear whenever there is a selection.
    pub fn set_gizmo_enabled(&mut self, on: bool) {
        self.gizmo_enabled = on;
    }

    /// Tears down all scene-side visuals owned by this operator (gizmo handles)
    /// and aborts any in-progress transform, restoring the affected nodes to
    /// their pre-transform state.
    pub fn teardown(&mut self, scene: &mut Scene) {
        if self.is_active() {
            for original in &self.original_transforms {
                scene.set_node_transform(original.node_id, original.local_transform);
            }
        }
        self.gizmo.hide(scene);
        self.gizmo_enabled = false;
        self.reset();
    }

    /// Handle an axis key (X/Y/Z): start the transform constrained to that axis
    /// if idle, otherwise cycle the existing constraint. Returns whether the key
    /// was acted on (false only when starting failed, e.g. empty selection).
    fn constrain_axis(&mut self, axis: char, ctx: &mut EventContext) -> bool {
        if !self.is_active() {
            self.start_transform(ctx);
            if !self.is_active() {
                return false;
            }
        }
        self.interaction.cycle_axis_constraint(axis);
        self.after_constraint_changed(ctx);
        true
    }

    /// Handle a plane key (Shift+X/Y/Z): start the transform constrained to that
    /// plane if idle, otherwise cycle the existing plane constraint. Returns
    /// whether the key was acted on (false only when starting failed).
    fn constrain_plane(&mut self, axis: char, ctx: &mut EventContext) -> bool {
        if !self.is_active() {
            self.start_transform(ctx);
            if !self.is_active() {
                return false;
            }
        }
        self.interaction.cycle_plane_constraint(axis);
        self.after_constraint_changed(ctx);
        true
    }

    /// Re-apply the preview and refresh the gizmo highlight after a keyboard
    /// constraint change.
    fn after_constraint_changed(&mut self, ctx: &mut EventContext) {
        self.apply_preview_transform(ctx);
        let highlight = self.interaction.highlight_handle();
        let mut scene = ctx.scene.lock().unwrap();
        self.gizmo.set_highlight(highlight, &mut scene);
    }

    /// Apply the current transform preview to all selected nodes.
    fn apply_preview_transform(&self, ctx: &mut EventContext) {
        if !self.is_active() {
            return;
        }
        let mode = self.interaction.mode();

        // Pre-compute camera-dependent values before locking the scene.
        // ctx.camera() acquires and releases the scene lock internally.
        let camera = ctx.camera();
        let translation_delta = if mode == TransformMode::Translate {
            Some(self.interaction.translation(&camera, ctx.size))
        } else {
            None
        };

        let rotation_quat = if mode == TransformMode::Rotate {
            Some(self.interaction.rotation(&camera))
        } else {
            None
        };

        let scale_factor = if mode == TransformMode::Scale {
            Some(self.interaction.scale(&camera, ctx.size))
        } else {
            None
        };

        let pivot_world = self.interaction.pivot();
        let frame_rotation = self.interaction.frame_rotation();

        // Now apply transforms to nodes under a single scene lock.
        let mut scene = ctx.scene.lock().unwrap();
        for orig in &self.original_transforms {
            let inv_parent = orig
                .parent_world_transform
                .to_matrix()
                .invert()
                .unwrap_or(Matrix4::identity());

            if !scene.has_node(orig.node_id) {
                continue;
            }

            match mode {
                TransformMode::Translate => {
                    let delta = translation_delta.unwrap();
                    let new_world_pos = orig.world_transform.position + delta;
                    let new_local_pos =
                        Point3::from_homogeneous(inv_parent * new_world_pos.to_homogeneous());
                    scene.set_node_position(orig.node_id, new_local_pos);
                }
                TransformMode::Rotate => {
                    let rotation = rotation_quat.unwrap();

                    // Rotate world position around world pivot, convert to local
                    let new_world_pos = rotate_position_about_pivot(
                        orig.world_transform.position,
                        pivot_world,
                        rotation,
                    );
                    let new_local_pos =
                        Point3::from_homogeneous(inv_parent * new_world_pos.to_homogeneous());
                    scene.set_node_position(orig.node_id, new_local_pos);

                    // Convert world rotation to local space
                    let pr = orig.parent_world_transform.rotation;
                    let pr_inv = pr.conjugate();
                    let local_rotation = pr_inv * rotation * pr;
                    let new_rotation =
                        compose_rotation(orig.local_transform.rotation, local_rotation);
                    scene.set_node_rotation(orig.node_id, new_rotation);
                }
                TransformMode::Scale => {
                    let scale = scale_factor.unwrap();

                    if self.interaction.axis_constraint().is_local() {
                        // Local axis: scale in local space, but use world positions for pivot
                        let new_world_pos = scale_position_about_pivot_local(
                            orig.world_transform.position,
                            pivot_world,
                            scale,
                            frame_rotation,
                        );
                        let new_local_pos =
                            Point3::from_homogeneous(inv_parent * new_world_pos.to_homogeneous());
                        scene.set_node_position(orig.node_id, new_local_pos);
                        let new_scale = apply_scale(orig.local_transform.scale, scale);
                        scene.set_node_scale(orig.node_id, new_scale);
                    } else {
                        // World axis: scale world position around pivot, convert to local
                        let new_world_pos = scale_position_about_pivot_world(
                            orig.world_transform.position,
                            pivot_world,
                            scale,
                        );
                        let new_local_pos =
                            Point3::from_homogeneous(inv_parent * new_world_pos.to_homogeneous());
                        scene.set_node_position(orig.node_id, new_local_pos);

                        // Convert world-axis scale to local space
                        let pr_inv = orig.parent_world_transform.rotation.conjugate();
                        let local_scale = world_scale_to_local(scale, pr_inv);
                        let new_scale = apply_scale(orig.local_transform.scale, local_scale);
                        scene.set_node_scale(orig.node_id, new_scale);
                    }
                }
            }
        }
    }

    /// Restore all nodes to their original transforms.
    fn restore_original_transforms(&self, ctx: &mut EventContext) {
        let mut scene = ctx.scene.lock().unwrap();
        for orig in &self.original_transforms {
            if scene.has_node(orig.node_id) {
                scene.set_node_transform(orig.node_id, orig.local_transform);
            }
        }
    }

    /// Show, reposition, or hide the gizmo based on `gizmo_enabled` and selection.
    fn sync_gizmo(&mut self, ctx: &mut EventContext) {
        let selected = ctx.selection.selected_nodes();

        // Hide the gizmo whenever it is disabled or there is nothing selected.
        if !self.gizmo_enabled || selected.is_empty() {
            let mut scene = ctx.scene.lock().unwrap();
            self.gizmo.hide(&mut scene);
            return;
        }

        let gizmo_type = self.interaction.mode().gizmo_type();
        let mut scene = ctx.scene.lock().unwrap();
        let positions: Vec<Point3> = selected
            .iter()
            .filter_map(|&nid| scene.nodes_bounding(nid).bounds.map(|aabb| aabb.center()))
            .collect();
        let pivot = centroid_of_slice(&positions).unwrap_or(Point3::origin());

        if self.gizmo.current_type() == Some(gizmo_type) {
            self.gizmo.update_position(pivot, &mut scene);
        } else {
            self.gizmo.show(gizmo_type, pivot, &mut scene);
        }
    }

    /// Reset state after transform completes (confirm or cancel).
    fn reset(&mut self) {
        self.interaction.finish();
        self.original_transforms.clear();
    }

    /// Start a transform operation in this operator's mode.
    fn start_transform(&mut self, ctx: &mut EventContext) {
        // Get selected nodes
        let selected_nodes = ctx.selection.selected_nodes();
        if selected_nodes.is_empty() {
            return;
        }

        // Store original transforms with world-space info
        let mut world_positions: Vec<Point3> = Vec::new();
        let mut frame_rotation: Option<Quaternion> = None;
        let model_radius;
        {
            let scene = ctx.scene.lock().unwrap();
            for node_id in &selected_nodes {
                if let Some(node) = scene.get_node(*node_id) {
                    let Some(world_matrix) = scene.nodes_transform(*node_id) else { continue };
                    let world_transform = decompose_matrix(&world_matrix);

                    let parent_world_transform = if let Some(parent_id) = node.parent() {
                        scene.nodes_transform(parent_id)
                            .map(|m| decompose_matrix(&m))
                            .unwrap_or(Transform::IDENTITY)
                    } else {
                        Transform::IDENTITY
                    };

                    let world_pos = scene.nodes_bounding(*node_id).bounds
                        .map(|aabb| aabb.center())
                        .unwrap_or(world_transform.position);
                    world_positions.push(world_pos);
                    self.original_transforms.push(OriginalTransform {
                        node_id: *node_id,
                        local_transform: node.transform(),
                        world_transform,
                        parent_world_transform,
                    });
                }
            }

            if world_positions.is_empty() {
                return;
            }

            // The primary selection's rotation orients local axis constraints
            if let Some(primary) = ctx.selection.primary()
                && let Some(node) = scene.get_node(primary.node_id()) {
                    frame_rotation = Some(node.rotation());
                }

            // Get model radius for sensitivity scaling
            model_radius =
                scene_scale::model_radius_from_bounds(scene.bounding().bounds.as_ref());
        }

        // Activate about the centroid of world-space positions
        let pivot = centroid_of_slice(&world_positions).unwrap_or(Point3::origin());
        self.interaction.start(pivot, frame_rotation, model_radius);
    }

    /// Confirm the transform (keep current state).
    ///
    /// The gizmo remains visible at the new position so the user can
    /// start another drag-based transform immediately.
    fn confirm_transform(&mut self, ctx: &mut EventContext) {
        // Notify downstream consumers of the confirmed nodes
        let nodes: Vec<NodeId> = self.original_transforms.iter().map(|o| o.node_id).collect();
        ctx.emit(AppEvent::TransformCommitted { nodes });
        {
            let mut scene = ctx.scene.lock().unwrap();
            self.gizmo.set_highlight(None, &mut scene);
        }
        self.reset();
        self.sync_gizmo(ctx);
    }

    /// Cancel the transform (restore original state).
    ///
    /// The gizmo remains visible at the original position.
    fn cancel_transform(&mut self, ctx: &mut EventContext) {
        self.restore_original_transforms(ctx);
        {
            let mut scene = ctx.scene.lock().unwrap();
            self.gizmo.set_highlight(None, &mut scene);
        }
        self.reset();
        self.sync_gizmo(ctx);
    }
}

/// Converts a world-axis-constrained scale into the parent's local space.
///
/// For uniform scale (no constraint), returns as-is. For single-axis world
/// constraints, rotates the scale axis into local space and decomposes it
/// into per-axis scale contributions.
fn world_scale_to_local(scale: Vector3, parent_rotation_inv: Quaternion) -> Vector3 {
    use duck_engine_common::{InnerSpace, Rotation};

    // For uniform scale (no constraint), return as-is
    if (scale.x - scale.y).abs() < 1e-6 && (scale.y - scale.z).abs() < 1e-6 {
        return scale;
    }
    // Find the world axis being scaled and its factor
    let (world_axis, factor) = if (scale.x - 1.0).abs() > 1e-6 {
        (Vector3::unit_x(), scale.x)
    } else if (scale.y - 1.0).abs() > 1e-6 {
        (Vector3::unit_y(), scale.y)
    } else {
        (Vector3::unit_z(), scale.z)
    };
    // Rotate to local space
    let local_axis = parent_rotation_inv.rotate_vector(world_axis).normalize();
    // Decompose into per-axis scale contributions
    let factor_minus_1 = factor - 1.0;
    Vector3::new(
        1.0 + local_axis.x.abs() * factor_minus_1,
        1.0 + local_axis.y.abs() * factor_minus_1,
        1.0 + local_axis.z.abs() * factor_minus_1,
    )
}

impl Operator for TransformOperator {
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
                    self.apply_preview_transform(ctx);
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
                    self.interaction.constrain_to_handle(handle, *start_pos);
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
                    self.apply_preview_transform(ctx);
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
                // Hover highlight on gizmo handles when gizmo is visible but no transform active
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
