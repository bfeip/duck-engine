//! [`TransformTarget`] over the selected scene nodes.


use duck_engine_common::{Matrix4, Point3, Quaternion, SquareMatrix, Vector3};

use super::driver::{TransformFrame, TransformTarget};
use super::interaction::{TransformInteraction, TransformMode};
use crate::common::{
    apply_scale, centroid_of_slice, compose_rotation, decompose_matrix,
    rotate_position_about_pivot, scale_position_about_pivot_local, scale_position_about_pivot_world,
    Transform,
};
use crate::event::EventContext;
use crate::scene::resource::NodeId;
use crate::scene::Scene;

/// Original transform state for a node (used for cancel/restore).
#[derive(Debug, Clone)]
struct OriginalTransform {
    node_id: NodeId,
    local_transform: Transform,
    world_transform: Transform,
    parent_world_transform: Transform,
}

/// Transforms the selected scene nodes: pivot at the centroid of their
/// bounds, live preview by mutating node transforms, restore-on-cancel.
#[derive(Default)]
pub struct NodeTransformTarget {
    /// Original transforms of the selected nodes (for cancel/restore).
    original_transforms: Vec<OriginalTransform>,
}

impl NodeTransformTarget {
    pub fn new() -> Self {
        Self::default()
    }

    /// The nodes captured by the in-progress transform.
    pub fn nodes(&self) -> Vec<NodeId> {
        self.original_transforms.iter().map(|o| o.node_id).collect()
    }

    /// Restore all captured nodes to their original transforms.
    fn restore(&self, scene: &Scene) {
        let mut scene = scene.lock();
        for orig in &self.original_transforms {
            if scene.has_node(orig.node_id) {
                scene.set_node_transform(orig.node_id, orig.local_transform);
            }
        }
    }
}

impl TransformTarget for NodeTransformTarget {
    fn frame(&mut self, ctx: &mut EventContext) -> Option<TransformFrame> {
        let selected = ctx.selection.selected_nodes();
        if selected.is_empty() {
            return None;
        }

        let scene = ctx.scene.lock();
        let positions: Vec<Point3> = selected
            .iter()
            .filter_map(|&nid| {
                scene
                    .nodes_bounding(nid)
                    .bounds
                    .map(|aabb| aabb.center())
                    .or_else(|| {
                        scene.nodes_transform(nid).map(|m| decompose_matrix(&m).position)
                    })
            })
            .collect();
        let pivot = centroid_of_slice(&positions)?;

        // The primary selection's rotation orients local axis constraints.
        let frame_rotation = ctx
            .selection
            .primary()
            .and_then(|primary| scene.get_node(primary.node_id()))
            .map(|node| node.rotation());

        Some(TransformFrame { pivot, frame_rotation })
    }

    fn begin(&mut self, ctx: &mut EventContext) -> bool {
        self.original_transforms.clear();
        let selected_nodes = ctx.selection.selected_nodes();

        let scene = ctx.scene.lock();
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

                self.original_transforms.push(OriginalTransform {
                    node_id: *node_id,
                    local_transform: node.transform(),
                    world_transform,
                    parent_world_transform,
                });
            }
        }

        !self.original_transforms.is_empty()
    }

    fn preview(&mut self, interaction: &TransformInteraction, ctx: &mut EventContext) {
        let mode = interaction.mode();

        let camera = ctx.camera.clone();
        let translation_delta = if mode == TransformMode::Translate {
            Some(interaction.translation(&camera, ctx.size))
        } else {
            None
        };

        let rotation_quat = if mode == TransformMode::Rotate {
            Some(interaction.rotation(&camera))
        } else {
            None
        };

        let scale_factor = if mode == TransformMode::Scale {
            Some(interaction.scale(&camera, ctx.size))
        } else {
            None
        };

        let pivot_world = interaction.pivot();
        let frame_rotation = interaction.frame_rotation();

        let mut scene = ctx.scene.lock();
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

                    if interaction.axis_constraint().is_local() {
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

    fn commit(&mut self, _interaction: &TransformInteraction, _ctx: &mut EventContext) {
        // The previewed node transforms are simply kept.
        self.original_transforms.clear();
    }

    fn cancel(&mut self, ctx: &mut EventContext) {
        self.restore(&ctx.scene);
        self.original_transforms.clear();
    }

    fn abort(&mut self, scene: &Scene) {
        self.restore(scene);
        self.original_transforms.clear();
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
