//! Constraint-axis annotation lines shown during an interactive transform.

use std::collections::HashMap;

use duck_engine_scene::{Mesh, NodeFlags, Scene};

use duck_engine_common::{InnerSpace, Vector3};

use crate::common::{Axis, Transform};
use crate::scene::{Instance, LineMaterial, LineMaterialId, NodeId};

use super::interaction::{AxisConstraint, TransformInteraction};

/// Owns the scene nodes for transform feedback lines: a colored line through
/// the pivot along the constrained axis, redrawn as the constraint changes.
pub struct TransformAnnotations {
    /// Root node for the annotation lines, created when needed.
    root: Option<NodeId>,

    /// Live annotation nodes, cleaned up after the transform completes.
    nodes: Vec<NodeId>,

    /// Materials for the colored annotations (so we're not making dozens of copies)
    axis_materials: HashMap<Axis, LineMaterialId>,
}

impl TransformAnnotations {
    pub fn new() -> Self {
        Self { root: None, nodes: Vec::new(), axis_materials: HashMap::new() }
    }

    /// Redraw the constraint-axis line for the interaction's current state
    /// (clears it when unconstrained).
    pub fn update(&mut self, interaction: &TransformInteraction, scene: &mut Scene) {
        // Create annotation root node if it does not exist
        self.root.get_or_insert(
            scene.add_node(
                None, Some("Transform annotations".to_owned()), Transform::IDENTITY, NodeFlags::inert()
            ).expect("Failed to create transform annotation root node")
        );

        // Clear previous annotations
        for id in self.nodes.drain(..) {
            scene.remove_node(id);
        }

        // A single-axis constraint draws one line along its axis; a plane
        // constraint draws its two in-plane axis lines.
        for (axis, direction) in constraint_lines(interaction) {
            self.draw_axis_line(axis, direction, interaction, scene);
        }
    }

    /// Draw one colored line through the pivot along `direction`, colored for `axis`.
    fn draw_axis_line(
        &mut self,
        axis: Axis,
        direction: Vector3,
        interaction: &TransformInteraction,
        scene: &mut Scene,
    ) {
        let half_length = interaction.model_radius() * 2.0;
        let start = interaction.pivot() - direction * half_length;
        let end = interaction.pivot() + direction * half_length;
        let mesh_id = scene.add_mesh(Mesh::line(start, end));

        // Reuse the cached material for this axis's color, recreating it if it
        // was pruned since last use (unreferenced resource sweep).
        let material = match self.axis_materials.get(&axis).copied() {
            Some(m) if scene.get_line_material(m).is_some() => m,
            _ => {
                let m = scene.add_line_material(LineMaterial::new(axis.color()));
                self.axis_materials.insert(axis, m);
                m
            }
        };

        let id = scene.add_instance_node(
            self.root,
            Instance::new(mesh_id).with_line_material(material),
            Some("Transform axis annotation".to_owned()),
            Transform::IDENTITY,
            NodeFlags::inert()
        ).expect("Failed to create axis annotation");
        self.nodes.push(id);
    }

    /// Remove the annotation lines.
    pub fn clear(&mut self, scene: &mut Scene) {
        for id in self.nodes.drain(..) {
            scene.remove_node(id);
        }
    }
}

impl Default for TransformAnnotations {
    fn default() -> Self {
        Self::new()
    }
}

/// The colored lines to draw for the interaction's current constraint:
/// `(axis-for-color, world-space direction)`. Empty when unconstrained.
fn constraint_lines(interaction: &TransformInteraction) -> Vec<(Axis, Vector3)> {
    use crate::common::{local_axis_x, local_axis_y, local_axis_z};
    use crate::gizmo::plane_in_axes;
    use super::interaction::ConstraintSpace;

    let frame = interaction.frame_rotation();
    let dir = |axis: Axis, space: ConstraintSpace| -> Vector3 {
        match space {
            ConstraintSpace::World => axis.direction(),
            ConstraintSpace::Local => match axis {
                Axis::X => local_axis_x(frame),
                Axis::Y => local_axis_y(frame),
                Axis::Z => local_axis_z(frame),
            },
        }
    };

    match interaction.axis_constraint() {
        AxisConstraint::None => Vec::new(),
        AxisConstraint::Axis(axis, space) => vec![(axis, dir(axis, space).normalize())],
        AxisConstraint::Plane(normal, space) => {
            let (u, v) = plane_in_axes(normal);
            vec![(u, dir(u, space).normalize()), (v, dir(v, space).normalize())]
        }
    }
}
