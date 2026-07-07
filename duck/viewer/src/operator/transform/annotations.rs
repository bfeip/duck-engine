//! Constraint-axis annotation lines shown during an interactive transform.

use std::collections::HashMap;

use duck_engine_scene::{Mesh, NodeFlags, Scene};

use crate::common::{Axis, Transform};
use crate::scene::{Instance, LineMaterial, LineMaterialId, NodeId};

use super::interaction::{axis_from_constraint, TransformInteraction};

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

        // Add axis constraint line if constrained
        let constraint = interaction.axis_constraint();
        if let Some(color) = constraint.color()
            && let Some(axis) = interaction.constraint_axis() {
                let half_length = interaction.model_radius() * 2.0;
                let start = interaction.pivot() - axis * half_length;
                let end = interaction.pivot() + axis * half_length;
                let mesh = Mesh::line(start, end);
                let mesh_id = scene.add_mesh(mesh);

                // Get or insert the material for this axis annotation
                let create_color_material = |scene: &mut Scene| {
                    scene.add_line_material(LineMaterial::new(color))
                };
                let mut material = self.axis_materials.entry(
                    axis_from_constraint(&constraint).unwrap()
                ).or_insert(create_color_material(scene)).to_owned();
                if scene.get_line_material(material).is_none() {
                    // Our material was removed from the scene since we last used it.
                    // This can happen if, while unused, the scene removed all unreferenced
                    // resources. We'll have to reinsert the material.
                    material = create_color_material(scene);
                }

                let id = scene.add_instance_node(
                    self.root,
                    Instance::new(mesh_id).with_line_material(material),
                    Some("Transform axis annotation".to_owned()),
                    Transform::IDENTITY,
                    NodeFlags::inert()
                ).expect("Failed to create axis annotation");
                self.nodes.push(id);
            }
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
