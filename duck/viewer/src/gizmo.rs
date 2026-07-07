//! Gizmo geometry builders for transform handles (translate, rotate, scale).
//!
//! Each builder produces a set of [`GizmoHandle`]s — one per axis — containing
//! a mesh and material ready to be added to the scene. All geometry is built at
//! the origin; positioning at the selection pivot is done via node transforms.

use duck_engine_common::{Deg, Matrix4, Point3, Vector3};
use duck_engine_scene::{NodeFlags, Scene};

use crate::common::{Axis, Ray, RgbaColor, Transform};
use crate::geom_query::{pick_all_from_ray, RayPickQuery};
use crate::scene::{
    AlphaMode, DisplayBehavior, FaceMaterial, FaceMaterialId, Instance, MaterialFlags, Mesh,
    MeshId, NodeId, PrimitiveType, RenderLayer,
};

const GIZMO_FLAGS: MaterialFlags = MaterialFlags::DO_NOT_LIGHT
    .union(MaterialFlags::DOUBLE_SIDED);

const SEGMENTS: u32 = 16;

/// Which type of gizmo to display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoType {
    Translate,
    Rotate,
    Scale,
}

/// A single gizmo handle (one axis of a gizmo).
pub struct GizmoHandle {
    pub mesh: Mesh,
    pub material: FaceMaterial,
    pub axis: Axis,
}

/// Brighter color for hover/active highlighting.
pub fn highlight_color(axis: Axis) -> RgbaColor {
    match axis {
        Axis::X => RgbaColor { r: 1.0, g: 0.5, b: 0.3, a: 1.0 },
        Axis::Y => RgbaColor { r: 0.5, g: 1.0, b: 0.3, a: 1.0 },
        Axis::Z => RgbaColor { r: 0.3, g: 0.5, b: 1.0, a: 1.0 },
    }
}

/// Create a gizmo material for the given axis color.
fn gizmo_material(color: RgbaColor) -> FaceMaterial {
    FaceMaterial::new()
        .with_base_color_factor(color)
        .with_flags(GIZMO_FLAGS)
        .with_alpha_mode(AlphaMode::Opaque)
}

/// Build translation gizmo handles: cylinder shaft + cone arrowhead per axis.
pub fn build_translate_handles(size: f32) -> Vec<GizmoHandle> {
    let shaft_radius = size * 0.025;
    let shaft_height = size * 0.8;
    let cone_radius = size * 0.08;
    let cone_height = size * 0.2;

    Axis::ALL
        .into_iter()
        .map(|axis| {
            let rotation = axis.rotation_from_y();

            let shaft = Mesh::cylinder(
                shaft_radius,
                shaft_height,
                SEGMENTS,
                false,
                PrimitiveType::TriangleList,
            )
            .transformed(
                &(rotation
                    * Matrix4::from_translation(Vector3::new(0.0, shaft_height / 2.0, 0.0))),
            );

            let cone_base_y = shaft_height + cone_height / 2.0;
            let cone = Mesh::cone(
                cone_radius,
                cone_height,
                SEGMENTS,
                true,
                PrimitiveType::TriangleList,
            )
            .transformed(
                &(rotation * Matrix4::from_translation(Vector3::new(0.0, cone_base_y, 0.0))),
            );

            let mesh = shaft.merged(&cone);
            let material = gizmo_material(axis.color());

            GizmoHandle {
                mesh,
                material,
                axis,
            }
        })
        .collect()
}

/// Build scale gizmo handles: cylinder shaft + cube tip per axis.
pub fn build_scale_handles(size: f32) -> Vec<GizmoHandle> {
    let shaft_radius = size * 0.025;
    let shaft_height = size * 0.8;
    let cube_size = size * 0.1;

    Axis::ALL
        .into_iter()
        .map(|axis| {
            let rotation = axis.rotation_from_y();

            let shaft = Mesh::cylinder(
                shaft_radius,
                shaft_height,
                SEGMENTS,
                false,
                PrimitiveType::TriangleList,
            )
            .transformed(
                &(rotation
                    * Matrix4::from_translation(Vector3::new(0.0, shaft_height / 2.0, 0.0))),
            );

            let cube_y = shaft_height + cube_size / 2.0;
            let cube = Mesh::cube(cube_size, PrimitiveType::TriangleList)
                .transformed(
                    &(rotation * Matrix4::from_translation(Vector3::new(0.0, cube_y, 0.0))),
                );

            let mesh = shaft.merged(&cube);
            let material = gizmo_material(axis.color());

            GizmoHandle {
                mesh,
                material,
                axis,
            }
        })
        .collect()
}

/// Build rotation gizmo handles: one torus ring per axis.
pub fn build_rotate_handles(size: f32) -> Vec<GizmoHandle> {
    let major_radius = size;
    let minor_radius = size * 0.02;

    Axis::ALL
        .into_iter()
        .map(|axis| {
            // Torus is built in the XZ plane (Y-axis rotation ring).
            // Rotate to orient for the other axes.
            let rotation = match axis {
                Axis::Y => Matrix4::from_scale(1.0),
                // XZ plane torus rotated 90 around Z → lies in YZ plane (X-axis ring)
                Axis::X => Matrix4::from_angle_z(Deg(90.0)),
                // XZ plane torus rotated 90 around X → lies in XY plane (Z-axis ring)
                Axis::Z => Matrix4::from_angle_x(Deg(90.0)),
            };

            let mesh = Mesh::torus(
                major_radius,
                minor_radius,
                32,
                SEGMENTS,
                PrimitiveType::TriangleList,
            )
            .transformed(&rotation);
            let material = gizmo_material(axis.color());

            GizmoHandle {
                mesh,
                material,
                axis,
            }
        })
        .collect()
}

/// Build gizmo handles for a given type.
pub fn build_handles(gizmo_type: GizmoType, size: f32) -> Vec<GizmoHandle> {
    match gizmo_type {
        GizmoType::Translate => build_translate_handles(size),
        GizmoType::Rotate => build_rotate_handles(size),
        GizmoType::Scale => build_scale_handles(size),
    }
}

/// Owns a gizmo's scene resources: handle nodes/meshes/materials, hover
/// highlighting, picking, and repositioning at a pivot.
///
/// Gizmo nodes are parented under a root node drawn on the overlay layer so
/// they are grouped with other overlay content.
pub struct GizmoState {
    /// Root node for all gizmo geometry. Created when first needed.
    root_node: Option<NodeId>,
    /// Node IDs of the gizmo handles (one per axis: X, Y, Z).
    node_ids: Vec<NodeId>,
    /// Mesh IDs added to the scene for gizmo geometry.
    mesh_ids: Vec<MeshId>,
    /// Material IDs added to the scene for gizmo handles.
    material_ids: Vec<FaceMaterialId>,
    /// Which axis is currently highlighted (hovered or active).
    highlighted_axis: Option<Axis>,
    /// Current gizmo type being displayed.
    current_type: Option<GizmoType>,
}

impl GizmoState {
    pub fn new() -> Self {
        Self {
            root_node: None,
            node_ids: Vec::new(),
            mesh_ids: Vec::new(),
            material_ids: Vec::new(),
            highlighted_axis: None,
            current_type: None,
        }
    }

    pub fn has_gizmo(&self) -> bool {
        self.current_type.is_some()
    }

    /// The handle set currently displayed, if any.
    pub fn current_type(&self) -> Option<GizmoType> {
        self.current_type
    }

    /// Build and add gizmo handles to the scene at the given pivot point.
    pub fn show(&mut self, gizmo_type: GizmoType, pivot: Point3, size: f32, scene: &mut Scene) {
        self.hide(scene);

        self.root_node.get_or_insert_with(|| {
            let id = scene.add_node(
                None, Some("Gizmo root".to_owned()), Transform::IDENTITY, NodeFlags::DO_NOT_EXPORT
            ).expect("Failed to create Gizmo root node");
            // Draw the gizmo on the overlay layer; handles inherit it.
            scene.set_node_display(id, DisplayBehavior { layer: RenderLayer::Overlay, ..Default::default() });
            id
        });

        let handles = build_handles(gizmo_type, size);
        let pivot_transform = Transform::from_position(pivot);

        for handle in handles {
            let mesh_id = scene.add_mesh(handle.mesh);
            let material_id = scene.add_face_material(handle.material);
            let node_id = scene
                .add_instance_node(
                    self.root_node,
                    Instance::new(mesh_id).with_face_material(material_id),
                    None,
                    pivot_transform,
                    NodeFlags::DO_NOT_EXPORT
                )
                .expect("Failed to add gizmo node");

            self.node_ids.push(node_id);
            self.mesh_ids.push(mesh_id);
            self.material_ids.push(material_id);
        }

        self.current_type = Some(gizmo_type);
    }

    /// Remove all gizmo geometry from the scene.
    pub fn hide(&mut self, scene: &mut Scene) {
        for &node_id in &self.node_ids {
            scene.remove_node(node_id);
        }
        for &mesh_id in &self.mesh_ids {
            scene.remove_mesh(mesh_id);
        }
        for &material_id in &self.material_ids {
            scene.remove_face_material(material_id);
        }

        self.node_ids.clear();
        self.mesh_ids.clear();
        self.material_ids.clear();
        self.highlighted_axis = None;
        self.current_type = None;
    }

    /// Update the gizmo position (e.g. when pivot changes).
    pub fn update_position(&self, pivot: Point3, scene: &mut Scene) {
        for &node_id in &self.node_ids {
            if scene.has_node(node_id) {
                scene.set_node_position(node_id, pivot);
            }
        }
    }

    /// Pick which gizmo handle (if any) the ray hits.
    pub fn pick_handle(&self, ray: Ray, scene: &Scene) -> Option<Axis> {
        if !self.has_gizmo() {
            return None;
        }

        let results = pick_all_from_ray(&RayPickQuery::faces(ray), scene);

        // Find the first hit that matches a gizmo node
        for result in &results {
            for (i, &node_id) in self.node_ids.iter().enumerate() {
                if result.node_id == node_id {
                    return Some(Axis::ALL[i]);
                }
            }
        }

        None
    }

    /// Highlight a specific axis handle (or clear highlight with None).
    pub fn set_highlight(&mut self, axis: Option<Axis>, scene: &mut Scene) {
        if self.highlighted_axis == axis {
            return;
        }

        // Restore previous highlight to normal color
        if let Some(prev_axis) = self.highlighted_axis {
            let idx = axis_index(prev_axis);
            if let Some(&mat_id) = self.material_ids.get(idx)
                && let Some(mat) = scene.get_face_material_mut(mat_id) {
                    mat.set_base_color_factor(prev_axis.color());
                }
        }

        // Apply highlight color to new axis
        if let Some(new_axis) = axis {
            let idx = axis_index(new_axis);
            if let Some(&mat_id) = self.material_ids.get(idx)
                && let Some(mat) = scene.get_face_material_mut(mat_id) {
                    mat.set_base_color_factor(highlight_color(new_axis));
                }
        }

        self.highlighted_axis = axis;
    }
}

impl Default for GizmoState {
    fn default() -> Self {
        Self::new()
    }
}

/// Maps an axis to its index in the gizmo handle arrays (X=0, Y=1, Z=2).
fn axis_index(axis: Axis) -> usize {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

#[cfg(test)]
mod tests {
    use duck_engine_scene::{Instance, NodeFlags};

use crate::geom_query::RayPickQuery;

    use super::*;

    #[test]
    fn translate_handles_have_three_axes() {
        let handles = build_translate_handles(1.0);
        assert_eq!(handles.len(), 3);
        assert_eq!(handles[0].axis, Axis::X);
        assert_eq!(handles[1].axis, Axis::Y);
        assert_eq!(handles[2].axis, Axis::Z);
    }

    #[test]
    fn translate_handles_have_triangle_geometry() {
        for handle in build_translate_handles(1.0) {
            assert!(!handle.mesh.vertices().is_empty());
            assert!(handle.mesh.has_primitive_type(PrimitiveType::TriangleList));
        }
    }

    #[test]
    fn scale_handles_have_three_axes() {
        let handles = build_scale_handles(1.0);
        assert_eq!(handles.len(), 3);
    }

    #[test]
    fn scale_handles_have_triangle_geometry() {
        for handle in build_scale_handles(1.0) {
            assert!(!handle.mesh.vertices().is_empty());
            assert!(handle.mesh.has_primitive_type(PrimitiveType::TriangleList));
        }
    }

    #[test]
    fn rotate_handles_have_three_axes() {
        let handles = build_rotate_handles(1.0);
        assert_eq!(handles.len(), 3);
    }

    #[test]
    fn rotate_handles_have_triangle_geometry() {
        for handle in build_rotate_handles(1.0) {
            assert!(!handle.mesh.vertices().is_empty());
            assert!(handle.mesh.has_primitive_type(PrimitiveType::TriangleList));
        }
    }

    #[test]
    fn build_handles_dispatches_correctly() {
        assert_eq!(build_handles(GizmoType::Translate, 1.0).len(), 3);
        assert_eq!(build_handles(GizmoType::Rotate, 1.0).len(), 3);
        assert_eq!(build_handles(GizmoType::Scale, 1.0).len(), 3);
    }

    #[test]
    fn gizmo_materials_have_correct_flags() {
        for handle in build_translate_handles(1.0) {
            let flags = handle.material.flags();
            assert!(flags.contains(MaterialFlags::DO_NOT_LIGHT));
            assert!(flags.contains(MaterialFlags::DOUBLE_SIDED));
        }
    }

    /// Fire a ray at the center of each gizmo handle's AABB and verify
    /// that both the AABB and mesh triangle intersection succeed.
    #[test]
    fn translate_handles_pickable_by_ray() {
        use crate::common::Ray;
        use crate::geom_query::intersect_ray;
        use duck_engine_common::Vector3;

        for handle in build_translate_handles(1.0) {
            let aabb = handle.mesh.bounding().expect("handle should have bounds");
            let center = aabb.center();

            // Try rays from all 3 principal directions toward the center
            let directions = [
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(-1.0, 0.0, 0.0),
            ];

            let mut any_mesh_hit = false;
            for dir in &directions {
                let origin = center - dir * 10.0;
                let ray = Ray::new(origin, *dir);

                let aabb_hit = aabb.intersects_ray(&ray);

                if aabb_hit.is_some() {
                    let mesh_hits = intersect_ray(&handle.mesh, &ray);
                    if !mesh_hits.is_empty() {
                        any_mesh_hit = true;
                    }
                }
            }

            assert!(
                any_mesh_hit,
                "Axis {:?}: ray through AABB center hit the AABB but missed all triangles. \
                 vertices={}, triangles={}, aabb_min={:?}, aabb_max={:?}",
                handle.axis,
                handle.mesh.vertices().len(),
                handle.mesh.triangles().count(),
                aabb.min,
                aabb.max,
            );
        }
    }

    #[test]
    fn scale_handles_pickable_by_ray() {
        use crate::common::Ray;
        use crate::geom_query::intersect_ray;
        use duck_engine_common::Vector3;

        for handle in build_scale_handles(1.0) {
            let aabb = handle.mesh.bounding().expect("handle should have bounds");
            let center = aabb.center();

            let directions = [
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(-1.0, 0.0, 0.0),
            ];

            let mut any_mesh_hit = false;
            for dir in &directions {
                let origin = center - dir * 10.0;
                let ray = Ray::new(origin, *dir);

                if aabb.intersects_ray(&ray).is_some() {
                    let mesh_hits = intersect_ray(&handle.mesh, &ray);
                    if !mesh_hits.is_empty() {
                        any_mesh_hit = true;
                    }
                }
            }

            assert!(
                any_mesh_hit,
                "Axis {:?}: ray through AABB center missed all triangles. \
                 vertices={}, triangles={}",
                handle.axis,
                handle.mesh.vertices().len(),
                handle.mesh.triangles().count(),
            );
        }
    }

    #[test]
    fn rotate_handles_pickable_by_ray() {
        use crate::common::Ray;
        use crate::geom_query::intersect_ray;
        use duck_engine_common::Vector3;

        for handle in build_rotate_handles(1.0) {
            let aabb = handle.mesh.bounding().expect("handle should have bounds");
            let center = aabb.center();

            let directions = [
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(-1.0, 0.0, 0.0),
            ];

            let mut any_mesh_hit = false;
            for dir in &directions {
                let origin = center - dir * 10.0;
                let ray = Ray::new(origin, *dir);

                if aabb.intersects_ray(&ray).is_some() {
                    let mesh_hits = intersect_ray(&handle.mesh, &ray);
                    if !mesh_hits.is_empty() {
                        any_mesh_hit = true;
                    }
                }
            }

            assert!(
                any_mesh_hit,
                "Axis {:?}: ray through AABB center missed all triangles. \
                 vertices={}, triangles={}",
                handle.axis,
                handle.mesh.vertices().len(),
                handle.mesh.triangles().count(),
            );
        }
    }

    /// Test picking gizmo handles through the full scene pipeline,
    /// simulating how GizmoState::show() sets up the handles.
    #[test]
    fn translate_handles_pickable_through_scene() {
        use crate::common::Ray;
        use crate::geom_query::pick_all_from_ray;
        use crate::scene::{FaceMaterial, Mesh, PrimitiveType, Scene};
        use duck_engine_common::{Point3, Vector3};

        let pivot = Point3::new(5.0, 3.0, 0.0);

        let mut scene = Scene::new();

        // Add a model cube so the scene isn't empty (like the real app)
        let cube = Mesh::cube(2.0, PrimitiveType::TriangleList);
        let cube_mesh_id = scene.add_mesh(cube);
        let cube_mat_id = scene.add_face_material(FaceMaterial::new());
        scene
            .add_instance_node(
                None,
                Instance::new(cube_mesh_id).with_face_material(cube_mat_id),
                Some("Cube".to_string()),
                crate::common::Transform::IDENTITY,
                NodeFlags::NONE
            )
            .unwrap();

        // Simulate sync_gizmo: compute scene.bounding() BEFORE showing gizmo
        // (this is what the real code does — it reads model_radius first)
        let _ = scene.bounding();

        let pivot_transform = crate::common::Transform::from_position(pivot);

        let handles = build_translate_handles(1.0);
        let mut node_ids = Vec::new();

        for handle in &handles {
            let mesh_id = scene.add_mesh(handle.mesh.clone());
            let material_id = scene.add_face_material(handle.material.clone());
            let node_id = scene
                .add_instance_node(
                    None,
                    Instance::new(mesh_id).with_face_material(material_id),
                    None,
                    pivot_transform,
                    NodeFlags::NONE
                )
                .expect("Failed to add gizmo node");
            node_ids.push(node_id);
        }

        // For each handle, fire a ray at its world-space AABB center
        for (i, handle) in handles.iter().enumerate() {
            let local_aabb = handle.mesh.bounding().expect("handle should have bounds");
            let local_center = local_aabb.center();
            let world_center = Point3::new(
                local_center.x + pivot.x,
                local_center.y + pivot.y,
                local_center.z + pivot.z,
            );

            let directions = [
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(-1.0, 0.0, 0.0),
            ];

            let mut found = false;
            for dir in &directions {
                let origin = world_center - dir * 10.0;
                let ray = Ray::new(origin, *dir);
                let query = RayPickQuery::faces(ray);

                let results = pick_all_from_ray(&query, &scene);
                for result in &results {
                    if result.node_id == node_ids[i] {
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }

            assert!(
                found,
                "Axis {:?}: full scene pipeline failed to pick gizmo handle at pivot {:?}",
                handle.axis, pivot,
            );
        }
    }

    /// Test that picking works after gizmo position is updated
    /// (simulates update_position path from sync_gizmo).
    #[test]
    fn translate_handles_pickable_after_position_update() {
        use crate::common::Ray;
        use crate::geom_query::pick_all_from_ray;
        use crate::scene::Scene;
        use duck_engine_common::{Point3, Vector3};

        let initial_pivot = Point3::new(0.0, 0.0, 0.0);
        let new_pivot = Point3::new(10.0, 5.0, 3.0);

        let mut scene = Scene::new();
        let pivot_transform = crate::common::Transform::from_position(initial_pivot);

        let handles = build_translate_handles(1.0);
        let mut node_ids = Vec::new();

        for handle in &handles {
            let mesh_id = scene.add_mesh(handle.mesh.clone());
            let material_id = scene.add_face_material(handle.material.clone());
            let node_id = scene
                .add_instance_node(
                    None,
                    Instance::new(mesh_id).with_face_material(material_id),
                    None,
                    pivot_transform,
                    NodeFlags::NONE
                )
                .expect("Failed to add gizmo node");
            node_ids.push(node_id);
        }

        // Cache bounds (simulates a frame between show and pick)
        let _ = scene.bounding();

        // Move gizmo to new position (simulates update_position)
        for &node_id in &node_ids {
            scene.set_node_position(node_id, new_pivot);
        }

        // Try to pick at the NEW position
        for (i, handle) in handles.iter().enumerate() {
            let local_aabb = handle.mesh.bounding().expect("handle should have bounds");
            let local_center = local_aabb.center();
            let world_center = Point3::new(
                local_center.x + new_pivot.x,
                local_center.y + new_pivot.y,
                local_center.z + new_pivot.z,
            );

            let directions = [
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(-1.0, 0.0, 0.0),
            ];

            let mut found = false;
            for dir in &directions {
                let origin = world_center - dir * 10.0;
                let ray = Ray::new(origin, *dir);
                let query = RayPickQuery::faces(ray);

                let results = pick_all_from_ray(&query, &scene);
                for result in &results {
                    if result.node_id == node_ids[i] {
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }

            assert!(
                found,
                "Axis {:?}: picking failed after position update to {:?}",
                handle.axis, new_pivot,
            );
        }
    }
}
