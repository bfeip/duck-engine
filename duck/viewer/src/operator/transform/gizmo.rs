//! Transform gizmo handles: geometry builders and scene-side state.
//!
//! The free builders produce a set of [`GizmoHandle`]s — one per axis —
//! containing a mesh and material, built at the origin at unit size.
//! [`GizmoState`] owns the gizmo's scene resources: it adds/removes the handle
//! nodes, positions them at the pivot, hit-tests them, and drives highlights.

use duck_engine_common::{Deg, Matrix4, Point3, Vector3, Vector4};
use duck_engine_scene::{NodeFlags, PositionedCamera, Scene, SceneData};

use crate::common::{Axis, Ray, RgbaColor, Transform};
use crate::geom_query::{pick_all_from_ray_with_view, PickView, RayPickQuery};
use crate::scene::{
    DisplayBehavior, FaceMaterial, FaceMaterialId, Instance, MaterialFlags, Mesh,
    MeshId, NodeId, PrimitiveType, RenderLayer,
};

const GIZMO_FLAGS: MaterialFlags = MaterialFlags::DO_NOT_LIGHT
    .union(MaterialFlags::DOUBLE_SIDED);

const SEGMENTS: u32 = 16;

/// On-screen pixel size of the gizmo's unit arm extent. Handle geometry is built
/// at a unit size (arms span 0..1) and held at this constant pixel size via
/// [`DisplayBehavior::screen_size`], so the gizmo no longer scales with zoom.
const GIZMO_SCREEN_SIZE: f32 = 90.0;

/// Neutral color for the center ball (uniform scale / view-plane translate).
const BALL_COLOR: RgbaColor = RgbaColor { r: 0.85, g: 0.85, b: 0.85, a: 1.0 };

/// Which type of gizmo to display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoType {
    Translate,
    Rotate,
    Scale,
}

/// Identifies a gizmo handle: a single axis arm, a coordinate-plane handle
/// (identified by its excluded/normal axis), or the center ball.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoHandleId {
    /// A single-axis arm (arrow / cube tip / rotation ring).
    Axis(Axis),
    /// A coordinate-plane handle. The axis is the plane's normal (excluded)
    /// axis, so `Plane(Axis::Z)` is the XY-plane handle.
    Plane(Axis),
    /// The center ball: uniform scale, or view-plane (camera-relative) translate.
    Ball,
}

/// A single gizmo handle.
pub struct GizmoHandle {
    pub mesh: Mesh,
    pub material: FaceMaterial,
    pub id: GizmoHandleId,
}

/// The two in-plane axes for a plane whose normal is `normal`.
pub fn plane_in_axes(normal: Axis) -> (Axis, Axis) {
    match normal {
        Axis::X => (Axis::Y, Axis::Z),
        Axis::Y => (Axis::Z, Axis::X),
        Axis::Z => (Axis::X, Axis::Y),
    }
}

/// Translucent fill color for a plane handle: blend of its two in-plane axes.
fn plane_color(normal: Axis) -> RgbaColor {
    let (a, b) = plane_in_axes(normal);
    let (ca, cb) = (a.color(), b.color());
    RgbaColor {
        r: (ca.r + cb.r) * 0.5,
        g: (ca.g + cb.g) * 0.5,
        b: (ca.b + cb.b) * 0.5,
        a: 0.5,
    }
}

/// Base (unhighlighted) color for a handle.
fn handle_base_color(id: GizmoHandleId) -> RgbaColor {
    match id {
        GizmoHandleId::Axis(axis) => axis.color(),
        GizmoHandleId::Plane(normal) => plane_color(normal),
        GizmoHandleId::Ball => BALL_COLOR,
    }
}

/// Brighter color for hover/active highlighting of an axis handle.
pub fn highlight_color(axis: Axis) -> RgbaColor {
    match axis {
        Axis::X => RgbaColor { r: 1.0, g: 0.5, b: 0.3, a: 1.0 },
        Axis::Y => RgbaColor { r: 0.5, g: 1.0, b: 0.3, a: 1.0 },
        Axis::Z => RgbaColor { r: 0.3, g: 0.5, b: 1.0, a: 1.0 },
    }
}

/// Highlight color for any handle (hover/active).
fn handle_highlight_color(id: GizmoHandleId) -> RgbaColor {
    match id {
        GizmoHandleId::Axis(axis) => highlight_color(axis),
        GizmoHandleId::Plane(normal) => {
            let c = plane_color(normal);
            // Brighten the fill and make it more opaque on highlight.
            RgbaColor {
                r: (c.r + 0.4).min(1.0),
                g: (c.g + 0.4).min(1.0),
                b: (c.b + 0.4).min(1.0),
                a: 0.8,
            }
        }
        GizmoHandleId::Ball => RgbaColor { r: 1.0, g: 1.0, b: 0.5, a: 1.0 },
    }
}

/// Create a gizmo material for the given color, blending when it is translucent.
fn gizmo_material(color: RgbaColor) -> FaceMaterial {
    FaceMaterial::new().with_base_color_factor(color).with_flags(GIZMO_FLAGS)
}

/// Build a plane handle: a small translucent quad in the coordinate plane whose
/// normal is `normal`, offset into the positive corner between its two axes.
fn build_plane_handle(normal: Axis, size: f32) -> GizmoHandle {
    let quad_size = size * 0.22;
    let center_dist = size * 0.3;
    let (u_axis, v_axis) = plane_in_axes(normal);
    let (u, v, n) = (u_axis.direction(), v_axis.direction(), normal.direction());
    let translation = (u + v) * center_dist;

    // Orient the authored XY quad (local x→u, local y→v, local z→normal) and
    // place its center in the positive corner between the two in-plane axes.
    let orient = Matrix4::from_cols(
        u.extend(0.0),
        v.extend(0.0),
        n.extend(0.0),
        Vector4::new(translation.x, translation.y, translation.z, 1.0),
    );

    let mesh = Mesh::quad(quad_size, quad_size, PrimitiveType::TriangleList)
        .transformed(&orient);
    let material = gizmo_material(plane_color(normal));

    GizmoHandle { mesh, material, id: GizmoHandleId::Plane(normal) }
}

/// Build the center ball handle (uniform scale / view-plane translate).
fn build_ball_handle(size: f32) -> GizmoHandle {
    let mesh = Mesh::sphere(size * 0.09, SEGMENTS, SEGMENTS / 2, PrimitiveType::TriangleList);
    let material = gizmo_material(BALL_COLOR);
    GizmoHandle { mesh, material, id: GizmoHandleId::Ball }
}

/// Build translation gizmo handles: cylinder shaft + cone arrowhead per axis.
pub fn build_translate_handles(size: f32) -> Vec<GizmoHandle> {
    let shaft_radius = size * 0.025;
    let shaft_height = size * 0.8;
    let cone_radius = size * 0.08;
    let cone_height = size * 0.2;

    let mut handles: Vec<GizmoHandle> = Axis::ALL
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

            GizmoHandle { mesh, material, id: GizmoHandleId::Axis(axis) }
        })
        .collect();

    for normal in Axis::ALL {
        handles.push(build_plane_handle(normal, size));
    }
    handles.push(build_ball_handle(size));
    handles
}

/// Build scale gizmo handles: cylinder shaft + cube tip per axis.
pub fn build_scale_handles(size: f32) -> Vec<GizmoHandle> {
    let shaft_radius = size * 0.025;
    let shaft_height = size * 0.8;
    let cube_size = size * 0.1;

    let mut handles: Vec<GizmoHandle> = Axis::ALL
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

            GizmoHandle { mesh, material, id: GizmoHandleId::Axis(axis) }
        })
        .collect();

    for normal in Axis::ALL {
        handles.push(build_plane_handle(normal, size));
    }
    handles.push(build_ball_handle(size));
    handles
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

            GizmoHandle { mesh, material, id: GizmoHandleId::Axis(axis) }
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

/// Scene resources backing one displayed handle.
struct HandleResources {
    node: NodeId,
    mesh: MeshId,
    material: FaceMaterialId,
    id: GizmoHandleId,
}

/// Owns a gizmo's scene resources: handle nodes/meshes/materials, hover
/// highlighting, picking, and repositioning at a pivot.
///
/// Gizmo nodes are parented under a root node drawn on the overlay layer so
/// they are grouped with other overlay content.
pub struct GizmoState {
    /// Root node for all gizmo geometry. Created when first needed.
    root_node: Option<NodeId>,
    /// Scene resources of the currently displayed handles.
    handles: Vec<HandleResources>,
    /// Which handle is currently highlighted (hovered or active).
    highlighted: Option<GizmoHandleId>,
    /// Current gizmo type being displayed.
    current_type: Option<GizmoType>,
}

impl GizmoState {
    pub fn new() -> Self {
        Self {
            root_node: None,
            handles: Vec::new(),
            highlighted: None,
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
    ///
    /// Handles are built at a unit size and held at a constant on-screen size via
    /// [`DisplayBehavior::screen_size`] on the root, which the handles inherit.
    pub fn show(&mut self, gizmo_type: GizmoType, pivot: Point3, scene: &Scene) {
        self.show_in(gizmo_type, pivot, &mut scene.lock());
    }

    fn show_in(&mut self, gizmo_type: GizmoType, pivot: Point3, scene: &mut SceneData) {
        self.hide_in(scene);

        self.root_node.get_or_insert_with(|| {
            let id = scene.add_node(
                None, Some("Gizmo root".to_owned()), Transform::IDENTITY, NodeFlags::NONE
            ).expect("Failed to create Gizmo root node");
            // Draw the gizmo on the overlay layer at a constant on-screen size;
            // handles inherit both.
            scene.set_node_display(id, DisplayBehavior {
                screen_size: Some(GIZMO_SCREEN_SIZE),
                layer: RenderLayer::Overlay,
                ..Default::default()
            });
            id
        });

        let handles = build_handles(gizmo_type, 1.0);
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
                    NodeFlags::NONE
                )
                .expect("Failed to add gizmo node");

            self.handles.push(HandleResources {
                node: node_id,
                mesh: mesh_id,
                material: material_id,
                id: handle.id,
            });
        }

        self.current_type = Some(gizmo_type);
    }

    /// Remove all gizmo geometry from the scene.
    pub fn hide(&mut self, scene: &Scene) {
        self.hide_in(&mut scene.lock());
    }

    fn hide_in(&mut self, scene: &mut SceneData) {
        for handle in &self.handles {
            scene.remove_node(handle.node);
            scene.remove_mesh(handle.mesh);
            scene.remove_face_material(handle.material);
        }

        self.handles.clear();
        self.highlighted = None;
        self.current_type = None;
    }

    /// Update the gizmo position (e.g. when pivot changes).
    pub fn update_position(&self, pivot: Point3, scene: &Scene) {
        let mut scene = scene.lock();
        for handle in &self.handles {
            if scene.has_node(handle.node) {
                scene.set_node_position(handle.node, pivot);
            }
        }
    }

    /// Pick which gizmo handle (if any) the ray hits.
    ///
    /// The gizmo is drawn at a constant on-screen size, so picking must resolve
    /// its handles against the same camera-dependent transform via [`PickView`];
    /// otherwise hits would be tested against the authored unit-size geometry.
    pub fn pick_handle(
        &self,
        ray: Ray,
        scene: &Scene,
        camera: &PositionedCamera,
        viewport: (u32, u32),
    ) -> Option<GizmoHandleId> {
        if !self.has_gizmo() {
            return None;
        }

        let view = PickView { camera, viewport };
        let results =
            pick_all_from_ray_with_view(&RayPickQuery::faces(ray), scene, Some(&view));

        // Find the first hit that matches a gizmo node
        for result in &results {
            if let Some(handle) = self.handles.iter().find(|h| h.node == result.node_id) {
                return Some(handle.id);
            }
        }

        None
    }

    /// The material backing a handle, by identity.
    fn handle_material(&self, id: GizmoHandleId) -> Option<FaceMaterialId> {
        self.handles.iter().find(|h| h.id == id).map(|h| h.material)
    }

    /// Highlight a specific handle (or clear highlight with None).
    pub fn set_highlight(&mut self, handle: Option<GizmoHandleId>, scene: &Scene) {
        if self.highlighted == handle {
            return;
        }

        let mut scene = scene.lock();

        // Restore previous highlight to normal color
        if let Some(prev) = self.highlighted
            && let Some(mat_id) = self.handle_material(prev)
            && let Some(mat) = scene.get_face_material_mut(mat_id) {
                mat.set_base_color_factor(handle_base_color(prev));
            }

        // Apply highlight color to new handle
        if let Some(new) = handle
            && let Some(mat_id) = self.handle_material(new)
            && let Some(mat) = scene.get_face_material_mut(mat_id) {
                mat.set_base_color_factor(handle_highlight_color(new));
            }

        self.highlighted = handle;
    }
}

impl Default for GizmoState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use duck_engine_scene::{Instance, NodeFlags};

use crate::geom_query::RayPickQuery;

    use super::*;

    #[test]
    fn translate_handles_have_axis_plane_and_ball() {
        let handles = build_translate_handles(1.0);
        // 3 axis arms + 3 plane handles + 1 center ball.
        assert_eq!(handles.len(), 7);
        assert_eq!(handles[0].id, GizmoHandleId::Axis(Axis::X));
        assert_eq!(handles[1].id, GizmoHandleId::Axis(Axis::Y));
        assert_eq!(handles[2].id, GizmoHandleId::Axis(Axis::Z));
        assert_eq!(handles[3].id, GizmoHandleId::Plane(Axis::X));
        assert_eq!(handles[4].id, GizmoHandleId::Plane(Axis::Y));
        assert_eq!(handles[5].id, GizmoHandleId::Plane(Axis::Z));
        assert_eq!(handles[6].id, GizmoHandleId::Ball);
    }

    #[test]
    fn translate_handles_have_triangle_geometry() {
        for handle in build_translate_handles(1.0) {
            assert!(!handle.mesh.vertices().is_empty());
            assert!(handle.mesh.has_primitive_type(PrimitiveType::TriangleList));
        }
    }

    #[test]
    fn scale_handles_have_axis_plane_and_ball() {
        let handles = build_scale_handles(1.0);
        // 3 axis arms + 3 plane handles + 1 center ball.
        assert_eq!(handles.len(), 7);
        assert_eq!(handles[6].id, GizmoHandleId::Ball);
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
        assert_eq!(build_handles(GizmoType::Translate, 1.0).len(), 7);
        assert_eq!(build_handles(GizmoType::Rotate, 1.0).len(), 3);
        assert_eq!(build_handles(GizmoType::Scale, 1.0).len(), 7);
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
                handle.id,
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
                handle.id,
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
                handle.id,
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
        use crate::scene::{FaceMaterial, Mesh, PrimitiveType, SceneData};
        use duck_engine_common::{Point3, Vector3};

        let pivot = Point3::new(5.0, 3.0, 0.0);

        let mut scene = SceneData::new();

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

        let scene = Scene::new(scene);

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
                handle.id, pivot,
            );
        }
    }

    /// Test that picking works after gizmo position is updated
    /// (simulates update_position path from sync_gizmo).
    #[test]
    fn translate_handles_pickable_after_position_update() {
        use crate::common::Ray;
        use crate::geom_query::pick_all_from_ray;
        use crate::scene::SceneData;
        use duck_engine_common::{Point3, Vector3};

        let initial_pivot = Point3::new(0.0, 0.0, 0.0);
        let new_pivot = Point3::new(10.0, 5.0, 3.0);

        let mut scene = SceneData::new();
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

        let scene = Scene::new(scene);

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
                handle.id, new_pivot,
            );
        }
    }
}
