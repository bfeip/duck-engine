use duck_engine_common::{transform_point, InnerSpace, Matrix4, Point3};

use crate::common::{Aabb, Ray};
use crate::{InstanceId, Mesh, NodeId, Scene, SubGeometryKind};

use super::mesh_intersection;
use super::pick_query::{pick_all, pick_all_with_view, PickQuery, PickView};

/// The primitive that was hit by a ray pick query.
#[derive(Debug, Clone)]
pub enum RayHit {
    /// A triangle face was hit.
    Triangle {
        /// Index of the triangle (0-based, into the mesh's triangle list)
        triangle_index: usize,
        /// Barycentric coordinates of the hit point (u, v, w) where w = 1 - u - v
        barycentric: (f32, f32, f32),
    },
    /// A line segment was hit within the query's tolerance.
    Segment {
        /// Index of the segment (0-based pair index into the mesh's line index buffer)
        segment_index: usize,
        /// World-space perpendicular distance from the ray to the segment
        distance_to_ray: f32,
    },
    /// A point was hit within the query's tolerance.
    Point {
        /// Index of the point (0-based index into the mesh's point list)
        point_index: usize,
        /// World-space distance from the ray to the point
        distance_to_ray: f32,
    },
}

impl RayHit {
    /// The sub-geometry kind of the hit primitive.
    pub fn kind(&self) -> SubGeometryKind {
        match self {
            RayHit::Triangle { .. } => SubGeometryKind::Face,
            RayHit::Segment { .. } => SubGeometryKind::Edge,
            RayHit::Point { .. } => SubGeometryKind::Pointset,
        }
    }

    /// The 0-based index of the hit primitive within its primitive list.
    pub fn primitive_index(&self) -> u32 {
        match self {
            RayHit::Triangle { triangle_index, .. } => *triangle_index as u32,
            RayHit::Segment { segment_index, .. } => *segment_index as u32,
            RayHit::Point { point_index, .. } => *point_index as u32,
        }
    }
}

/// Result of a ray pick query against a scene instance.
///
/// For `RayHit::Triangle`, `hit_point` is the exact ray-triangle intersection point.
/// For `RayHit::Segment`, `hit_point` is the closest point on the segment to the ray.
/// For `RayHit::Point`, `hit_point` is the point's position.
#[derive(Debug, Clone)]
pub struct RayPickResult {
    /// The node that was hit
    pub node_id: NodeId,
    /// The instance that was hit
    pub instance_id: InstanceId,
    /// World-space distance from the ray origin to the hit (used for depth sorting)
    pub distance: f32,
    /// World-space hit location
    pub hit_point: Point3,
    /// Which primitive was hit and its geometry-specific data
    pub hit: RayHit,
}

/// Ray picking query that implements the generic PickQuery trait.
///
/// - Construct with:
/// - [`RayPickQuery::faces`] for face-only picking.
/// - [`RayPickQuery::lines`] for line-only picking.
/// - [`RayPickQuery::points`] for point-only picking.
/// - [`RayPickQuery::all`] to pick all primitive types.
/// [`RayPickQuery::for_kinds`] allows an arbitrary combination.
pub struct RayPickQuery {
    /// The ray in current coordinate space (may be transformed to local space)
    ray: Ray,
    /// The original ray in world space (for distance calculations)
    world_ray: Ray,
    pick_faces: bool,
    pick_lines: bool,
    pick_points: bool,
    /// World-space tolerance for line and point picking
    line_tolerance: f32,
    /// Tolerance scaled to the current (possibly local) coordinate space
    local_line_tolerance: f32,
}

impl RayPickQuery {
    /// Creates a query picking an arbitrary combination of primitive types.
    ///
    /// `tolerance` is the maximum world-space distance between the ray and a
    /// segment/point for it to be considered a hit (ignored for faces).
    pub fn for_kinds(
        ray: Ray,
        tolerance: f32,
        pick_faces: bool,
        pick_lines: bool,
        pick_points: bool,
    ) -> Self {
        Self {
            ray,
            world_ray: ray,
            pick_faces,
            pick_lines,
            pick_points,
            line_tolerance: tolerance,
            local_line_tolerance: tolerance,
        }
    }

    /// Creates a face-only query. Lines and points are ignored.
    pub fn faces(ray: Ray) -> Self {
        Self::for_kinds(ray, 0.0, true, false, false)
    }

    /// Creates a line-only query. Triangles and points are ignored.
    ///
    /// `tolerance` is the maximum world-space distance between the ray and a segment
    /// for the segment to be considered a hit.
    pub fn lines(ray: Ray, tolerance: f32) -> Self {
        Self::for_kinds(ray, tolerance, false, true, false)
    }

    /// Creates a point-only query. Triangles and lines are ignored.
    ///
    /// `tolerance` is the maximum world-space distance between the ray and a point
    /// for the point to be considered a hit.
    pub fn points(ray: Ray, tolerance: f32) -> Self {
        Self::for_kinds(ray, tolerance, false, false, true)
    }

    /// Creates a query that picks triangle faces, line segments, and points.
    ///
    /// `tolerance` is the maximum world-space distance for segment and point hits.
    pub fn all(ray: Ray, tolerance: f32) -> Self {
        Self::for_kinds(ray, tolerance, true, true, true)
    }
}

impl PickQuery for RayPickQuery {
    type Result = RayPickResult;

    fn might_intersect_bounds(&self, bounds: &Aabb) -> bool {
        if self.pick_faces && bounds.intersects_ray(&self.ray).is_some() {
            return true;
        }
        if self.pick_lines || self.pick_points {
            // Inflate bounds by the tolerance so segments/points near-but-outside
            // the AABB are not incorrectly culled during the broad phase.
            let t = self.local_line_tolerance;
            let inflated = Aabb::new(
                Point3::new(bounds.min.x - t, bounds.min.y - t, bounds.min.z - t),
                Point3::new(bounds.max.x + t, bounds.max.y + t, bounds.max.z + t),
            );
            return inflated.intersects_ray(&self.ray).is_some();
        }
        false
    }

    fn transform(&self, matrix: &Matrix4) -> Self {
        // Scale the line tolerance into the local coordinate space.
        // world_to_local (= matrix) embeds the inverse of the world scale, so a
        // world-space distance d maps to d * column_magnitude in local space.
        // We take the max column magnitude to stay conservative for non-uniform scale.
        let scale = [matrix.x, matrix.y, matrix.z]
            .iter()
            .map(|col| col.truncate().magnitude())
            .fold(0.0_f32, f32::max);

        Self {
            ray: self.ray.transform(matrix),
            world_ray: self.world_ray,
            pick_faces: self.pick_faces,
            pick_lines: self.pick_lines,
            pick_points: self.pick_points,
            line_tolerance: self.line_tolerance,
            local_line_tolerance: self.line_tolerance * scale,
        }
    }

    fn collect_mesh_hits(
        &self,
        mesh: &Mesh,
        node_id: NodeId,
        instance_id: InstanceId,
        world_transform: &Matrix4,
        results: &mut Vec<Self::Result>,
    ) {
        if self.pick_faces {
            for mesh_hit in mesh_intersection::intersect_ray(mesh, &self.ray) {
                let world_hit_point =
                    transform_point(world_transform, mesh_hit.hit_point);
                let distance = (world_hit_point - self.world_ray.origin).magnitude();
                results.push(RayPickResult {
                    node_id,
                    instance_id,
                    distance,
                    hit_point: world_hit_point,
                    hit: RayHit::Triangle {
                        triangle_index: mesh_hit.triangle_index,
                        barycentric: mesh_hit.barycentric,
                    },
                });
            }
        }

        if self.pick_lines {
            for line_hit in
                mesh_intersection::intersect_ray_with_lines(mesh, &self.ray, self.local_line_tolerance)
            {
                let world_closest =
                    transform_point(world_transform, line_hit.closest_point);
                let distance = (world_closest - self.world_ray.origin).magnitude();

                // Compute world-space perpendicular distance by comparing the closest
                // point on the segment against the local-space ray point at the same t,
                // then transforming both to world space.
                let local_ray_point = self.ray.point_at(line_hit.t);
                let world_ray_point =
                    transform_point(world_transform, local_ray_point);
                let distance_to_ray = (world_closest - world_ray_point).magnitude();

                results.push(RayPickResult {
                    node_id,
                    instance_id,
                    distance,
                    hit_point: world_closest,
                    hit: RayHit::Segment {
                        segment_index: line_hit.segment_index,
                        distance_to_ray,
                    },
                });
            }
        }

        if self.pick_points {
            for point_hit in
                mesh_intersection::intersect_ray_with_points(mesh, &self.ray, self.local_line_tolerance)
            {
                let world_point =
                    transform_point(world_transform, point_hit.closest_point);
                let distance = (world_point - self.world_ray.origin).magnitude();

                // World-space distance to the ray: compare the point against the
                // local-space ray point at the same t, both transformed to world.
                let local_ray_point = self.ray.point_at(point_hit.t);
                let world_ray_point =
                    transform_point(world_transform, local_ray_point);
                let distance_to_ray = (world_point - world_ray_point).magnitude();

                results.push(RayPickResult {
                    node_id,
                    instance_id,
                    distance,
                    hit_point: world_point,
                    hit: RayHit::Point {
                        point_index: point_hit.point_index,
                        distance_to_ray,
                    },
                });
            }
        }
    }
}

/// Picks all instances intersected by a ray, sorted by distance (closest first).
///
/// Use [`RayPickQuery::faces`] to pick faces only, [`RayPickQuery::lines`] /
/// [`RayPickQuery::points`] to pick segments/points only, or [`RayPickQuery::all`]
/// to pick all primitive types.
pub fn pick_all_from_ray(query: &RayPickQuery, scene: &Scene) -> Vec<RayPickResult> {
    sort_by_distance(pick_all(query, scene))
}

/// Like [`pick_all_from_ray`], but resolves screen-space nodes against `view` so
/// picking matches the rendered placement of screen-sized / screen-facing
/// geometry (e.g. constant-pixel-size gizmo handles).
pub fn pick_all_from_ray_with_view(
    query: &RayPickQuery,
    scene: &Scene,
    view: Option<&PickView>,
) -> Vec<RayPickResult> {
    sort_by_distance(pick_all_with_view(query, scene, view))
}

/// Sorts pick results by world-space distance, closest first.
fn sort_by_distance(mut results: Vec<RayPickResult>) -> Vec<RayPickResult> {
    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_common::{InnerSpace, Vector3};

    use crate::common::Transform;
    use crate::geom_query::PickView;
    use crate::{
        DisplayBehavior, FaceMaterial, Instance, Mesh, NodeFlags, PositionedCamera, PrimitiveType,
        RenderLayer, Scene,
    };

    fn camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        }
    }

    /// A screen-sized node is drawn (and must be picked) larger than its authored
    /// geometry when the camera is close. A ray aimed between the authored and
    /// on-screen extents hits only when the pick is view-aware.
    #[test]
    fn screen_sized_node_picked_at_on_screen_extent() {
        let mut scene = Scene::new();
        // Unit cube at the origin: authored half-extent 0.5.
        let mesh_id = scene.add_mesh(Mesh::cube(1.0, PrimitiveType::TriangleList));
        let mat_id = scene.add_face_material(FaceMaterial::new());
        let node = scene
            .add_instance_node(
                None,
                Instance::new(mesh_id).with_face_material(mat_id),
                None,
                Transform::IDENTITY,
                NodeFlags::NONE,
            )
            .unwrap();
        let display = DisplayBehavior {
            screen_size: Some(200.0),
            layer: RenderLayer::Overlay,
            ..Default::default()
        };
        scene.set_node_display(node, display);
        let _ = scene.bounding();

        let cam = camera();
        let viewport = (256, 256);

        // On-screen uniform scale for this node (identity world transform).
        let s = display
            .effective_transform(duck_engine_common::Matrix4::from_scale(1.0), &cam, viewport)
            .x
            .truncate()
            .magnitude();
        assert!(s > 1.0, "expected the node to render larger than authored (s = {s})");

        // Aim between the authored half-extent (0.5) and the on-screen half (0.5*s).
        let x = 0.5 * (0.5 + 0.5 * s);
        assert!(x > 0.5 && x < 0.5 * s);
        let ray = Ray::new(Point3::new(x, 0.0, 5.0), Vector3::new(0.0, 0.0, -1.0));
        let query = RayPickQuery::faces(ray);

        // View-aware pick honors the on-screen size and hits.
        let view = PickView { camera: &cam, viewport };
        let with_view = pick_all_from_ray_with_view(&query, &scene, Some(&view));
        assert!(
            with_view.iter().any(|r| r.node_id == node),
            "view-aware pick should hit the screen-sized node",
        );

        // Authored-size pick tests the unit cube and misses.
        let without_view = pick_all_from_ray(&query, &scene);
        assert!(
            !without_view.iter().any(|r| r.node_id == node),
            "authored-size pick should miss (ray is outside the unit cube)",
        );
    }
}
