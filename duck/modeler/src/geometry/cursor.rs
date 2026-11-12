//! What a scene shows near the cursor: the face under it, and the segments
//! and corners of the meshes within its reach on screen.

use duck_engine_scene::common::{
    array_to_point3, array_to_vec3, transform_normal, transform_point, Aabb, InnerSpace, Matrix4,
    Point3, Ray, Real, Vector3,
};
use duck_engine_scene::geom_query::{pick_all, PickQuery, PickView};
use duck_engine_scene::resource::{
    InstanceId, Mesh, Node, NodeId, PrimitiveFlags, PrimitiveType, SubGeometryKind, Topology,
};
use duck_engine_scene::{PositionedCamera, Scene};

/// The cursor over a view.
pub struct Cursor<'a> {
    pub view: PickView<'a>,
    /// Where the cursor is in the viewport, in pixels.
    pub at: (f32, f32),
    /// The world ray through the cursor.
    pub ray: Ray,
    /// How far from the cursor geometry is still near it, in pixels.
    pub reach: f32,
}

/// What a mesh shows near the cursor.
pub struct NearMesh {
    pub node: NodeId,
    /// The primitive types the node draws.
    pub shown: PrimitiveFlags,
    /// Where the cursor ray first meets the mesh's triangles.
    pub face: Option<FaceHit>,
    /// The mesh's segments within reach of the cursor.
    pub segments: Vec<SegmentHit>,
    /// The mesh's corners, the vertices its topology names, near or far.
    pub corners: Vec<Point3>,
}

/// Where the cursor ray meets a mesh's triangles.
pub struct FaceHit {
    pub point: Point3,
    /// The unit surface normal there, blended from the triangle's vertices.
    pub normal: Vector3,
    /// The face the triangle is part of, as the mesh's topology numbers it.
    pub element: Option<u32>,
}

/// A segment of a mesh the cursor ray passes near.
pub struct SegmentHit {
    /// The point of the segment nearest the ray.
    pub point: Point3,
    /// The segment's unit direction.
    pub direction: Vector3,
    /// How far the ray passes from the segment, in the mesh's own space.
    pub distance: Real,
    /// The edge the segment is part of, as the mesh's topology numbers it.
    pub element: Option<u32>,
}

/// What each mesh of `scene` shows near `cursor`, of the primitives sought:
/// `TRIANGLES` for the face under the cursor, `LINES` for the segments within
/// its reach, `POINTS` for the mesh's corners. Every mesh whose bounds come
/// within reach is reported, but for the `exclude` nodes and the subtrees
/// [`pick_all`] skips; hidden primitives are not found.
pub fn near_cursor(scene: &Scene, cursor: &Cursor, seek: PrimitiveFlags, exclude: &[NodeId]) -> Vec<NearMesh> {
    let probe = Probe {
        ray: cursor.ray,
        camera: cursor.view.camera,
        viewport: cursor.view.viewport,
        at: cursor.at,
        reach: cursor.reach,
        local_scale: 1.0,
        seek,
        exclude,
    };
    pick_all(&probe, scene)
}

/// The [`PickQuery`] behind [`near_cursor`]. Its broad phase is a screen-space
/// test; its narrow phase runs in the mesh's space and answers in the world.
#[derive(Clone, Copy)]
struct Probe<'a> {
    /// Cursor ray: world space at broad phase, local space after [`Self::transform`].
    ray: Ray,
    camera: &'a PositionedCamera,
    viewport: (u32, u32),
    at: (f32, f32),
    reach: f32,
    /// Factor mapping world-space distances into the current (possibly local)
    /// coordinate space; updated alongside the ray by [`Self::transform`].
    local_scale: Real,
    seek: PrimitiveFlags,
    exclude: &'a [NodeId],
}

impl Probe<'_> {
    /// Conservative local-space distance bound for segments: one farther than
    /// this from the ray lies more than `reach` pixels from the cursor.
    ///
    /// Every segment lies inside the mesh bounds, so its depth is at most the
    /// farthest corner's, and `world_size_per_pixel` is non-decreasing in depth;
    /// the 2× margin covers off-axis perspective distortion.
    fn local_edge_tolerance(&self, mesh: &Mesh, world_transform: &Matrix4) -> Real {
        let Some(bounds) = mesh.bounding() else {
            return 0.0;
        };
        let forward = self.camera.forward();
        let mut far_depth = self.camera.projection.depth_range().0;
        for corner in bounds.transform(world_transform).corners() {
            far_depth = far_depth.max((corner - self.camera.eye).dot(forward));
        }
        let world_tolerance =
            self.reach as Real * self.camera.world_size_per_pixel(far_depth, self.viewport.1) * 2.0;
        world_tolerance * self.local_scale
    }

    /// Where the ray first meets `mesh`'s triangles.
    fn face_hit(&self, mesh: &Mesh, world_transform: &Matrix4) -> Option<FaceHit> {
        let hit = mesh.spatial_index().nearest_triangle(mesh, &self.ray)?;
        let [v0, v1, v2] = mesh.triangle(hit.triangle_index)?;
        // Möller–Trumbore weights: hit = w·v0 + u·v1 + v·v2.
        let (u, v, w) = hit.barycentric;
        let mut normal =
            array_to_vec3(v0.normal) * w + array_to_vec3(v1.normal) * u + array_to_vec3(v2.normal) * v;
        if normal.magnitude2() <= Real::EPSILON {
            // Missing/cancelling vertex normals: use the flat one.
            let p0 = array_to_point3(v0.position);
            normal = (array_to_point3(v1.position) - p0).cross(array_to_point3(v2.position) - p0);
        }
        Some(FaceHit {
            point: transform_point(world_transform, hit.hit_point),
            normal: transform_normal(world_transform, normal),
            element: mesh.sub_geometry_for_primitive(SubGeometryKind::Face, hit.triangle_index as u32),
        })
    }

    /// The segments of `mesh` within reach of the cursor.
    fn segments_near(&self, mesh: &Mesh, world_transform: &Matrix4) -> Vec<SegmentHit> {
        // The index skips subtrees outside the bound.
        let tolerance = self.local_edge_tolerance(mesh, world_transform);
        let mut segments = Vec::new();
        mesh.spatial_index().for_each_segment_within(mesh, &self.ray, tolerance, |i, approach| {
            let Some([v0, v1]) = mesh.segment(i) else { return };
            let w0 = transform_point(world_transform, array_to_point3(v0.position));
            let w1 = transform_point(world_transform, array_to_point3(v1.position));
            segments.push(SegmentHit {
                point: transform_point(world_transform, approach.closest_on_segment),
                direction: (w1 - w0).normalize(),
                distance: approach.distance,
                element: mesh.sub_geometry_for_primitive(SubGeometryKind::Edge, i as u32),
            });
        });
        segments
    }
}

impl PickQuery for Probe<'_> {
    type Result = NearMesh;

    fn might_intersect_bounds(&self, bounds: &Aabb) -> bool {
        let (w, h) = self.viewport;
        let forward = self.camera.forward();
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for corner in bounds.corners() {
            // A corner behind the camera has no meaningful screen projection, so
            // don't risk culling a node straddling the camera plane — keep it.
            if (corner - self.camera.eye).dot(forward) <= 0.0 {
                return true;
            }
            let s = self.camera.project_point_screen(corner, w, h);
            let (sx, sy) = (s.x as f32, s.y as f32);
            min_x = min_x.min(sx);
            min_y = min_y.min(sy);
            max_x = max_x.max(sx);
            max_y = max_y.max(sy);
        }

        let reach = self.reach;
        let (cx, cy) = self.at;
        cx >= min_x - reach && cx <= max_x + reach && cy >= min_y - reach && cy <= max_y + reach
    }

    fn transform(&self, matrix: &Matrix4) -> Self {
        // world_to_local (= matrix) embeds the inverse of the world scale, so a
        // world-space distance d maps to d * column_magnitude in local space.
        // The max column magnitude stays conservative under non-uniform scale.
        let scale = [matrix.x, matrix.y, matrix.z]
            .iter()
            .map(|col| col.truncate().magnitude())
            .fold(0.0, Real::max);

        Self {
            ray: self.ray.transform(matrix),
            local_scale: self.local_scale * scale,
            ..*self
        }
    }

    fn collect_mesh_hits(
        &self,
        mesh: &Mesh,
        node: &Node,
        _instance_id: InstanceId,
        world_transform: &Matrix4,
        shown: PrimitiveFlags,
        results: &mut Vec<NearMesh>,
    ) {
        // The one per-node filter the scene walk doesn't apply; it already
        // skips invisible subtrees.
        if self.exclude.contains(&node.id) {
            return;
        }
        let found = self.seek & shown;
        let mut near = NearMesh { node: node.id, shown, face: None, segments: Vec::new(), corners: Vec::new() };

        if self.seek.contains(PrimitiveFlags::POINTS) {
            if let Some(topology) = mesh.topology() {
                collect_mesh_corners(mesh, topology, shown, world_transform, &mut near.corners);
            }
        }
        // Both narrow phases run in this mesh's local space (the ray is already
        // there) through the mesh's cached spatial index, and lift results to
        // world.
        if found.contains(PrimitiveFlags::TRIANGLES) {
            near.face = self.face_hit(mesh, world_transform);
        }
        if found.contains(PrimitiveFlags::LINES) {
            near.segments = self.segments_near(mesh, world_transform);
        }
        results.push(near);
    }
}

/// Extracts a mesh's B-rep corner vertices into `out` (in world space), from
/// `shown` primitives only.
///
/// Prefers explicit point topology when present; otherwise derives corners from
/// edge-range endpoints — the first/last vertex of each edge is its B-rep vertex,
/// while intermediate curve-approximation points are not corners.
fn collect_mesh_corners(
    mesh: &Mesh,
    topology: &Topology,
    shown: PrimitiveFlags,
    world: &Matrix4,
    out: &mut Vec<Point3>,
) {
    let vertices = mesh.vertices();
    let mut push_corner = |local: [f32; 3]| out.push(transform_point(world, array_to_point3(local)));

    if !topology.pointset_ranges.is_empty() {
        if !shown.contains(PrimitiveFlags::POINTS) {
            return;
        }
        // TODO: Only works on first point primitive. Not a big deal right now but...
        if let Some(points) = mesh
            .primitives()
            .iter()
            .find(|p| p.primitive_type == PrimitiveType::PointList)
        {
            for range in &topology.pointset_ranges {
                for i in range.start..range.start + range.count {
                    if let Some(vtx) = points
                        .indices
                        .get(i as usize)
                        .and_then(|&idx| vertices.get(idx as usize))
                    {
                        push_corner(vtx.position);
                    }
                }
            }
        }
        return;
    }

    if !shown.contains(PrimitiveFlags::LINES) {
        return;
    }
    // TODO: Only works on first line primitive. Not a big deal right now but...
    let Some(lines) = mesh
        .primitives()
        .iter()
        .find(|p| p.primitive_type == PrimitiveType::LineList)
    else {
        return;
    };

    for range in &topology.edge_ranges {
        if range.count == 0 {
            continue;
        }
        // Edge ranges count segments; the LineList holds 2 indices per segment.
        let first = range.start as usize * 2;
        let last = (range.start + range.count - 1) as usize * 2 + 1;
        for slot in [first, last] {
            if let Some(vtx) = lines
                .indices
                .get(slot)
                .and_then(|&idx| vertices.get(idx as usize))
            {
                push_corner(vtx.position);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::common::{MetricSpace, Vector3};
    use duck_engine_scene::resource::{MeshPrimitive, SubMeshRange, Vertex};
    use duck_engine_scene::Projection;

    use crate::document::Document;
    use crate::testing::doc_with_box;

    const EPSILON: Real = 1e-3;

    fn close(a: Point3, b: Point3) -> bool {
        a.distance2(b) < EPSILON * EPSILON
    }

    /// A camera looking straight down −Y from (0, 10, 0), over an 800×600 view.
    fn camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 10.0, 0.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 0.0, -1.0),
            aspect: 4.0 / 3.0,
            projection: Projection::Perspective { fovy: 45.0, znear: 0.01, zfar: 1000.0 },
        }
    }

    /// The cursor over world point `over`, seen by `camera`, reaching 12 pixels.
    fn cursor_over(camera: &PositionedCamera, over: Point3) -> Cursor<'_> {
        let screen = camera.project_point_screen(over, 800, 600);
        let at = (screen.x as f32, screen.y as f32);
        Cursor {
            view: PickView { camera, viewport: (800, 600) },
            at,
            ray: camera.ray_from_screen_point(at.0, at.1, 800, 600),
            reach: 12.0,
        }
    }

    /// What the one mesh of [`doc_with_box`]'s document, a 2-unit box centred on
    /// the origin, shows of `seek` near the cursor over `over`.
    fn near_box(document: &Document, over: Point3, seek: PrimitiveFlags) -> NearMesh {
        let camera = camera();
        let mut near = near_cursor(document.scene(), &cursor_over(&camera, over), seek, &[]);
        assert_eq!(near.len(), 1, "the box is the scene's one mesh");
        near.remove(0)
    }

    #[test]
    fn the_face_under_the_cursor_is_named_with_its_point_and_normal() {
        let (document, node) = doc_with_box();
        let part = document.part_at(node).unwrap();

        let near = near_box(&document, Point3::new(0.3, 1.0, 0.2), PrimitiveFlags::TRIANGLES);

        let hit = near.face.expect("the cursor is over the top face");
        assert!(close(hit.point, Point3::new(0.3, 1.0, 0.2)), "{:?}", hit.point);
        assert!((hit.normal - Vector3::unit_y()).magnitude() < EPSILON, "{:?}", hit.normal);
        let face = part.face(hit.element.expect("a part's mesh numbers its faces")).unwrap();
        assert!((face.center_of_mass().y - 1.0).abs() < 1e-9, "not the top face");
        assert!(near.segments.is_empty() && near.corners.is_empty(), "only the face was sought");
    }

    /// Seen from above, the box's edges along x = 1 lie one behind the other;
    /// the rest are far across the screen.
    #[test]
    fn only_segments_within_reach_are_found_each_on_its_edge() {
        let (document, node) = doc_with_box();
        let part = document.part_at(node).unwrap();

        let near = near_box(&document, Point3::new(1.02, 1.0, 0.3), PrimitiveFlags::LINES);

        assert!(near.segments.iter().any(|segment| close(segment.point, Point3::new(1.0, 1.0, 0.3))), "the top edge is under the cursor");
        for segment in &near.segments {
            assert!((segment.point.x - 1.0).abs() < EPSILON, "{:?}", segment.point);
            assert!(segment.direction.z.abs() > 0.99, "{:?}", segment.direction);
            let edge = part.edge(segment.element.expect("a part's mesh numbers its edges")).unwrap().curve();
            let along_x1 = (edge.start_point().x - 1.0).abs() < 1e-9 && (edge.end_point().x - 1.0).abs() < 1e-9;
            assert!(along_x1 && (edge.start_point().y - edge.end_point().y).abs() < 1e-9, "not an edge along x = 1");
        }
        assert!(near.face.is_none(), "only segments were sought");
    }

    #[test]
    fn corners_are_reported_near_or_far() {
        let (document, _node) = doc_with_box();

        let near = near_box(&document, Point3::new(0.0, 1.0, 0.0), PrimitiveFlags::POINTS);

        assert_eq!(near.corners.len(), 8);
        assert!(near.corners.iter().all(|c| (c.x.abs() - 1.0).abs() < EPSILON && (c.y.abs() - 1.0).abs() < EPSILON));
    }

    #[test]
    fn a_mesh_within_reach_is_reported_even_with_nothing_sought() {
        let (document, node) = doc_with_box();

        let near = near_box(&document, Point3::new(0.0, 1.0, 0.0), PrimitiveFlags::empty());

        assert_eq!(near.node, node);
        assert_eq!(near.shown, PrimitiveFlags::all());
        assert!(near.face.is_none() && near.segments.is_empty() && near.corners.is_empty());
    }

    #[test]
    fn excluded_and_distant_meshes_are_passed_over() {
        let (document, node) = doc_with_box();
        let camera = camera();
        let over = cursor_over(&camera, Point3::new(0.0, 1.0, 0.0));
        assert!(near_cursor(document.scene(), &over, PrimitiveFlags::all(), &[node]).is_empty());

        let aside = cursor_over(&camera, Point3::new(4.0, 0.0, 3.0));
        assert!(near_cursor(document.scene(), &aside, PrimitiveFlags::all(), &[]).is_empty());
    }

    #[test]
    fn might_intersect_bounds_tests_screen_proximity() {
        let camera = camera();
        let cursor = cursor_over(&camera, Point3::new(0.0, 0.0, 0.0));
        let probe = Probe {
            ray: cursor.ray,
            camera: &camera,
            viewport: (800, 600),
            at: cursor.at,
            reach: 12.0,
            local_scale: 1.0,
            seek: PrimitiveFlags::all(),
            exclude: &[],
        };

        // An AABB straddling the origin projects under the cursor → kept.
        let near = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        assert!(probe.might_intersect_bounds(&near));

        // A small AABB off to the side projects far from the cursor → culled.
        let far = Aabb::new(Point3::new(7.8, -0.2, -0.2), Point3::new(8.2, 0.2, 0.2));
        assert!(!probe.might_intersect_bounds(&far));
    }

    #[test]
    fn corners_from_edge_endpoints_in_world_space() {
        // One straight edge A→B as a single LineList segment.
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mesh = Mesh::from_raw(
            vec![
                Vertex { position: a, tex_coords: [0.0; 3], normal: [0.0; 3] },
                Vertex { position: b, tex_coords: [0.0; 3], normal: [0.0; 3] },
            ],
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::LineList,
                indices: vec![0, 1],
            }],
        );
        let topology = Topology {
            face_ranges: vec![],
            edge_ranges: vec![SubMeshRange { start: 0, count: 1 }],
            pointset_ranges: vec![],
        };
        // Translate the whole part by +10 in X.
        let world = Matrix4::from_translation(Vector3::new(10.0, 0.0, 0.0));

        let mut corners = Vec::new();
        collect_mesh_corners(&mesh, &topology, PrimitiveFlags::all(), &world, &mut corners);

        assert_eq!(corners.len(), 2);
        assert!(corners.iter().any(|&corner| close(corner, Point3::new(11.0, 2.0, 3.0))));
        assert!(corners.iter().any(|&corner| close(corner, Point3::new(14.0, 5.0, 6.0))));
    }
}
