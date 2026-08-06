//! The built-in [`SnapProvider`]s. Each is a stateless strategy; all live
//! context arrives through [`SnapInput`] and the [`SceneData`].

use duck_engine_viewer::common::{
    transform_normal, transform_point, Aabb, EuclideanSpace, InnerSpace, Matrix4, Point3, Ray,
    Vector3,
};
use duck_engine_viewer::scene::geom_query::{pick_all, PickQuery};
use duck_engine_viewer::scene::camera::PositionedCamera;
use duck_engine_viewer::scene::resource::{
    InstanceId, Mesh, Node, NodeId, PrimitiveType, Topology, Visibility,
};
use duck_engine_viewer::scene::Scene;

use super::{Snap, SnapInput, SnapKind, SnapFlags, SnapProvider, SnapSettings};

/// Free point on the active construction plane. The always-present, lowest-tier
/// fallback so a plane-constrained operator always gets a position.
pub(crate) struct ConstructionPlaneSnap;

impl SnapProvider for ConstructionPlaneSnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::CONSTRUCTION_PLANE
    }

    fn collect(&self, input: &SnapInput, _scene: &Scene, _settings: &SnapSettings) -> Vec<Snap> {
        if let Some((_, position)) = input.ray.intersect_plane(input.plane) {
            vec![Snap {
                position,
                direction: None,
                kind: SnapKind::ConstructionPlane,
            }]
        } else {
            Vec::new()
        }
    }
}

/// The construction-frame origin. A single high-tier point; ranking decides
/// whether it is near enough to the cursor to win.
pub(crate) struct OriginSnap;

impl SnapProvider for OriginSnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::ORIGIN
    }

    fn collect(&self, _input: &SnapInput, _scene: &Scene, _settings: &SnapSettings) -> Vec<Snap> {
         return vec![Snap {
            position: Point3::origin(),
            direction: None,
            kind: SnapKind::Origin,
        }];
    }
}

/// The start vertex of an in-progress wire, offered so a path can be closed
/// back onto itself. Transient: constructed per cursor move with the live start
/// point, then passed to [`SnapEngine::snap`](super::SnapEngine::snap) as an
/// extra provider.
pub(crate) struct WireStartSnap {
    pub start: Point3,
}

impl SnapProvider for WireStartSnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::WIRE_START
    }

    fn collect(&self, _input: &SnapInput, _scene: &Scene, _settings: &SnapSettings) -> Vec<Snap> {
        vec![Snap {
            position: self.start,
            direction: None,
            kind: SnapKind::WireStart,
        }]
    }
}

/// The two construction-frame snaps the grid visualizes: the principal axes
/// (U, V, N — higher tier) and gridline-intersection guide points (lower tier).
pub(crate) struct GridSnap;

impl SnapProvider for GridSnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::GRID_AXIS | SnapFlags::GRID_GUIDE
    }

    fn collect(&self, input: &SnapInput, _scene: &Scene, _settings: &SnapSettings) -> Vec<Snap> {
        let (u, v) = input.plane.basis();
        let origin = input.plane.project_point(Point3::origin());
        let n = input.plane.normal.normalize();
        // Treat the axes as long segments spanning the visible grid.
        let half = (input.grid.size * 0.5).max(input.grid.minor_spacing);

        let mut out = Vec::new();

        // Principal axes through the origin: snap to the closest point on each
        // axis line to the cursor ray (the same primitive edge snapping reuses).
        for dir in [u, v, n] {
            if let Some(approach) = input
                .ray
                .closest_approach_to_segment(origin - dir * half, origin + dir * half)
            {
                out.push(Snap {
                    position: approach.closest_on_segment,
                    direction: Some(dir),
                    kind: SnapKind::GridAxis,
                });
            }
        }

        // Guide point: round the in-plane ray hit to the nearest gridline
        // intersection.
        if let Some((_, hit)) = input.ray.intersect_plane(input.plane) {
            let spacing = input.grid.minor_spacing.max(f32::EPSILON);
            let rel = hit - origin;
            let cu = (rel.dot(u) / spacing).round() * spacing;
            let cv = (rel.dot(v) / spacing).round() * spacing;
            out.push(Snap {
                position: origin + u * cu + v * cv,
                direction: None,
                kind: SnapKind::GridGuide,
            });
        };

        return out;
    }
}

/// Snaps to existing geometry: B-rep corners (vertices, [`SnapKind::Corner`]),
/// points along edges/wires ([`SnapKind::Edge`]), and ray–surface hits on faces
/// ([`SnapKind::Face`]). All kinds share one scene walk, the expensive part,
/// so they cost a single traversal.
///
/// The walk is delegated to the scene's [`PickQuery`] framework ([`pick_all`]),
/// which traverses the node tree from the roots, prunes `DO_NOT_SELECT` subtrees,
/// and culls whole nodes by their cached world-space AABB before any per-primitive
/// work — so meshes nowhere near the cursor are skipped entirely.
pub(crate) struct GeometrySnap;

impl SnapProvider for GeometrySnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::CORNER | SnapFlags::EDGE | SnapFlags::FACE
    }

    fn collect(&self, input: &SnapInput, scene: &Scene, settings: &SnapSettings) -> Vec<Snap> {
        let query = GeometrySnapQuery {
            ray: input.ray,
            camera: input.camera,
            cursor: input.cursor,
            viewport: input.viewport,
            pixel_tolerance: settings.pixel_tolerance,
            local_scale: 1.0,
            exclude_nodes: input.exclude_nodes,
            // Only do the work for kinds the caller actually asked for; the engine
            // still drops anything globally disabled afterwards.
            want_corners: input.requested.contains(SnapFlags::CORNER),
            want_edges: input.requested.contains(SnapFlags::EDGE),
            want_faces: input.requested.contains(SnapFlags::FACE),
        };

        let mut snaps = pick_all(&query, scene);

        // Every face hit lies on the cursor ray, so screen-space ranking cannot
        // tie-break between overlapping shapes: keep only the hit nearest along
        // the ray (the visible surface).
        let mut nearest_face: Option<Snap> = None;
        let mut nearest_d2 = f32::INFINITY;
        snaps.retain(|s| {
            if s.kind != SnapKind::Face {
                return true;
            }
            let d2 = (s.position - input.ray.origin).magnitude2();
            if d2 < nearest_d2 {
                nearest_d2 = d2;
                nearest_face = Some(*s);
            }
            false
        });
        snaps.extend(nearest_face);

        // Collapse coincident corners (shared B-rep vertices emitted by the same
        // mesh). Guard on kind so an edge candidate is never merged into a corner.
        const EPSILON: f32 = 1e-10;
        snaps.dedup_by(|a: &mut Snap, b: &mut Snap| {
            a.kind == SnapKind::Corner
                && b.kind == SnapKind::Corner
                && (a.position - b.position).magnitude() < EPSILON
        });
        snaps
    }
}

/// The [`PickQuery`] that drives [`GeometrySnap`]'s single scene walk. Broad phase
/// is a screen-space test (the same metric the ranking uses); the narrow phase runs
/// in local space and transforms results back to world.
#[derive(Clone, Copy)]
struct GeometrySnapQuery<'a> {
    /// Cursor ray: world space at broad phase, local space after [`Self::transform`].
    ray: Ray,
    camera: &'a PositionedCamera,
    cursor: (f32, f32),
    viewport: (u32, u32),
    pixel_tolerance: f32,
    /// Factor mapping world-space distances into the current (possibly local)
    /// coordinate space; updated alongside the ray by [`Self::transform`].
    local_scale: f32,
    exclude_nodes: &'a [NodeId],
    want_corners: bool,
    want_edges: bool,
    want_faces: bool,
}

impl GeometrySnapQuery<'_> {
    /// Conservative local-space distance bound for edge candidates: a segment
    /// farther than this from the ray projects to more than `pixel_tolerance`
    /// pixels from the cursor, so `rank()` could never accept it.
    ///
    /// Every segment lies inside the mesh bounds, so its depth is at most the
    /// farthest corner's, and `world_size_per_pixel` is non-decreasing in depth;
    /// the 2× margin covers off-axis perspective distortion.
    fn local_edge_tolerance(&self, mesh: &Mesh, world_transform: &Matrix4) -> f32 {
        let Some(bounds) = mesh.bounding() else {
            return 0.0;
        };
        let forward = self.camera.forward();
        let mut far_depth = self.camera.znear;
        for corner in bounds.transform(world_transform).corners() {
            far_depth = far_depth.max((corner - self.camera.eye).dot(forward));
        }
        let world_tolerance = self.pixel_tolerance
            * self.camera.world_size_per_pixel(far_depth, self.viewport.1)
            * 2.0;
        world_tolerance * self.local_scale
    }
}

impl PickQuery for GeometrySnapQuery<'_> {
    type Result = Snap;

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
            min_x = min_x.min(s.x);
            min_y = min_y.min(s.y);
            max_x = max_x.max(s.x);
            max_y = max_y.max(s.y);
        }

        let tol = self.pixel_tolerance;
        let (cx, cy) = self.cursor;
        cx >= min_x - tol && cx <= max_x + tol && cy >= min_y - tol && cy <= max_y + tol
    }

    fn transform(&self, matrix: &Matrix4) -> Self {
        // world_to_local (= matrix) embeds the inverse of the world scale, so a
        // world-space distance d maps to d * column_magnitude in local space.
        // The max column magnitude stays conservative under non-uniform scale.
        let scale = [matrix.x, matrix.y, matrix.z]
            .iter()
            .map(|col| col.truncate().magnitude())
            .fold(0.0_f32, f32::max);

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
        results: &mut Vec<Snap>,
    ) {
        // Per-node filters the broad phase doesn't apply. Visibility is checked per
        // node (not propagated to descendants), matching the previous behavior.
        if self.exclude_nodes.contains(&node.id) {
            return;
        }
        if node.visibility() != Visibility::Visible {
            return;
        }

        if self.want_corners {
            if let Some(topology) = mesh.topology() {
                collect_mesh_corners(mesh, topology, world_transform, results);
            }
        }

        if !self.want_faces && !self.want_edges {
            return;
        }
        // Both narrow phases run in this mesh's local space (the ray is already
        // there) through the mesh's cached spatial index, and lift results to
        // world.
        let index = mesh.spatial_index();

        if self.want_faces {
            // Nearest ray–triangle hit on this mesh; `GeometrySnap::collect`
            // picks the nearest across all meshes.
            if let Some(hit) = index.nearest_triangle(mesh, &self.ray) {
                if let Some([v0, v1, v2]) = mesh.triangle(hit.triangle_index) {
                    // Möller–Trumbore weights: hit = w·v0 + u·v1 + v·v2.
                    let (u, v, w) = hit.barycentric;
                    let mut normal = Vector3::from(v0.normal) * w
                        + Vector3::from(v1.normal) * u
                        + Vector3::from(v2.normal) * v;
                    if normal.magnitude2() <= f32::EPSILON {
                        // Missing/cancelling vertex normals: use the flat one.
                        let p0 = Point3::from(v0.position);
                        normal = (Point3::from(v1.position) - p0)
                            .cross(Point3::from(v2.position) - p0);
                    }
                    results.push(Snap {
                        position: transform_point(world_transform, hit.hit_point),
                        direction: Some(transform_normal(world_transform, normal)),
                        kind: SnapKind::Face,
                    });
                }
            }
        }

        if self.want_edges {
            // Bounded by the pixel tolerance: any segment farther from the ray
            // can never rank, and the index skips subtrees outside the bound.
            let local_tolerance = self.local_edge_tolerance(mesh, world_transform);
            index.for_each_segment_within(mesh, &self.ray, local_tolerance, |i, approach| {
                let Some([v0, v1]) = mesh.segment(i) else { return };
                let w0 = transform_point(world_transform, Point3::from(v0.position));
                let w1 = transform_point(world_transform, Point3::from(v1.position));
                results.push(Snap {
                    position: transform_point(world_transform, approach.closest_on_segment),
                    direction: Some((w1 - w0).normalize()),
                    kind: SnapKind::Edge,
                });
            });
        }
    }
}

/// Extracts a mesh's B-rep corner vertices into `out` (in world space).
///
/// Prefers explicit point topology when present; otherwise derives corners from
/// edge-range endpoints — the first/last vertex of each edge is its B-rep vertex,
/// while intermediate curve-approximation points are not corners.
fn collect_mesh_corners(mesh: &Mesh, topology: &Topology, world: &Matrix4, out: &mut Vec<Snap>) {
    let vertices = mesh.vertices();
    let mut push_corner = |local: [f32; 3]| {
        let position = transform_point(world, Point3::new(local[0], local[1], local[2]));
        out.push(Snap {
            position,
            direction: None,
            kind: SnapKind::Corner,
        });
    };

    if !topology.pointset_ranges.is_empty() {
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
    use duck_engine_viewer::common::{Plane, Ray, Transform, Vector3};
    use duck_engine_viewer::scene::camera::PositionedCamera;
    use duck_engine_viewer::scene::resource::{
        Instance, Mesh, MeshPrimitive, NodeFlags, PrimitiveType, SubMeshRange, Topology, Vertex,
    };

    use crate::grid::GridConfig;

    const EPSILON: f32 = 1e-3;

    fn close(a: Point3, b: Point3) -> bool {
        (a - b).magnitude2() < EPSILON * EPSILON
    }

    /// A `SnapInput` with the ray and cursor supplied directly. Grid/wire providers
    /// ignore the cursor; `GeometrySnap` uses it (and the camera) to cull, so tests
    /// hitting geometry must pass the cursor where that geometry projects.
    fn input_with_ray<'a>(
        ray: Ray,
        cursor: (f32, f32),
        cam: &'a PositionedCamera,
        plane: &'a Plane,
        grid: &'a GridConfig,
    ) -> SnapInput<'a> {
        SnapInput {
            ray,
            cursor,
            viewport: (800, 600),
            camera: cam,
            plane,
            grid,
            requested: SnapFlags::all(),
            exclude_nodes: &[],
        }
    }

    /// Screen pixel a world point projects to under the test viewport (800×600).
    fn screen_of(cam: &PositionedCamera, p: Point3) -> (f32, f32) {
        let s = cam.project_point_screen(p, 800, 600);
        (s.x, s.y)
    }

    fn dummy_camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 10.0, 0.0),
            target: Point3::origin(),
            up: Vector3::new(0.0, 0.0, -1.0),
            aspect: 4.0 / 3.0,
            fovy: 45.0,
            znear: 0.01,
            zfar: 1000.0,
            ortho: false,
        }
    }

    #[test]
    fn grid_guide_rounds_to_nearest_intersection() {
        let cam = dummy_camera();
        let plane = Plane::xz(); // u = +X, v = +Z, origin at world origin
        let grid = GridConfig::default(); // minor_spacing = 5.0
        // Ray straight down hitting the XZ plane at (5.2, 0, -3.1).
        let ray = Ray::new(Point3::new(5.2, 10.0, -3.1), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, (0.0, 0.0), &cam, &plane, &grid);

        let snaps = GridSnap.collect(&inp, &Scene::default(), &SnapSettings::default());

        let guide = snaps
            .iter()
            .find(|s| s.kind == SnapKind::GridGuide)
            .expect("expected a grid guide candidate");
        // 5.2 → 5.0, −3.1 → −5.0
        assert!(close(guide.position, Point3::new(5.0, 0.0, -5.0)), "{:?}", guide.position);
    }

    #[test]
    fn grid_axis_snaps_to_principal_axis_line() {
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        // Vertical ray near the X axis: the X-axis closest point is directly
        // below the ray's x, on the axis (y = z = 0).
        let ray = Ray::new(Point3::new(7.0, 10.0, -1.5), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, (0.0, 0.0), &cam, &plane, &grid);

        let snaps = GridSnap.collect(&inp, &Scene::default(), &SnapSettings::default());

        // Three principal axes (U, V, N) should each yield a candidate.
        let axes: Vec<_> = snaps.iter().filter(|s| s.kind == SnapKind::GridAxis).collect();
        assert_eq!(axes.len(), 3);

        // The U (X) axis candidate sits at (7, 0, 0) with direction ±X.
        let x_axis = axes
            .iter()
            .find(|s| s.direction.map_or(false, |d| d.x.abs() > 0.9))
            .expect("expected an X-axis candidate");
        assert!(close(x_axis.position, Point3::new(7.0, 0.0, 0.0)), "{:?}", x_axis.position);
    }

    #[test]
    fn wire_start_emits_its_start_point() {
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let ray = Ray::new(Point3::new(0.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, (0.0, 0.0), &cam, &plane, &grid);

        let start = Point3::new(2.0, 0.0, -3.0);
        let snaps = WireStartSnap { start }.collect(&inp, &Scene::default(), &SnapSettings::default());

        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].kind, SnapKind::WireStart);
        assert!(close(snaps[0].position, start), "{:?}", snaps[0].position);
    }

    #[test]
    fn edge_snaps_to_closest_point_on_segment_in_world_space() {
        // One straight edge along +X in local space, on a node translated +2 in Z,
        // so the world segment runs (0,0,2) → (10,0,2).
        let mesh = Mesh::from_raw(
            vec![
                Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0; 3], normal: [0.0; 3] },
                Vertex { position: [10.0, 0.0, 0.0], tex_coords: [0.0; 3], normal: [0.0; 3] },
            ],
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::LineList,
                indices: vec![0, 1],
            }],
        );

        let scene = Scene::default();
        let mesh_id = scene.add_mesh(mesh);
        scene
            .add_instance_node(
                None,
                Instance::new(mesh_id),
                Some("edge".to_owned()),
                Transform::from_position(Point3::new(0.0, 0.0, 2.0)),
                NodeFlags::NONE,
            )
            .expect("instance node");

        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        // Ray straight down at x=3, z=2: closest point on the segment is (3,0,2).
        let hit = Point3::new(3.0, 0.0, 2.0);
        let ray = Ray::new(Point3::new(3.0, 10.0, 2.0), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, screen_of(&cam, hit), &cam, &plane, &grid);

        let snaps = GeometrySnap.collect(&inp, &scene, &SnapSettings::default());

        let edge = snaps
            .iter()
            .find(|s| s.kind == SnapKind::Edge)
            .expect("expected an edge candidate");
        assert!(close(edge.position, Point3::new(3.0, 0.0, 2.0)), "{:?}", edge.position);
        // Direction is the edge tangent (±X).
        assert!(edge.direction.map_or(false, |d| d.x.abs() > 0.9), "{:?}", edge.direction);
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

        let mut snaps = Vec::new();
        collect_mesh_corners(&mesh, &topology, &world, &mut snaps);

        assert_eq!(snaps.len(), 2);
        assert!(snaps.iter().any(|s| close(s.position, Point3::new(11.0, 2.0, 3.0))));
        assert!(snaps.iter().any(|s| close(s.position, Point3::new(14.0, 5.0, 6.0))));
        assert!(snaps.iter().all(|s| s.kind == SnapKind::Corner));
    }

    /// Adds a single-triangle node (vertices at `positions` with `normals`,
    /// translated by `offset`) to `scene`.
    fn add_triangle_node(
        scene: &Scene,
        positions: [[f32; 3]; 3],
        normals: [[f32; 3]; 3],
        offset: Vector3,
    ) {
        let vertices = positions
            .iter()
            .zip(normals.iter())
            .map(|(&position, &normal)| Vertex { position, tex_coords: [0.0; 3], normal })
            .collect();
        let mesh = Mesh::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices: vec![0, 1, 2],
            }],
        );
        let mesh_id = scene.add_mesh(mesh);
        scene
            .add_instance_node(
                None,
                Instance::new(mesh_id),
                Some("triangle".to_owned()),
                Transform::from_position(Point3::origin() + offset),
                NodeFlags::NONE,
            )
            .expect("instance node");
    }

    #[test]
    fn face_snaps_to_ray_hit_in_world_space() {
        // A triangle in the local XZ plane, on a node translated +2 in Z.
        let scene = Scene::default();
        add_triangle_node(
            &scene,
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
            [[0.0, 1.0, 0.0]; 3],
            Vector3::new(0.0, 0.0, 2.0),
        );

        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        // Ray straight down at (3, 4): local hit (3, 0, 2) is inside the triangle.
        let hit = Point3::new(3.0, 0.0, 4.0);
        let ray = Ray::new(Point3::new(3.0, 10.0, 4.0), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, screen_of(&cam, hit), &cam, &plane, &grid);

        let snaps = GeometrySnap.collect(&inp, &scene, &SnapSettings::default());

        let face = snaps
            .iter()
            .find(|s| s.kind == SnapKind::Face)
            .expect("expected a face candidate");
        assert!(close(face.position, hit), "{:?}", face.position);
        let normal = face.direction.expect("face snap carries the surface normal");
        assert!((normal - Vector3::new(0.0, 1.0, 0.0)).magnitude() < EPSILON, "{normal:?}");
    }

    #[test]
    fn face_snap_keeps_only_nearest_hit_along_ray() {
        // Two overlapping triangles: one at y=0, one at y=2. A downward ray from
        // y=10 sees the y=2 face first.
        let mut scene = Scene::default();
        let positions = [[-10.0, 0.0, -10.0], [10.0, 0.0, -10.0], [0.0, 0.0, 10.0]];
        let normals = [[0.0, 1.0, 0.0]; 3];
        add_triangle_node(&mut scene, positions, normals, Vector3::new(0.0, 0.0, 0.0));
        add_triangle_node(&mut scene, positions, normals, Vector3::new(0.0, 2.0, 0.0));

        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let ray = Ray::new(Point3::new(0.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, screen_of(&cam, Point3::origin()), &cam, &plane, &grid);

        let snaps = GeometrySnap.collect(&inp, &scene, &SnapSettings::default());

        let faces: Vec<_> = snaps.iter().filter(|s| s.kind == SnapKind::Face).collect();
        assert_eq!(faces.len(), 1, "only the nearest face hit survives");
        assert!(close(faces[0].position, Point3::new(0.0, 2.0, 0.0)), "{:?}", faces[0].position);
    }

    #[test]
    fn face_snap_interpolates_vertex_normals() {
        // Distinct vertex normals; the hit at local (2, 0, 2) has barycentric
        // weights (w, u, v) = (0.6, 0.2, 0.2), so the blend is (0.2, 0.6, 0.2).
        let mut scene = Scene::default();
        add_triangle_node(
            &mut scene,
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            Vector3::new(0.0, 0.0, 0.0),
        );

        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let hit = Point3::new(2.0, 0.0, 2.0);
        let ray = Ray::new(Point3::new(2.0, 10.0, 2.0), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, screen_of(&cam, hit), &cam, &plane, &grid);

        let snaps = GeometrySnap.collect(&inp, &scene, &SnapSettings::default());

        let face = snaps
            .iter()
            .find(|s| s.kind == SnapKind::Face)
            .expect("expected a face candidate");
        let normal = face.direction.expect("face snap carries the surface normal");
        let expected = Vector3::new(0.2, 0.6, 0.2).normalize();
        assert!((normal - expected).magnitude() < EPSILON, "{normal:?}");
    }

    /// One identity-placed part: a straight edge (0,0,0)→(10,0,0) with B-rep
    /// topology, so it yields both corner and edge snaps. Returns its node id.
    fn scene_with_edge() -> (Scene, NodeId) {
        let mut mesh = Mesh::from_raw(
            vec![
                Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0; 3], normal: [0.0; 3] },
                Vertex { position: [10.0, 0.0, 0.0], tex_coords: [0.0; 3], normal: [0.0; 3] },
            ],
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::LineList,
                indices: vec![0, 1],
            }],
        );
        mesh.set_topology(Topology {
            face_ranges: vec![],
            edge_ranges: vec![SubMeshRange { start: 0, count: 1 }],
            pointset_ranges: vec![],
        });
        let scene = Scene::default();
        let mesh_id = scene.add_mesh(mesh);
        let node = scene
            .add_instance_node(
                None,
                Instance::new(mesh_id),
                Some("edge".to_owned()),
                Transform::IDENTITY,
                NodeFlags::NONE,
            )
            .expect("instance node");
        let node = node.id();
        (scene, node)
    }

    #[test]
    fn far_cursor_culls_the_node() {
        let (scene, _node) = scene_with_edge();
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        // The edge projects around screen center; a top-left cursor is well outside
        // its (tolerance-inflated) screen bounds, so the broad phase must cull it.
        let ray = Ray::new(Point3::new(5.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, (0.0, 0.0), &cam, &plane, &grid);

        let snaps = GeometrySnap.collect(&inp, &scene, &SnapSettings::default());
        assert!(snaps.is_empty(), "far cursor should cull the node: {snaps:?}");
    }

    #[test]
    fn might_intersect_bounds_tests_screen_proximity() {
        let cam = dummy_camera();
        let query = GeometrySnapQuery {
            ray: Ray::new(Point3::new(0.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
            camera: &cam,
            cursor: screen_of(&cam, Point3::origin()),
            viewport: (800, 600),
            pixel_tolerance: 12.0,
            local_scale: 1.0,
            exclude_nodes: &[],
            want_corners: true,
            want_edges: true,
            want_faces: true,
        };

        // An AABB straddling the origin projects under the cursor → kept.
        let near = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        assert!(query.might_intersect_bounds(&near));

        // A small AABB off to the side projects far from the cursor → culled.
        let far = Aabb::new(Point3::new(7.8, -0.2, -0.2), Point3::new(8.2, 0.2, 0.2));
        assert!(!query.might_intersect_bounds(&far));
    }

    #[test]
    fn requested_flags_gate_which_kinds_are_produced() {
        let (scene, _node) = scene_with_edge();
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let mid = Point3::new(5.0, 0.0, 0.0);
        let ray = Ray::new(Point3::new(5.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let mut inp = input_with_ray(ray, screen_of(&cam, mid), &cam, &plane, &grid);

        inp.requested = SnapFlags::CORNER;
        let corners = GeometrySnap.collect(&inp, &scene, &SnapSettings::default());
        assert!(!corners.is_empty());
        assert!(corners.iter().all(|s| s.kind == SnapKind::Corner));

        inp.requested = SnapFlags::EDGE;
        let edges = GeometrySnap.collect(&inp, &scene, &SnapSettings::default());
        assert!(!edges.is_empty());
        assert!(edges.iter().all(|s| s.kind == SnapKind::Edge));
    }

    #[test]
    fn excluded_and_invisible_nodes_yield_no_snaps() {
        let (scene, node) = scene_with_edge();
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let mid = Point3::new(5.0, 0.0, 0.0);
        let ray = Ray::new(Point3::new(5.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let cursor = screen_of(&cam, mid);

        // Baseline: visible, not excluded → snaps present.
        let base = input_with_ray(ray, cursor, &cam, &plane, &grid);
        assert!(!GeometrySnap.collect(&base, &scene, &SnapSettings::default()).is_empty());

        // Excluded node → nothing.
        let excluded = [node];
        let mut inp = input_with_ray(ray, cursor, &cam, &plane, &grid);
        inp.exclude_nodes = &excluded;
        assert!(GeometrySnap.collect(&inp, &scene, &SnapSettings::default()).is_empty());

        // Invisible node → nothing (per-node visibility check in the narrow phase).
        scene.set_node_visibility(node, Visibility::Invisible);
        let inp = input_with_ray(ray, cursor, &cam, &plane, &grid);
        assert!(GeometrySnap.collect(&inp, &scene, &SnapSettings::default()).is_empty());
    }
}
