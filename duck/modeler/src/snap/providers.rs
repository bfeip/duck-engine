//! The built-in [`SnapProvider`]s. Each is a stateless strategy; all live
//! context arrives through [`SnapInput`] and the [`Document`].

use duck_engine_viewer::common::{EuclideanSpace, InnerSpace, Point3, Real};
use duck_engine_viewer::scene::geom_query::PickView;
use duck_engine_viewer::scene::resource::PrimitiveFlags;

use super::stroke::{edge_relations, face_relations, Stroke};
use super::{Candidate, Snap, SnapInput, SnapKind, SnapFlags, SnapProvider, SnapSettings};
use crate::document::{CadPart, Document};
use crate::geometry::{near_cursor, CurvePoint, Cursor, PartEdge, PartFace, SegmentHit, SurfacePoint};

/// Free point on the active construction plane. The always-present, lowest-tier
/// fallback so a plane-constrained operator always gets a position.
pub(crate) struct ConstructionPlaneSnap;

impl SnapProvider for ConstructionPlaneSnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::CONSTRUCTION_PLANE
    }

    fn collect(&self, input: &SnapInput, _document: &Document, _settings: &SnapSettings) -> Vec<Candidate> {
        match input.ray.intersect_plane(input.plane) {
            Some((_, position)) => vec![Snap::new(position, SnapKind::ConstructionPlane).into()],
            None => Vec::new(),
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

    fn collect(&self, _input: &SnapInput, _document: &Document, _settings: &SnapSettings) -> Vec<Candidate> {
        vec![Snap::new(Point3::origin(), SnapKind::Origin).into()]
    }
}

/// The two construction-frame snaps the grid visualizes: the principal axes
/// (U, V, N — higher tier) and gridline-intersection guide points (lower tier).
pub(crate) struct GridSnap;

impl SnapProvider for GridSnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::GRID_AXIS | SnapFlags::GRID_GUIDE
    }

    fn collect(&self, input: &SnapInput, _document: &Document, _settings: &SnapSettings) -> Vec<Candidate> {
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
                out.push(Snap::new(approach.closest_on_segment, SnapKind::GridAxis).with_direction(dir).into());
            }
        }

        // Guide point: round the in-plane ray hit to the nearest gridline
        // intersection.
        if let Some((_, hit)) = input.ray.intersect_plane(input.plane) {
            let spacing = input.grid.minor_spacing.max(Real::EPSILON);
            let rel = hit - origin;
            let cu = (rel.dot(u) / spacing).round() * spacing;
            let cv = (rel.dot(v) / spacing).round() * spacing;
            out.push(Snap::new(origin + u * cu + v * cv, SnapKind::GridGuide).into());
        };

        out
    }
}

/// Snaps to existing geometry: B-rep corners (vertices, [`SnapKind::Corner`]),
/// points along edges/wires ([`SnapKind::Edge`]), edge midpoints from the
/// part's B-rep ([`SnapKind::Midpoint`]), the centres of a part's circular
/// edges ([`SnapKind::Center`]), ray–surface hits on faces
/// ([`SnapKind::Face`]), and the centre of the flat face hit
/// ([`SnapKind::FaceCenter`]). With a stroke, the part edges near the cursor
/// and the face under it also relate it to the stroke's last point
/// ([`edge_relations`], [`face_relations`]). All kinds share one scene walk
/// ([`near_cursor`]), the expensive part, so they cost a single traversal.
///
/// Centres are offered for the parts the walk reaches near the cursor, so the
/// centre of an arc that lies off its part's bounds is missed.
pub(crate) struct GeometrySnap;

impl SnapProvider for GeometrySnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::CORNER
            | SnapFlags::EDGE
            | SnapFlags::MIDPOINT
            | SnapFlags::CENTER
            | SnapFlags::FACE
            | SnapFlags::FACE_CENTER
            | RELATION_KINDS
    }

    fn collect(&self, input: &SnapInput, document: &Document, settings: &SnapSettings) -> Vec<Candidate> {
        let mut kinds = input.requested & self.produces();
        let stroke = input.stroke.filter(|stroke| !stroke.is_empty());
        if stroke.is_none() {
            kinds -= RELATION_KINDS;
        }
        let cursor = Cursor {
            view: PickView { camera: input.camera, viewport: input.viewport },
            at: input.cursor,
            ray: input.ray,
            reach: settings.pixel_tolerance,
        };

        // Every face hit lies on the cursor ray, so screen-space ranking cannot
        // tie-break between overlapping shapes: keep only the hit nearest along
        // the ray (the visible surface), and offer only that face's centre and
        // relations.
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut nearest_face = None;
        let mut nearest_d2 = Real::INFINITY;
        for near in near_cursor(document.scene(), &cursor, sought(kinds), input.exclude_nodes) {
            let part = document.part_at(near.node);
            candidates.extend(near.corners.iter().map(|&corner| Candidate::from(Snap::new(corner, SnapKind::Corner))));
            if kinds.contains(SnapFlags::CENTER) && near.shown.contains(PrimitiveFlags::LINES) {
                if let Some(part) = part {
                    collect_circle_centers(part, &mut candidates);
                }
            }
            edge_snaps(&near.segments, part, kinds, stroke, document, &mut candidates);
            if let Some(hit) = near.face {
                let d2 = (hit.point - input.ray.origin).magnitude2();
                if d2 < nearest_d2 {
                    nearest_d2 = d2;
                    nearest_face = Some((near.node, hit));
                }
            }
        }
        if let Some((node, hit)) = nearest_face {
            let face = document.part_at(node).and_then(|part| PartFace::new(part, hit.element?).ok());
            let snap = Snap::new(hit.point, SnapKind::Face).with_direction(hit.normal);
            let snap = Snap { on: face.map(|face| face.element()), ..snap };
            if kinds.contains(SnapFlags::FACE) {
                candidates.push(snap.into());
            }
            if let Some(face) = face {
                if kinds.contains(SnapFlags::FACE_CENTER) {
                    candidates.extend(face_center(&face).map(Candidate::from));
                }
                if let Some(stroke) = stroke {
                    face_relations(&face, snap.position, stroke, kinds, &mut candidates);
                }
            }
        }

        // Collapse coincident corners (shared B-rep vertices emitted by the same
        // mesh). Guard on kind so an edge candidate is never merged into a corner.
        const EPSILON: Real = 1e-10;
        candidates.dedup_by(|a, b| {
            a.snap.kind == SnapKind::Corner
                && b.snap.kind == SnapKind::Corner
                && (a.snap.position - b.snap.position).magnitude() < EPSILON
        });
        candidates
    }
}

/// The kinds relating what is under the cursor to a stroke.
const RELATION_KINDS: SnapFlags =
    SnapFlags::CLOSEST.union(SnapFlags::PERPENDICULAR).union(SnapFlags::TANGENT_TANGENT);

/// The primitives `kinds` are found on: triangles where the cursor ray meets a
/// face, lines for the segments near it, points for corners.
fn sought(kinds: SnapFlags) -> PrimitiveFlags {
    let on_faces = SnapFlags::FACE | SnapFlags::FACE_CENTER | SnapFlags::CLOSEST | SnapFlags::PERPENDICULAR;
    let on_segments = SnapFlags::EDGE | SnapFlags::MIDPOINT | RELATION_KINDS;
    let mut sought = PrimitiveFlags::empty();
    sought.set(PrimitiveFlags::TRIANGLES, kinds.intersects(on_faces));
    sought.set(PrimitiveFlags::LINES, kinds.intersects(on_segments));
    sought.set(PrimitiveFlags::POINTS, kinds.contains(SnapFlags::CORNER));
    sought
}

/// Pushes into `out` the `kinds` of snap found on the `segments` near the
/// cursor on one mesh, `part`'s if it is a part's: onto its edges, their
/// midpoints, and with a `stroke`, what relates them to its last point.
fn edge_snaps(
    segments: &[SegmentHit],
    part: Option<&CadPart>,
    kinds: SnapFlags,
    stroke: Option<&Stroke>,
    document: &Document,
    out: &mut Vec<Candidate>,
) {
    let on_tessellation =
        |segment: &SegmentHit| Snap::new(segment.point, SnapKind::Edge).with_direction(segment.direction);
    // A part's edge is found once, from its segment nearest the ray, on its
    // B-rep; any other segment is snapped onto as it is.
    let mut nearest: Vec<(u32, &SegmentHit)> = Vec::new();
    for segment in segments {
        match segment.element.filter(|_| part.is_some()) {
            // A part's edge also relates to the stroke.
            Some(element) if kinds.intersects(SnapFlags::EDGE | RELATION_KINDS) => {
                match nearest.iter_mut().find(|(found, _)| *found == element) {
                    Some(best) if segment.distance < best.1.distance => best.1 = segment,
                    Some(_) => {}
                    None => nearest.push((element, segment)),
                }
            }
            None if kinds.contains(SnapFlags::EDGE) => out.push(on_tessellation(segment).into()),
            _ => {}
        }
    }
    let Some(part) = part else { return };

    for (index, segment) in nearest {
        let snap = on_tessellation(segment);
        let edge = match PartEdge::new(part, index) {
            Ok(edge) => edge,
            Err(e) => {
                log::debug!("Edge snap left on its tessellation: {e}");
                out.push(snap.into());
                continue;
            }
        };
        let snap = onto_edge(edge, snap);
        if kinds.contains(SnapFlags::EDGE) {
            out.push(snap.into());
        }
        if let Some(stroke) = stroke {
            edge_relations(&edge, snap.position, stroke, kinds, document, out);
        }
    }

    if kinds.contains(SnapFlags::MIDPOINT) {
        // An edge's midpoint lies on (or within tessellation deflection of) one
        // of its segments, so every edge whose midpoint could rank has a
        // segment here.
        let mut edges: Vec<u32> = segments.iter().filter_map(|segment| segment.element).collect();
        edges.sort_unstable();
        edges.dedup();
        collect_edge_midpoints(part, &edges, out);
    }
}

/// The centre of `face` if it is flat, along its outward normal.
fn face_center(face: &PartFace) -> Option<Snap> {
    if !face.is_flat() {
        return None;
    }
    match face.centroid() {
        Ok(SurfacePoint { point, normal }) => {
            Some(Snap::new(point, SnapKind::FaceCenter).with_direction(normal).with_on(face.element()))
        }
        Err(e) => {
            log::debug!("No centre snap on {face}: {e}");
            None
        }
    }
}

/// `snap`, made on the tessellation of `edge`, moved onto the edge's B-rep
/// curve and along its tangent there. It stays where it was if the edge has no
/// curve to project onto.
fn onto_edge(edge: PartEdge, snap: Snap) -> Snap {
    let snap = snap.with_on(edge.element());
    match edge.nearest(snap.position) {
        Ok(CurvePoint { point, tangent }) => Snap {
            position: point,
            direction: if tangent.magnitude2() > 0.0 { Some(tangent) } else { snap.direction },
            ..snap
        },
        Err(e) => {
            log::debug!("Edge snap on {edge} left on its tessellation: {e}");
            snap
        }
    }
}

/// Pushes the midpoint of each of `part`'s `edges` (element indices) into `out`,
/// taken on the B-rep curve, with the curve's tangent there.
fn collect_edge_midpoints(part: &CadPart, edges: &[u32], out: &mut Vec<Candidate>) {
    for &index in edges {
        let midpoint = PartEdge::new(part, index).and_then(|edge| Ok((edge, edge.midpoint()?)));
        match midpoint {
            Ok((edge, CurvePoint { point, tangent })) => {
                let midpoint = Snap::new(point, SnapKind::Midpoint).with_direction(tangent);
                out.push(midpoint.with_on(edge.element()).into());
            }
            Err(e) => log::debug!("No midpoint snap on edge {index} of {}: {e}", part.name),
        }
    }
}

/// Pushes the centre of each of `part`'s circular edges into `out`, along the
/// circle's axis.
fn collect_circle_centers(part: &CadPart, out: &mut Vec<Candidate>) {
    for index in 0..part.elements().edges().len() as u32 {
        if let Some(circle) = PartEdge::new(part, index).ok().and_then(|edge| edge.circle()) {
            out.push(Snap::new(circle.center, SnapKind::Center).with_direction(circle.normal).into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_viewer::common::{Plane, Ray, Transform, Vector3};
    use duck_engine_viewer::scene::cad::CadTessellationOptions;
    use duck_engine_viewer::scene::{PositionedCamera, Projection, Scene};
    use duck_engine_viewer::scene::resource::{
        Instance, Mesh, MeshPrimitive, NodeFlags, NodeId, PrimitiveType, SubMeshRange, Topology,
        Vertex, Visibility,
    };
    use glam::DVec3;
    use opencascade::primitives::Shape;

    use crate::grid::GridConfig;
    use crate::testing::{doc_with_box, doc_with_shape};

    const EPSILON: Real = 1e-3;

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
            stroke: None,
        }
    }

    /// The snaps `provider` proposes for `input`.
    fn proposed(provider: &dyn SnapProvider, input: &SnapInput, document: &Document) -> Vec<Snap> {
        let candidates = provider.collect(input, document, &SnapSettings::default());
        candidates.into_iter().map(|candidate| candidate.snap).collect()
    }

    /// Screen pixel a world point projects to under the test viewport (800×600).
    fn screen_of(cam: &PositionedCamera, p: Point3) -> (f32, f32) {
        let s = cam.project_point_screen(p, 800, 600);
        (s.x as f32, s.y as f32)
    }

    fn dummy_camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 10.0, 0.0),
            target: Point3::origin(),
            up: Vector3::new(0.0, 0.0, -1.0),
            aspect: 4.0 / 3.0,
            projection: Projection::Perspective { fovy: 45.0, znear: 0.01, zfar: 1000.0 },
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

        let snaps = proposed(&GridSnap, &inp, &Document::new(Scene::default()));

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

        let snaps = proposed(&GridSnap, &inp, &Document::new(Scene::default()));

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

        let document = Document::new(Scene::default());
        let scene = document.scene();
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

        let snaps = proposed(&GeometrySnap, &inp, &document);

        let edge = snaps
            .iter()
            .find(|s| s.kind == SnapKind::Edge)
            .expect("expected an edge candidate");
        assert!(close(edge.position, Point3::new(3.0, 0.0, 2.0)), "{:?}", edge.position);
        // Direction is the edge tangent (±X).
        assert!(edge.direction.map_or(false, |d| d.x.abs() > 0.9), "{:?}", edge.direction);
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
        let document = Document::new(Scene::default());
        add_triangle_node(
            document.scene(),
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

        let snaps = proposed(&GeometrySnap, &inp, &document);

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
        let document = Document::new(Scene::default());
        let positions = [[-10.0, 0.0, -10.0], [10.0, 0.0, -10.0], [0.0, 0.0, 10.0]];
        let normals = [[0.0, 1.0, 0.0]; 3];
        add_triangle_node(document.scene(), positions, normals, Vector3::new(0.0, 0.0, 0.0));
        add_triangle_node(document.scene(), positions, normals, Vector3::new(0.0, 2.0, 0.0));

        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let ray = Ray::new(Point3::new(0.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, screen_of(&cam, Point3::origin()), &cam, &plane, &grid);

        let snaps = proposed(&GeometrySnap, &inp, &document);

        let faces: Vec<_> = snaps.iter().filter(|s| s.kind == SnapKind::Face).collect();
        assert_eq!(faces.len(), 1, "only the nearest face hit survives");
        assert!(close(faces[0].position, Point3::new(0.0, 2.0, 0.0)), "{:?}", faces[0].position);
    }

    #[test]
    fn face_snap_interpolates_vertex_normals() {
        // Distinct vertex normals; the hit at local (2, 0, 2) has barycentric
        // weights (w, u, v) = (0.6, 0.2, 0.2), so the blend is (0.2, 0.6, 0.2).
        let document = Document::new(Scene::default());
        add_triangle_node(
            document.scene(),
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

        let snaps = proposed(&GeometrySnap, &inp, &document);

        let face = snaps
            .iter()
            .find(|s| s.kind == SnapKind::Face)
            .expect("expected a face candidate");
        let normal = face.direction.expect("face snap carries the surface normal");
        let expected = Vector3::new(0.2, 0.6, 0.2).normalize();
        assert!((normal - expected).magnitude() < EPSILON, "{normal:?}");
    }

    /// One identity-placed mesh, not a part: a straight edge (0,0,0)→(10,0,0)
    /// with B-rep topology, so it yields both corner and edge snaps. Returns its
    /// node id.
    fn scene_with_edge() -> (Document, NodeId) {
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
        let document = Document::new(Scene::default());
        let mesh_id = document.scene().add_mesh(mesh);
        let node = document
            .scene()
            .add_instance_node(
                None,
                Instance::new(mesh_id),
                Some("edge".to_owned()),
                Transform::IDENTITY,
                NodeFlags::NONE,
            )
            .expect("instance node");
        let node = node.id();
        (document, node)
    }

    #[test]
    fn far_cursor_culls_the_node() {
        let (document, _node) = scene_with_edge();
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        // The edge projects around screen center; a top-left cursor is well outside
        // its (tolerance-inflated) screen bounds, so the broad phase must cull it.
        let ray = Ray::new(Point3::new(5.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, (0.0, 0.0), &cam, &plane, &grid);

        let snaps = proposed(&GeometrySnap, &inp, &document);
        assert!(snaps.is_empty(), "far cursor should cull the node: {snaps:?}");
    }

    #[test]
    fn requested_flags_gate_which_kinds_are_produced() {
        let (document, _node) = scene_with_edge();
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let mid = Point3::new(5.0, 0.0, 0.0);
        let ray = Ray::new(Point3::new(5.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let mut inp = input_with_ray(ray, screen_of(&cam, mid), &cam, &plane, &grid);

        inp.requested = SnapFlags::CORNER;
        let corners = proposed(&GeometrySnap, &inp, &document);
        assert!(!corners.is_empty());
        assert!(corners.iter().all(|s| s.kind == SnapKind::Corner));

        inp.requested = SnapFlags::EDGE;
        let edges = proposed(&GeometrySnap, &inp, &document);
        assert!(!edges.is_empty());
        assert!(edges.iter().all(|s| s.kind == SnapKind::Edge));
    }

    #[test]
    fn excluded_and_invisible_nodes_yield_no_snaps() {
        let (document, node) = scene_with_edge();
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let mid = Point3::new(5.0, 0.0, 0.0);
        let ray = Ray::new(Point3::new(5.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let cursor = screen_of(&cam, mid);

        // Baseline: visible, not excluded → snaps present.
        let base = input_with_ray(ray, cursor, &cam, &plane, &grid);
        assert!(!proposed(&GeometrySnap, &base, &document).is_empty());

        // Excluded node → nothing.
        let excluded = [node];
        let mut inp = input_with_ray(ray, cursor, &cam, &plane, &grid);
        inp.exclude_nodes = &excluded;
        assert!(proposed(&GeometrySnap, &inp, &document).is_empty());

        // Invisible node → nothing (the scene walk skips hidden subtrees).
        document.scene().set_node_visibility(node, Visibility::Invisible);
        let inp = input_with_ray(ray, cursor, &cam, &plane, &grid);
        assert!(proposed(&GeometrySnap, &inp, &document).is_empty());
    }

    #[test]
    fn midpoint_snaps_to_the_middle_of_a_part_edge() {
        // A 2-unit box centred on the origin; the ray drops onto its top edge
        // x = 1, y = 1, off that edge's middle.
        let (document, _node) = doc_with_box();
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let on_edge = Point3::new(1.0, 1.0, 0.5);
        let ray = Ray::new(Point3::new(1.0, 10.0, 0.5), Vector3::new(0.0, -1.0, 0.0));
        let mut inp = input_with_ray(ray, screen_of(&cam, on_edge), &cam, &plane, &grid);
        inp.requested = SnapFlags::MIDPOINT;

        let snaps = proposed(&GeometrySnap, &inp, &document);

        assert!(snaps.iter().all(|s| s.kind == SnapKind::Midpoint), "{snaps:?}");
        let midpoint = snaps
            .iter()
            .find(|s| close(s.position, Point3::new(1.0, 1.0, 0.0)))
            .expect("expected the top edge's midpoint");
        // Direction is the edge tangent (±Z).
        assert!(midpoint.direction.map_or(false, |d| d.z.abs() > 0.9), "{:?}", midpoint.direction);
    }

    /// A chord of the tessellated rim sags inside the circle; the B-rep midpoint
    /// lies on it.
    #[test]
    fn midpoint_lies_on_a_curved_edge() {
        const ON_CURVE: Real = 1e-6;
        // A radius-2 cylinder standing on the XZ plane; the ray drops onto both
        // rims at x = 2.
        let (document, _node) = doc_with_shape(Shape::cylinder(DVec3::ZERO, 2.0, DVec3::Y, 1.0));
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let on_rim = Point3::new(2.0, 1.0, 0.0);
        let ray = Ray::new(Point3::new(2.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let mut inp = input_with_ray(ray, screen_of(&cam, on_rim), &cam, &plane, &grid);
        inp.requested = SnapFlags::MIDPOINT;

        let snaps = proposed(&GeometrySnap, &inp, &document);

        assert!(!snaps.is_empty(), "expected the rims' midpoints");
        for snap in &snaps {
            let radial = Vector3::new(snap.position.x, 0.0, snap.position.z);
            assert!((radial.magnitude() - 2.0).abs() < ON_CURVE, "off the rim: {:?}", snap.position);
            let tangent = snap.direction.expect("a midpoint carries the curve tangent");
            assert!(tangent.y.abs() < ON_CURVE && tangent.dot(radial).abs() < ON_CURVE, "{tangent:?}");
        }
    }

    /// The rim's tessellation cuts chords across the circle; an edge snap
    /// lies on the circle itself, along its tangent.
    #[test]
    fn edge_snaps_lie_on_a_curved_edge() {
        const ON_CURVE: Real = 1e-6;
        // A radius-2 cylinder standing on the XZ plane; the ray drops onto
        // both rims a radian round from the seam.
        let (document, _node) = doc_with_shape(Shape::cylinder(DVec3::ZERO, 2.0, DVec3::Y, 1.0));
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let (x, z) = (2.0 * (1.0 as Real).cos(), 2.0 * (1.0 as Real).sin());
        let ray = Ray::new(Point3::new(x, 10.0, z), Vector3::new(0.0, -1.0, 0.0));
        let mut inp = input_with_ray(ray, screen_of(&cam, Point3::new(x, 1.0, z)), &cam, &plane, &grid);
        inp.requested = SnapFlags::EDGE;

        let snaps = proposed(&GeometrySnap, &inp, &document);

        assert_eq!(snaps.len(), 2, "one snap per rim: {snaps:?}");
        for snap in &snaps {
            let radial = Vector3::new(snap.position.x, 0.0, snap.position.z);
            assert!((radial.magnitude() - 2.0).abs() < ON_CURVE, "off the rim: {:?}", snap.position);
            let tangent = snap.direction.expect("an edge snap carries the curve tangent");
            assert!(tangent.y.abs() < ON_CURVE && tangent.dot(radial).abs() < ON_CURVE, "{tangent:?}");
        }
    }

    /// A box at coordinates no `f32` holds: its tessellation is a few ulps
    /// off, but an edge snap lands on the B-rep.
    #[test]
    fn edge_snaps_land_on_the_b_rep_not_its_rounded_tessellation() {
        const EXACT: Real = 1e-9;
        let (x, y) = (100.3, 0.7);
        let box_shape = Shape::box_from_corners(DVec3::new(97.1, 0.0, -1.0), DVec3::new(x, y, 1.0));
        let (document, _node) = doc_with_shape(box_shape);
        let on_edge = Point3::new(x as Real, y as Real, 0.3);
        let cam = PositionedCamera {
            eye: Point3::new(x as Real, 10.0, 0.0),
            target: Point3::new(x as Real, 0.0, 0.0),
            ..dummy_camera()
        };
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let ray = Ray::new(on_edge + Vector3::new(0.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let mut inp = input_with_ray(ray, screen_of(&cam, on_edge), &cam, &plane, &grid);
        inp.requested = SnapFlags::EDGE;

        let snaps = proposed(&GeometrySnap, &inp, &document);

        let edge = snaps
            .iter()
            .find(|s| (s.position - on_edge).magnitude() < 1e-3)
            .unwrap_or_else(|| panic!("expected the top edge: {snaps:?}"));
        let off = (edge.position.x - x as Real).abs() + (edge.position.y - y as Real).abs();
        assert!(off < EXACT, "{:?}", edge.position);
        let direction = edge.direction.expect("an edge snap carries the curve tangent");
        assert!(direction.x.abs() + direction.y.abs() < EXACT, "{direction:?}");
    }

    #[test]
    fn meshes_that_are_not_parts_have_no_midpoints() {
        let (document, _node) = scene_with_edge();
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let mid = Point3::new(5.0, 0.0, 0.0);
        let ray = Ray::new(Point3::new(5.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let inp = input_with_ray(ray, screen_of(&cam, mid), &cam, &plane, &grid);

        let snaps = proposed(&GeometrySnap, &inp, &document);

        assert!(snaps.iter().any(|s| s.kind == SnapKind::Edge), "{snaps:?}");
        assert!(snaps.iter().all(|s| s.kind != SnapKind::Midpoint), "{snaps:?}");
    }

    /// Circle-centre candidates for a radius-2 cylinder standing on the XZ
    /// plane, 1 high, seen from above with the cursor over its axis, and the
    /// kinds `requested`.
    fn cylinder_centers(requested: SnapFlags) -> Vec<Snap> {
        let (document, _node) = doc_with_shape(Shape::cylinder(DVec3::ZERO, 2.0, DVec3::Y, 1.0));
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let ray = Ray::new(Point3::new(0.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let mut inp = input_with_ray(ray, screen_of(&cam, Point3::new(0.0, 1.0, 0.0)), &cam, &plane, &grid);
        inp.requested = requested;

        let snaps = proposed(&GeometrySnap, &inp, &document);
        snaps.into_iter().filter(|s| s.kind == SnapKind::Center).collect()
    }

    #[test]
    fn rim_centers_lie_on_the_axis_along_it() {
        const EXACT: Real = 1e-9;
        let mut heights: Vec<Real> = cylinder_centers(SnapFlags::CENTER)
            .iter()
            .map(|center| {
                let p = center.position;
                assert!(p.x.abs() < EXACT && p.z.abs() < EXACT, "off the axis: {p:?}");
                let normal = center.direction.expect("a centre carries the circle's axis");
                assert!((normal.y.abs() - 1.0).abs() < EXACT, "{normal:?}");
                p.y
            })
            .collect();
        heights.sort_by(Real::total_cmp);
        assert_eq!(heights.len(), 2, "one centre per rim: {heights:?}");
        assert!(heights[0].abs() < EXACT && (heights[1] - 1.0).abs() < EXACT, "{heights:?}");
    }

    #[test]
    fn centers_come_only_when_requested() {
        assert!(cylinder_centers(SnapFlags::all() - SnapFlags::CENTER).is_empty());
    }

    #[test]
    fn meshes_that_are_not_parts_have_no_centers() {
        let (document, _node) = scene_with_edge();
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let mid = Point3::new(5.0, 0.0, 0.0);
        let ray = Ray::new(Point3::new(5.0, 10.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let mut inp = input_with_ray(ray, screen_of(&cam, mid), &cam, &plane, &grid);
        inp.requested = SnapFlags::CENTER;

        assert!(proposed(&GeometrySnap, &inp, &document).is_empty());
    }

    /// Face-centre candidates for a ray straight down through `(x, z)`.
    fn face_centers_below(document: &Document, x: Real, z: Real) -> Vec<Snap> {
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let ray = Ray::new(Point3::new(x, 10.0, z), Vector3::new(0.0, -1.0, 0.0));
        let mut inp = input_with_ray(ray, screen_of(&cam, Point3::new(x, 0.0, z)), &cam, &plane, &grid);
        inp.requested = SnapFlags::FACE_CENTER;

        let snaps = proposed(&GeometrySnap, &inp, document);

        assert!(snaps.iter().all(|s| s.kind == SnapKind::FaceCenter), "{snaps:?}");
        snaps
    }

    /// A moved part's mesh sits in its frame; its B-rep snaps still land on the
    /// B-rep, not a second frame away.
    #[test]
    fn b_rep_snaps_follow_a_moved_part() {
        use duck_engine_scene::common::{Deg, Matrix4};

        let (mut document, node) = doc_with_box();
        let part = document.part_for_node(node).expect("node is a part");
        let moved = Matrix4::from_translation(Vector3::new(3.0, 0.0, 0.0)) * Matrix4::from_angle_y(Deg(90.0));
        document.bake_transform(part, moved).expect("move bakes");

        let centers = face_centers_below(&document, 3.2, 0.1);
        let [center] = centers.as_slice() else { panic!("expected one centre: {centers:?}") };
        assert!(close(center.position, Point3::new(3.0, 1.0, 0.0)), "{:?}", center.position);
        let normal = center.direction.expect("a face centre carries the face normal");
        assert!((normal - Vector3::new(0.0, 1.0, 0.0)).magnitude() < EPSILON, "{normal:?}");

        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let on_edge = Point3::new(4.0, 1.0, 0.5);
        let ray = Ray::new(Point3::new(4.0, 10.0, 0.5), Vector3::new(0.0, -1.0, 0.0));
        let mut inp = input_with_ray(ray, screen_of(&cam, on_edge), &cam, &plane, &grid);
        inp.requested = SnapFlags::MIDPOINT;
        let snaps = proposed(&GeometrySnap, &inp, &document);
        let midpoint = snaps
            .iter()
            .find(|s| close(s.position, Point3::new(4.0, 1.0, 0.0)))
            .unwrap_or_else(|| panic!("expected the top edge's midpoint: {snaps:?}"));
        assert!(midpoint.direction.is_some_and(|d| d.z.abs() > 0.9), "{:?}", midpoint.direction);

        inp.requested = SnapFlags::EDGE;
        let snaps = proposed(&GeometrySnap, &inp, &document);
        let edge = snaps
            .iter()
            .find(|s| close(s.position, on_edge))
            .unwrap_or_else(|| panic!("expected the top edge: {snaps:?}"));
        assert!(edge.direction.is_some_and(|d| d.z.abs() > 0.9), "{:?}", edge.direction);
    }

    #[test]
    fn face_center_snaps_to_the_middle_of_a_flat_face() {
        // A 2-unit box centred on the origin; the ray drops onto its top face
        // y = 1, off the face's middle.
        let (document, _node) = doc_with_box();

        let centers = face_centers_below(&document, 0.3, 0.2);

        let [center] = centers.as_slice() else { panic!("expected one centre: {centers:?}") };
        assert!(close(center.position, Point3::new(0.0, 1.0, 0.0)), "{:?}", center.position);
        let normal = center.direction.expect("a face centre carries the face normal");
        assert!((normal - Vector3::new(0.0, 1.0, 0.0)).magnitude() < EPSILON, "{normal:?}");
    }

    #[test]
    fn curved_faces_have_no_center() {
        // A radius-2 cylinder standing on the XZ plane, 1 tall.
        let (document, _node) = doc_with_shape(Shape::cylinder(DVec3::ZERO, 2.0, DVec3::Y, 1.0));
        let cam = dummy_camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        // Sideways onto the curved face, clear of the seam.
        let on_side = Point3::new((4.0 - 0.09 as Real).sqrt(), 0.5, 0.3);
        let ray = Ray::new(Point3::new(10.0, 0.5, 0.3), Vector3::new(-1.0, 0.0, 0.0));
        let mut inp = input_with_ray(ray, screen_of(&cam, on_side), &cam, &plane, &grid);
        inp.requested = SnapFlags::FACE | SnapFlags::FACE_CENTER;

        let snaps = proposed(&GeometrySnap, &inp, &document);

        assert!(snaps.iter().any(|s| s.kind == SnapKind::Face), "the ray misses the side: {snaps:?}");
        assert!(snaps.iter().all(|s| s.kind != SnapKind::FaceCenter), "{snaps:?}");

        // The flat cap still has one, on the axis.
        let centers = face_centers_below(&document, 0.5, 0.3);
        let [center] = centers.as_slice() else { panic!("expected one centre: {centers:?}") };
        assert!(close(center.position, Point3::new(0.0, 1.0, 0.0)), "{:?}", center.position);
    }

    #[test]
    fn only_the_visible_face_offers_its_center() {
        // A 2-unit box centred on the origin, under another whose top face is
        // y = 5.
        let (mut document, _node) = doc_with_box();
        let upper = Shape::box_from_corners(DVec3::new(-1.0, 3.0, -1.0), DVec3::new(1.0, 5.0, 1.0));
        document.add_part("upper", upper, Transform::IDENTITY, &CadTessellationOptions::default()).expect("box tessellates");

        let centers = face_centers_below(&document, 0.3, 0.2);

        let [center] = centers.as_slice() else { panic!("expected one centre: {centers:?}") };
        assert!(close(center.position, Point3::new(0.0, 5.0, 0.0)), "{:?}", center.position);
    }

    #[test]
    fn meshes_that_are_not_parts_have_no_face_centers() {
        let document = Document::new(Scene::default());
        add_triangle_node(
            document.scene(),
            [[-10.0, 0.0, -10.0], [10.0, 0.0, -10.0], [0.0, 0.0, 10.0]],
            [[0.0, 1.0, 0.0]; 3],
            Vector3::new(0.0, 0.0, 0.0),
        );

        assert!(face_centers_below(&document, 0.0, 0.0).is_empty());
    }
}
