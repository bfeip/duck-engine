use anyhow::{Context, Result};
use opencascade::primitives::{EdgeType, Shape};

use crate::common::{RgbaColor, Transform};
use crate::{
    FaceMaterial, Instance, LineMaterial, Mesh, MeshPrimitive, NodeFlags, NodeId, NodePayload,
    PrimitiveType, Scene, SubMeshRange, Topology, Vertex,
};

/// Options controlling tessellation and presentation when producing scene geometry from CAD data.
///
/// Used for both file import and programmatic authoring via [`tessellate_into`].
// TODO: It's odd to have `include_edges` and `include_points` here _and_ a material for each.
// if someone is't interesting in edges or points and doesn't include them they
// still have to provide a material.
#[derive(Clone)]
pub struct CadTessellationOptions {
    /// Tolerance used for OCCT incremental mesh tessellation. Lower values
    /// produce finer meshes. Units match the file's unit system (typically mm).
    pub tessellation_tolerance: f64,
    /// Uniform scale applied to all vertex positions. Use `0.001` to convert
    /// from millimeters (STEP default) to metres.
    pub scale_factor: f32,
    /// Material applied to triangle faces. Acts as a template: each tessellated
    /// part receives a clone with a fresh id.
    pub face_material: FaceMaterial,
    /// Material applied to wireframe edges. Acts as a template: each tessellated
    /// part receives a clone with a fresh id.
    pub line_material: LineMaterial,
    /// Whether to include wireframe edges as `LineList` meshes.
    pub include_edges: bool,
    /// Whether to include B-Rep vertices as a `PointList` mesh. Points are not
    /// drawn unless the instance carries a point material, but they are required
    /// for point picking/selection.
    pub include_points: bool,
}

impl Default for CadTessellationOptions {
    fn default() -> Self {
        Self {
            tessellation_tolerance: 0.01,
            scale_factor: 1.0,
            face_material: FaceMaterial::new()
                .with_base_color_factor(RgbaColor { r: 0.8, g: 0.8, b: 0.8, a: 1.0 }),
            line_material: LineMaterial::new(RgbaColor { r: 0.15, g: 0.15, b: 0.15, a: 1.0 }),
            include_edges: true,
            include_points: true,
        }
    }
}

/// Tessellates an OpenCASCADE B-Rep shape into a [`Mesh`] containing face triangles
/// and, optionally, wireframe edge line segments.
///
/// This is the shared tessellation kernel used by both the XCAF import path
/// and the interactive authoring path ([`tessellate_into`]).
pub fn tessellate_occ_shape(
    shape: &Shape,
    tolerance: f64,
    scale_factor: f32,
    include_edges: bool,
    include_points: bool,
) -> Result<Mesh> {
    let s = scale_factor;

    // --- Faces ---
    let (occt_mesh, occt_face_ranges) = shape
        .mesh_with_tolerance_and_ranges(tolerance)
        .context("OCCT tessellation failed")?;

    let mut vertices: Vec<Vertex> = (0..occt_mesh.vertices.len())
        .map(|i| {
            let pos = occt_mesh.vertices[i];
            let norm = occt_mesh.normals.get(i).copied().unwrap_or_default();
            let uv = occt_mesh.uvs.get(i).copied().unwrap_or_default();
            Vertex {
                position: [pos.x as f32 * s, pos.y as f32 * s, pos.z as f32 * s],
                normal: [norm.x as f32, norm.y as f32, norm.z as f32],
                tex_coords: [uv.x as f32, uv.y as f32, 0.0],
            }
        })
        .collect();

    let face_indices: Vec<u32> = occt_mesh.indices.iter().map(|&i| i as u32).collect();
    let face_ranges: Vec<SubMeshRange> = occt_face_ranges
        .iter()
        .map(|r| SubMeshRange { start: r.start, count: r.count })
        .collect();

    // --- Edges ---
    // Edge vertices are appended after face vertices; absolute vertex indices are used
    // so the LineList primitive correctly references into the combined vertex buffer.
    //
    // One `edge_range` is emitted per `shape.edges()` entry, in iteration order —
    // including a zero-length range for any edge that produces no segments. Seam
    // edges (closed-surface parameterization artifacts, e.g. a sphere's meridian)
    // and degenerate edges (no 3D curve, e.g. a sphere's poles) are suppressed but
    // still get their zero-length range. This keeps `edge_ranges` index-aligned 1:1
    // with `Shape::edges()` (mirroring how faces work), so an `edge_index` resolves
    // back to its OCCT edge by plain position.
    let mut edge_indices: Vec<u32> = Vec::new();
    let mut edge_ranges: Vec<SubMeshRange> = Vec::new();

    if include_edges {
        let seams = shape.seam_edges();
        for edge in shape.edges() {
            let suppressed = edge.is_degenerated() || seams.iter().any(|s| s.is_same(&edge));
            let points: Vec<_> = if suppressed {
                Vec::new()
            } else {
                match edge.edge_type() {
                    EdgeType::Line => vec![edge.start_point(), edge.end_point()],
                    _ => edge.approximation_segments().collect(),
                }
            };

            let seg_start = (edge_indices.len() / 2) as u32;
            let mut seg_count = 0u32;

            for window in points.windows(2) {
                let base = vertices.len() as u32;
                for p in window {
                    vertices.push(Vertex {
                        position: [p.x as f32 * s, p.y as f32 * s, p.z as f32 * s],
                        normal: [0.0, 0.0, 0.0],
                        tex_coords: [0.0, 0.0, 0.0],
                    });
                }
                edge_indices.push(base);
                edge_indices.push(base + 1);
                seg_count += 1;
            }

            edge_ranges.push(SubMeshRange { start: seg_start, count: seg_count });
        }
    }

    // --- Points ---
    // One point is emitted per `shape.vertices()` entry, in iteration order, so a
    // `point_index` resolves back to its OCCT vertex by plain position (mirroring
    // how faces and edges work). Point vertices are appended after face/edge
    // vertices and referenced by absolute index.
    let mut point_indices: Vec<u32> = Vec::new();
    let mut point_ranges: Vec<SubMeshRange> = Vec::new();

    if include_points {
        for vertex in shape.vertices() {
            let p = vertex.point();
            let base = vertices.len() as u32;
            vertices.push(Vertex {
                position: [p.x as f32 * s, p.y as f32 * s, p.z as f32 * s],
                normal: [0.0, 0.0, 0.0],
                tex_coords: [0.0, 0.0, 0.0],
            });
            point_ranges.push(SubMeshRange { start: point_indices.len() as u32, count: 1 });
            point_indices.push(base);
        }
    }

    // --- Assemble mesh ---
    let mut primitives = Vec::new();
    if !face_indices.is_empty() {
        primitives.push(MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: face_indices,
        });
    }
    if !edge_indices.is_empty() {
        primitives.push(MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: edge_indices,
        });
    }
    if !point_indices.is_empty() {
        primitives.push(MeshPrimitive {
            primitive_type: PrimitiveType::PointList,
            indices: point_indices,
        });
    }

    let mut mesh = Mesh::from_raw(vertices, primitives);
    mesh.set_topology(Topology { face_ranges, edge_ranges, pointset_ranges: point_ranges });

    Ok(mesh)
}

/// Tessellates `shape` and wires it into `scene` as a mesh + material + instance + node.
///
/// Returns the [`NodeId`] of the created node.
pub fn tessellate_into(
    shape: &Shape,
    scene: &mut Scene,
    options: &CadTessellationOptions,
    parent: Option<NodeId>,
    name: Option<&str>,
) -> Result<NodeId> {
    let mesh = tessellate_occ_shape(
        shape,
        options.tessellation_tolerance,
        options.scale_factor,
        options.include_edges,
        options.include_points,
    )?;
    let face_mat = scene.add_face_material(options.face_material.clone().with_fresh_id());
    let line_mat = scene.add_line_material(options.line_material.clone().with_fresh_id());
    let mesh_id = scene.add_mesh(mesh);
    let instance_id = scene.add_instance(
        Instance::new(mesh_id)
            .with_face_material(face_mat)
            .with_line_material(line_mat),
    );

    let node_name = name.map(|s| s.to_string());
    let node = scene
        .add_node(parent, node_name, Transform::IDENTITY, NodeFlags::NONE)
        .context("Failed to add shape node")?;
    scene.set_node_payload(node, NodePayload::Instance(instance_id));

    Ok(node)
}

/// Re-tessellates `shape` into an existing `node`, preserving its [`NodeId`] and
/// reusing its material slots. The node must already carry a
/// [`NodePayload::Instance`]; the previous mesh and instance are removed.
pub fn retessellate_node(
    shape: &Shape,
    scene: &mut Scene,
    options: &CadTessellationOptions,
    node: NodeId,
) -> Result<()> {
    let NodePayload::Instance(old_instance_id) = *scene
        .get_node(node)
        .context("node not found")?
        .payload()
    else {
        anyhow::bail!("node has no instance payload");
    };

    let (old_mesh_id, face_mat, line_mat) = {
        let old = scene
            .get_instance(old_instance_id)
            .context("instance not found")?;
        (old.mesh(), old.face_material(), old.line_material())
    };

    let mesh = tessellate_occ_shape(
        shape,
        options.tessellation_tolerance,
        options.scale_factor,
        options.include_edges,
        options.include_points,
    )?;
    let mesh_id = scene.add_mesh(mesh);

    let mut instance = Instance::new(mesh_id);
    instance.set_face_material_unchecked(face_mat);
    instance.set_line_material_unchecked(line_mat);
    let instance_id = scene.add_instance(instance);

    scene.set_node_payload(node, NodePayload::Instance(instance_id));

    if scene.is_instance_orphaned(old_instance_id) {
        scene.remove_instance(old_instance_id);
    }
    if scene.is_mesh_orphaned(old_mesh_id) {
        scene.remove_mesh(old_mesh_id);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_options() -> CadTessellationOptions {
        CadTessellationOptions::default()
    }

    #[test]
    fn sphere_tessellates_to_nonempty_mesh() {
        let shape = opencascade::primitives::Shape::sphere(1.0).build();
        let mut scene = Scene::new();
        tessellate_into(&shape, &mut scene, &default_options(), None, Some("sphere"))
            .expect("tessellation failed");
        assert!(scene.mesh_count() > 0);
        assert!(scene.node_count() > 0);
    }

    #[test]
    fn cuboid_tessellates_to_nonempty_mesh() {
        let shape = opencascade::primitives::Shape::box_centered(2.0, 2.0, 2.0);
        let mut scene = Scene::new();
        tessellate_into(&shape, &mut scene, &default_options(), None, Some("box"))
            .expect("tessellation failed");
        assert!(scene.mesh_count() > 0);
    }

    #[test]
    fn union_of_cuboid_and_sphere_tessellates() {
        let a = opencascade::primitives::Shape::box_centered(2.0, 2.0, 2.0);
        let b = opencascade::primitives::Shape::sphere(1.5).build();
        let combined = a.union(&b).expect("fuse succeeds").shape;
        let mut scene = Scene::new();
        tessellate_into(&combined, &mut scene, &default_options(), None, None)
            .expect("union tessellation failed");
        assert!(scene.mesh_count() > 0);
    }

    #[test]
    fn cylinder_tessellates() {
        let shape = opencascade::primitives::Shape::cylinder_radius_height(0.5, 2.0);
        let mut scene = Scene::new();
        tessellate_into(&shape, &mut scene, &default_options(), None, None).unwrap();
        assert!(scene.mesh_count() > 0);
    }

    #[test]
    fn torus_tessellates() {
        let shape = opencascade::primitives::Shape::torus().radius_1(2.0).radius_2(0.5).build();
        let mut scene = Scene::new();
        tessellate_into(&shape, &mut scene, &default_options(), None, None).unwrap();
        assert!(scene.mesh_count() > 0);
    }

    #[test]
    fn sphere_renders_without_seam_or_pole_edges() {
        // All of a full sphere's edges are parameterization artifacts (one seam
        // meridian + two degenerate poles): none may produce line segments, but
        // each still gets its zero-length range for positional index alignment.
        let shape = opencascade::primitives::Shape::sphere(1.0).build();
        let mesh = tessellate_occ_shape(&shape, 0.01, 1.0, true, false).unwrap();

        let topology = mesh.topology().expect("topology");
        assert_eq!(topology.edge_ranges.len(), shape.edges().count());
        assert!(topology.edge_ranges.iter().all(|r| r.count == 0));
        assert!(!mesh
            .primitives()
            .iter()
            .any(|p| p.primitive_type == PrimitiveType::LineList));
    }

    #[test]
    fn cylinder_seam_suppressed_but_rims_kept_and_aligned() {
        // Exactly the seam edge's ranges are empty; the two rim circles render.
        // Alignment with `shape.edges()` positions is what edge picking relies on.
        let shape = opencascade::primitives::Shape::cylinder_radius_height(0.5, 2.0);
        let mesh = tessellate_occ_shape(&shape, 0.01, 1.0, true, false).unwrap();

        let topology = mesh.topology().expect("topology");
        let edges: Vec<_> = shape.edges().collect();
        assert_eq!(topology.edge_ranges.len(), edges.len());

        let seams = shape.seam_edges();
        assert_eq!(seams.len(), 1);
        // The explorer yields an edge once per incident face, so rendered
        // occurrences are deduped by topology to count the actual rim circles.
        let mut rendered: Vec<&opencascade::primitives::Edge> = Vec::new();
        for (edge, range) in edges.iter().zip(&topology.edge_ranges) {
            let suppressed =
                edge.is_degenerated() || seams.iter().any(|s| s.is_same(edge));
            assert_eq!(range.count == 0, suppressed);
            if range.count > 0 && !rendered.iter().any(|r| r.is_same(edge)) {
                rendered.push(edge);
            }
        }
        assert_eq!(rendered.len(), 2, "both rim circles must render");
    }

    #[test]
    fn each_part_gets_distinct_materials() {
        // The material fields act as templates: tessellating multiple parts from
        // one options value must produce a distinct material per part, otherwise
        // they collide on the same id in the scene's material maps.
        let options = default_options();
        let mut scene = Scene::new();
        for _ in 0..3 {
            let shape = opencascade::primitives::Shape::box_centered(1.0, 1.0, 1.0);
            tessellate_into(&shape, &mut scene, &options, None, None).unwrap();
        }
        assert_eq!(scene.face_material_count(), 3);
        assert_eq!(scene.line_material_count(), 3);
    }

    #[test]
    fn gtransform_applies_non_uniform_scale() {
        // A 2×2×2 box spans [-1, 1] on each axis. A non-uniform scale of 3× in X
        // (identity elsewhere) must stretch only the X extent.
        let shape = opencascade::primitives::Shape::box_centered(2.0, 2.0, 2.0);
        let scaled = shape.gtransform([
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        let mut scene = Scene::new();
        let node = tessellate_into(&scaled, &mut scene, &default_options(), None, None).unwrap();
        let aabb = scene.nodes_bounding(node).bounds.expect("scaled box has bounds");
        let (sx, sy, sz) = aabb.size();
        assert!((sx - 6.0).abs() < 1e-3, "expected X extent ~6, got {sx}");
        assert!((sy - 2.0).abs() < 1e-3, "expected Y extent ~2, got {sy}");
        assert!((sz - 2.0).abs() < 1e-3, "expected Z extent ~2, got {sz}");
    }

    #[test]
    fn retessellate_node_preserves_node_id_and_updates_geometry() {
        let options = default_options();
        let mut scene = Scene::new();
        let shape = opencascade::primitives::Shape::box_centered(2.0, 2.0, 2.0);
        let node = tessellate_into(&shape, &mut scene, &options, None, None).unwrap();

        let before = scene.nodes_bounding(node).bounds.expect("box has bounds");
        let (bx, _, _) = before.size();
        assert!((bx - 2.0).abs() < 1e-3);

        // Re-tessellate the same node with a stretched copy of the shape.
        let scaled = shape.gtransform([
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        retessellate_node(&scaled, &mut scene, &options, node).unwrap();

        // Same node id, geometry updated, and no leaked mesh/instance.
        assert!(scene.get_node(node).is_some(), "node id must be preserved");
        assert_eq!(scene.mesh_count(), 1, "old mesh should be removed");
        assert_eq!(scene.instance_count(), 1, "old instance should be removed");
        let after = scene.nodes_bounding(node).bounds.expect("rescaled box has bounds");
        let (ax, _, _) = after.size();
        assert!((ax - 6.0).abs() < 1e-3, "expected X extent ~6 after retess, got {ax}");
    }

    #[test]
    fn retessellate_node_keeps_shared_instance_and_mesh() {
        let options = default_options();
        let mut scene = Scene::new();
        let shape = opencascade::primitives::Shape::box_centered(2.0, 2.0, 2.0);
        let node1 = tessellate_into(&shape, &mut scene, &options, None, None).unwrap();

        // Capture the instance + mesh the part created.
        let NodePayload::Instance(instance_id) = *scene.get_node(node1).unwrap().payload() else {
            panic!("expected instance payload");
        };
        let mesh_id = scene.get_instance(instance_id).unwrap().mesh();

        // A second node deliberately sharing the same instance.
        let node2 = scene.add_node(None, None, Transform::IDENTITY, NodeFlags::NONE).unwrap();
        scene.set_node_payload(node2, NodePayload::Instance(instance_id));

        let scaled = shape.gtransform([
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        retessellate_node(&scaled, &mut scene, &options, node1).unwrap();

        // The shared instance and mesh must survive — node2 still references them.
        assert!(scene.get_instance(instance_id).is_some(), "shared instance must survive");
        assert!(scene.get_mesh(mesh_id).is_some(), "shared mesh must survive");

        // node1's geometry was still updated to the stretched shape.
        let aabb = scene.nodes_bounding(node1).bounds.expect("bounds");
        let (sx, _, _) = aabb.size();
        assert!((sx - 6.0).abs() < 1e-3, "expected X extent ~6, got {sx}");
    }
}
