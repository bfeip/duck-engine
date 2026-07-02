//! CAD format support: STEP and IGES.
//!
//! Import options, load functions, and the XCAF assembly pipeline.
//! Low-level tessellation is handled by [`duck_engine_scene::cad`].

use std::path::Path;

use anyhow::{bail, Context, Result};
use duck_engine_common::{decompose_matrix, Matrix4, RgbaColor};
use duck_engine_scene::cad::{tessellate_occ_shape, CadTessellationOptions};
use duck_engine_scene::common::Transform;
use duck_engine_scene::{
    Instance, LineMaterial, Mesh, MeshPrimitive, NodeFlags, NodeId, NodePayload,
    PrimitiveType, Scene, Vertex,
};
use opencascade::primitives::{Compound, EdgeType, Shape};
use opencascade::xcaf::{XcafColorTool, XcafDimTolTool, XcafDocument, XcafLabel, XcafShapeTool};

/// Options controlling how a CAD file is imported.
///
/// Tessellation-related options live in the nested [`CadTessellationOptions`].
/// PMI options are top-level here because they only apply during file import,
/// not during programmatic shape authoring.
pub struct CadImportOptions {
    pub tessellation: CadTessellationOptions,
    /// Whether to import PMI graphical presentation geometry as `LineList` meshes.
    pub include_pmi: bool,
    /// Color applied to PMI annotation lines.
    pub pmi_color: RgbaColor,
}

impl Default for CadImportOptions {
    fn default() -> Self {
        Self {
            tessellation: CadTessellationOptions::default(),
            include_pmi: true,
            pmi_color: RgbaColor { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        }
    }
}

/// Result of a hierarchy-preserving CAD file import.
pub struct CadImportResult {
    /// Root node of the imported assembly in the scene graph.
    pub root: NodeId,
    /// Root node of the PMI geometry sub-tree, if PMI was found.
    pub pmi_root: Option<NodeId>,
    /// Camera nodes for named views imported from the CAD file.
    ///
    /// TODO(cad-views): implement once camera/view node representation is finalized.
    pub views: Vec<NodeId>,
}

/// Import a STEP file into `scene`, returning a [`CadImportResult`] that mirrors
/// the assembly hierarchy.
pub fn load_step(
    path: impl AsRef<Path>,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = XcafDocument::read_step(path.as_ref())
        .with_context(|| format!("Failed to read STEP file: {}", path.as_ref().display()))?;
    import_xcaf_document(&doc, scene, options)
}

/// Import STEP data from a string into `scene`.
pub fn load_step_from_str(
    s: &str,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = XcafDocument::read_step_from_str(s).context("Failed to read STEP data")?;
    import_xcaf_document(&doc, scene, options)
}

/// Import an IGES file into `scene`, returning a [`CadImportResult`].
pub fn load_iges(
    path: impl AsRef<Path>,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = XcafDocument::read_iges(path.as_ref())
        .with_context(|| format!("Failed to read IGES file: {}", path.as_ref().display()))?;
    import_xcaf_document(&doc, scene, options)
}

/// Import IGES data from a string into `scene`.
pub fn load_iges_from_str(
    s: &str,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = XcafDocument::read_iges_from_str(s).context("Failed to read IGES data")?;
    import_xcaf_document(&doc, scene, options)
}

pub(crate) fn import_xcaf_document(
    doc: &XcafDocument,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let shape_tool = doc.shape_tool();
    let color_tool = doc.color_tool();

    let root = scene
        .add_node(None, Some("cad_import".to_string()), Transform::IDENTITY, NodeFlags::NONE)
        .context("Failed to add root CAD node")?;

    for label in shape_tool.free_shapes() {
        import_xcaf_label(&label, &shape_tool, &color_tool, scene, options, Some(root))?;
    }

    let pmi_root = if options.include_pmi {
        import_pmi(&doc.dim_tol_tool(), scene, options, root)?
    } else {
        None
    };

    let views = import_views(&doc.view_tool(), scene, options);

    Ok(CadImportResult { root, pmi_root, views })
}

fn import_xcaf_label(
    label: &XcafLabel,
    shape_tool: &XcafShapeTool,
    color_tool: &XcafColorTool,
    scene: &mut Scene,
    options: &CadImportOptions,
    parent: Option<NodeId>,
) -> Result<NodeId> {
    let name = label.name();
    let transform = matrix_to_transform(shape_tool.location_matrix(label));
    let is_assembly = shape_tool.is_assembly(label);

    let node_id = scene
        .add_node(parent, name.clone(), transform, NodeFlags::NONE)
        .context("Failed to add XCAF node")?;
    if is_assembly {
        for child in shape_tool.components(label) {
            import_xcaf_label(&child, shape_tool, color_tool, scene, options, Some(node_id))?;
        }
    } else {
        let shape = shape_tool.shape(label);

        let face_color = color_tool
            .color_of_label(label)
            .or_else(|| color_tool.color_of_shape(&shape))
            .map(|(r, g, b)| RgbaColor { r, g, b, a: 1.0 });

        import_leaf_part(&shape, scene, options, node_id, face_color)?;
    };

    Ok(node_id)
}

fn import_leaf_part(
    shape: &Shape,
    scene: &mut Scene,
    options: &CadImportOptions,
    node: NodeId,
    face_color: Option<RgbaColor>,
) -> Result<()> {
    let t = &options.tessellation;
    let mesh =
        tessellate_occ_shape(shape, t.tessellation_tolerance, t.scale_factor, t.include_edges, t.include_points)?;
    // Start from the configured material template; honor the file's per-part
    // color override when present, keeping the template's other PBR properties.
    let mut face_material = t.face_material.clone().with_fresh_id();
    if let Some(color) = face_color {
        face_material.set_base_color_factor(color);
    }
    let face_mat = scene.add_face_material(face_material);
    let line_mat = scene.add_line_material(t.line_material.clone().with_fresh_id());
    let mesh_id = scene.add_mesh(mesh);
    let instance_id = scene.add_instance(
        Instance::new(mesh_id)
            .with_face_material(face_mat)
            .with_line_material(line_mat),
    );
    scene.set_node_payload(node, NodePayload::Instance(instance_id));
    Ok(())
}

/// Converts a row-major `[[f64; 4]; 4]` matrix into a [`Transform`] via matrix decomposition.
fn matrix_to_transform(mat: [[f64; 4]; 4]) -> Transform {
    // cgmath::Matrix4::new takes arguments in column-major order:
    //   new(c0r0, c0r1, c0r2, c0r3,  c1r0, c1r1, ...)
    // mat is row-major: mat[row][col], so c{col}r{row} = mat[row][col].
    let m = Matrix4::new(
        mat[0][0] as f32, mat[1][0] as f32, mat[2][0] as f32, mat[3][0] as f32, // col 0
        mat[0][1] as f32, mat[1][1] as f32, mat[2][1] as f32, mat[3][1] as f32, // col 1
        mat[0][2] as f32, mat[1][2] as f32, mat[2][2] as f32, mat[3][2] as f32, // col 2
        mat[0][3] as f32, mat[1][3] as f32, mat[2][3] as f32, mat[3][3] as f32, // col 3
    );
    decompose_matrix(&m)
}

fn import_pmi(
    dim_tol_tool: &XcafDimTolTool,
    scene: &mut Scene,
    options: &CadImportOptions,
    parent: NodeId,
) -> Result<Option<NodeId>> {
    let s = options.tessellation.scale_factor;

    let pmi_root = scene
        .add_node(Some(parent), Some("pmi".to_string()), Transform::IDENTITY, NodeFlags::NONE)
        .context("Failed to add PMI root node")?;
    let mat = scene.add_line_material(LineMaterial::new(options.pmi_color));

    let all_labels = dim_tol_tool
        .dimension_labels()
        .chain(dim_tol_tool.geom_tolerance_labels())
        .chain(dim_tol_tool.datum_labels());

    let mut any_annotation = false;
    let (mut dim_count, mut tol_count, mut datum_count) = (0u32, 0u32, 0u32);

    for label in all_labels {
        let shape = label
            .dimension_presentation()
            .or_else(|| label.geom_tolerance_presentation())
            .or_else(|| label.datum_presentation());
        let Some(shape) = shape else { continue };

        let fallback_name = if label.is_dimension() {
            dim_count += 1;
            format!("dimension {dim_count}")
        } else if label.is_geom_tolerance() {
            tol_count += 1;
            format!("geom_tolerance {tol_count}")
        } else {
            datum_count += 1;
            format!("datum {datum_count}")
        };

        let mut verts: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for edge in shape.edges() {
            let points: Vec<_> = match edge.edge_type() {
                EdgeType::Line => vec![edge.start_point(), edge.end_point()],
                _ => edge.approximation_segments().collect(),
            };

            if points.len() < 2 {
                continue;
            }

            for window in points.windows(2) {
                let base = verts.len() as u32;
                for p in window {
                    verts.push(Vertex {
                        position: [p.x as f32 * s, p.y as f32 * s, p.z as f32 * s],
                        normal: [0.0, 0.0, 0.0],
                        tex_coords: [0.0, 0.0, 0.0],
                    });
                }
                indices.push(base);
                indices.push(base + 1);
            }
        }

        if indices.is_empty() {
            continue;
        }

        let name = label.name().unwrap_or(fallback_name);
        let primitive = MeshPrimitive { primitive_type: PrimitiveType::LineList, indices };
        let mesh_id = scene.add_mesh(Mesh::from_raw(verts, vec![primitive]));
        scene
            .add_instance_node(
                Some(pmi_root),
                Instance::new(mesh_id).with_line_material(mat),
                Some(name),
                Transform::IDENTITY,
                NodeFlags::NONE,
            )
            .context("Failed to add PMI annotation node")?;
        any_annotation = true;
    }

    if !any_annotation {
        return Ok(None);
    }

    Ok(Some(pmi_root))
}

/// TODO(cad-views): Import CAD views as camera nodes. Currently stubbed out pending
/// finalization of the camera/view node representation.
pub(crate) fn import_views(
    _view_tool: &opencascade::xcaf::XcafViewTool,
    _scene: &mut Scene,
    _options: &CadImportOptions,
) -> Vec<NodeId> {
    vec![]
}

// ============================================================================
// File extension helpers
// ============================================================================

/// File extensions handled by the CAD importer.
pub const CAD_EXTENSIONS: &[&str] = &["step", "stp", "iges", "igs"];

/// Returns true if `ext` is a CAD file extension (case-insensitive).
pub fn is_cad_extension(ext: &str) -> bool {
    CAD_EXTENSIONS.iter().any(|e| e.eq_ignore_ascii_case(ext))
}

/// Returns true if `ext` identifies a STEP file specifically.
pub fn is_step_extension(ext: &str) -> bool {
    ext.eq_ignore_ascii_case("step") || ext.eq_ignore_ascii_case("stp")
}

// ============================================================================
// Scene-free B-Rep loading and saving
// ============================================================================

/// A leaf part extracted from a CAD assembly, with its accumulated assembly
/// transform (and the caller's unit scale) baked into the B-Rep shape.
pub struct ImportedCadPart {
    pub name: Option<String>,
    pub shape: Shape,
    pub color: Option<RgbaColor>,
}

/// Load a STEP or IGES file (chosen by extension) as a flat list of world-space
/// leaf parts. `scale` is baked into each shape (e.g. `0.001` for mm→m).
pub fn load_cad_parts(path: impl AsRef<Path>, scale: f64) -> Result<Vec<ImportedCadPart>> {
    let path = path.as_ref();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let doc = if is_step_extension(ext) {
        XcafDocument::read_step(path)
            .with_context(|| format!("Failed to read STEP file: {}", path.display()))?
    } else if is_cad_extension(ext) {
        XcafDocument::read_iges(path)
            .with_context(|| format!("Failed to read IGES file: {}", path.display()))?
    } else {
        bail!("Unsupported CAD extension: {ext:?}");
    };

    let shape_tool = doc.shape_tool();
    let color_tool = doc.color_tool();
    let mut parts = Vec::new();
    for label in shape_tool.free_shapes() {
        collect_xcaf_parts(&label, &shape_tool, &color_tool, uniform_scale(1.0), scale, &mut parts);
    }
    Ok(parts)
}

/// Write `shapes` as a single compound to STEP or IGES (chosen by extension).
/// `scale` is baked into each shape before writing (e.g. `1000.0` for m→mm).
pub fn save_cad_shapes<'a>(
    path: impl AsRef<Path>,
    shapes: impl IntoIterator<Item = &'a Shape>,
    scale: f64,
) -> Result<()> {
    let path = path.as_ref();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if !is_cad_extension(ext) {
        bail!("Unsupported CAD extension: {ext:?}");
    }
    let scaled: Vec<Shape> =
        shapes.into_iter().map(|s| s.gtransform(uniform_scale(scale))).collect();
    let compound: Shape = Compound::from_shapes(&scaled).into();
    if is_step_extension(ext) {
        compound
            .write_step_to_file(path)
            .with_context(|| format!("Failed to write STEP file: {}", path.display()))
    } else {
        compound
            .write_iges_to_file(path)
            .with_context(|| format!("Failed to write IGES file: {}", path.display()))
    }
}

/// Depth-first collection of leaf parts. `acc` accumulates ancestor assembly
/// locations only: `shape(label)` already carries the label's own location, so
/// composing it here would double-apply it.
fn collect_xcaf_parts(
    label: &XcafLabel,
    shape_tool: &XcafShapeTool,
    color_tool: &XcafColorTool,
    acc: [[f64; 4]; 4],
    scale: f64,
    out: &mut Vec<ImportedCadPart>,
) {
    let referred = shape_tool.referred_label(label);
    let assembly = if referred.as_ref().is_some_and(|d| shape_tool.is_assembly(d)) {
        referred.as_ref()
    } else if shape_tool.is_assembly(label) {
        Some(label)
    } else {
        None
    };

    if let Some(assembly) = assembly {
        let world = mat_mul(&acc, &shape_tool.location_matrix(label));
        for child in shape_tool.components(assembly) {
            collect_xcaf_parts(&child, shape_tool, color_tool, world, scale, out);
        }
    } else {
        let shape = shape_tool.shape(label);
        let color = color_tool
            .color_of_label(label)
            .or_else(|| color_tool.color_of_shape(&shape))
            .map(|(r, g, b)| RgbaColor { r, g, b, a: 1.0 });
        // Reference labels get auto-generated "=>[entry]" names; prefer the
        // first real name between the instance and its prototype.
        let name = [label.name(), referred.as_ref().and_then(|d| d.name())]
            .into_iter()
            .flatten()
            .find(|n| !n.starts_with("=>"));
        let shape = shape.gtransform(mat_mul(&uniform_scale(scale), &acc));
        out.push(ImportedCadPart { name, shape, color });
    }
}

/// Row-major 4×4 matrix product `a · b`.
fn mat_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut c = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            c[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

/// Row-major uniform scale matrix.
fn uniform_scale(s: f64) -> [[f64; 4]; 4] {
    [
        [s, 0.0, 0.0, 0.0],
        [0.0, s, 0.0, 0.0],
        [0.0, 0.0, s, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Saving with one scale and loading with its inverse must round-trip
    /// geometry at the original size.
    #[test]
    fn cad_parts_round_trip_scale() {
        let path = std::env::temp_dir().join("duck_cad_round_trip.step");
        let shape = Shape::box_centered(2.0, 2.0, 2.0);
        save_cad_shapes(&path, [&shape], 1000.0).unwrap();
        let parts = load_cad_parts(&path, 0.001).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(parts.len(), 1);
        let mesh = tessellate_occ_shape(&parts[0].shape, 0.01, 1.0, false, false).unwrap();
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in mesh.vertices() {
            for i in 0..3 {
                min[i] = min[i].min(v.position[i]);
                max[i] = max[i].max(v.position[i]);
            }
        }
        for i in 0..3 {
            assert!((max[i] - min[i] - 2.0).abs() < 1e-3, "extent {i}: {}", max[i] - min[i]);
        }
    }
}
