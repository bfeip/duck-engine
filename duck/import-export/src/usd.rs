//! USD (Universal Scene Description) loading.
//!
//! Supports USDC (binary), USDA (text), and USDZ (archive) formats.
//! Uses the `openusd` crate for low-level SDF access and manually extracts
//! geometry, materials, transforms, lights, and cameras.

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use anyhow::{Result, anyhow};
use duck_engine_common::{
    Deg, Matrix3, Matrix4, Point3, Quaternion, Rotation3, SquareMatrix, Vector3,
};

use openusd::sdf::{self, AbstractData, Value};

use duck_engine_scene::{Light, PositionedCamera, SceneData};
use duck_engine_scene::resource::{
    FaceMaterial, FaceMaterialHandle, Instance, Mesh, MeshHandle, MeshPrimitive, NodeFlags, NodeId,
    NodePayload, PrimitiveType, Vertex,
};
use duck_engine_scene::common::{RgbaColor, Transform, decompose_matrix};

/// Result of loading a scene from USD.
pub struct UsdLoadResult {
    pub scene: SceneData,
    pub camera: Option<PositionedCamera>,
}

/// File extensions handled by the USD loader.
pub const USD_EXTENSIONS: &[&str] = &["usd", "usda", "usdc", "usdz"];

/// Check if a file extension is a USD format.
pub fn is_usd_extension(ext: &str) -> bool {
    USD_EXTENSIONS.contains(&ext.to_lowercase().as_str())
}

/// Load a USD scene from a file path.
pub fn load_usd_scene_from_path(path: &Path) -> Result<UsdLoadResult> {
    let bytes = std::fs::read(path)?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("scene.usd");
    load_usd_scene_from_bytes(&bytes, name)
}

/// Load a USD scene from bytes in memory.
pub fn load_usd_scene_from_bytes(bytes: &[u8], name: &str) -> Result<UsdLoadResult> {
    let mut data = open_usd_data(bytes, name)?;
    convert_scene(data.as_mut())
}

// ============================================================================
// Format Detection & Parsing
// ============================================================================

/// Open USD data from bytes, auto-detecting the subformat (USDC/USDA/USDZ).
fn open_usd_data(bytes: &[u8], name: &str) -> Result<Box<dyn AbstractData>> {
    if bytes.len() >= 8 && bytes.starts_with(b"PXR-USDC") {
        let cursor = Cursor::new(bytes.to_vec());
        let data = openusd::usdc::CrateData::open(cursor, true)
            .map_err(|e| anyhow!("Failed to parse USDC: {}", e))?;
        Ok(Box::new(data))
    } else if bytes.len() >= 2 && bytes[0] == b'P' && bytes[1] == b'K' {
        open_usdz_from_bytes(bytes)
    } else {
        open_usda_from_bytes(bytes, name)
    }
}

/// Extract and parse the main scene file from a USDZ archive.
fn open_usdz_from_bytes(bytes: &[u8]) -> Result<Box<dyn AbstractData>> {
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| anyhow!("Failed to open USDZ archive: {}", e))?;

    // Find the first .usdc or .usda file
    let scene_file_name = (0..archive.len())
        .find_map(|i| {
            let file = archive.by_index(i).ok()?;
            let name = file.name().to_string();
            if name.ends_with(".usdc") || name.ends_with(".usda") || name.ends_with(".usd") {
                Some(name)
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow!("No USD scene file found in USDZ archive"))?;

    let mut file = archive
        .by_name(&scene_file_name)
        .map_err(|e| anyhow!("Failed to read {} from USDZ: {}", scene_file_name, e))?;

    let mut inner_bytes = Vec::new();
    file.read_to_end(&mut inner_bytes)?;
    drop(file);

    // Recursively parse the extracted content
    open_usd_data(&inner_bytes, &scene_file_name)
}

/// Parse USDA text format from bytes (writes to temp file since TextReader needs a path).
fn open_usda_from_bytes(bytes: &[u8], name: &str) -> Result<Box<dyn AbstractData>> {
    let tmp_path = std::env::temp_dir().join(format!("duck_usd_{}", name));
    std::fs::write(&tmp_path, bytes)?;
    let result = openusd::usda::TextReader::read(
        tmp_path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid temp path"))?,
    )
    .map_err(|e| anyhow!("Failed to parse USDA: {}", e));
    let _ = std::fs::remove_file(&tmp_path);
    Ok(Box::new(result?))
}

// ============================================================================
// SDF Value Helpers
// ============================================================================

/// Get a Token (string) field from a spec.
fn get_token(data: &mut dyn AbstractData, path: &sdf::Path, field: &str) -> Option<String> {
    let val = data.get(path, field).ok()?;
    match val.as_ref() {
        Value::Token(s) => Some(s.clone()),
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// Get a TokenVec field from a spec.
fn get_token_vec(data: &mut dyn AbstractData, path: &sdf::Path, field: &str) -> Vec<String> {
    let val = match data.get(path, field) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    match val.as_ref() {
        Value::TokenVec(v) => v.clone(),
        _ => Vec::new(),
    }
}

/// Get a float array (Vec3f, FloatVec, Vec2f, etc.) as Vec<f32>.
fn get_float_array(data: &mut dyn AbstractData, path: &sdf::Path, field: &str) -> Option<Vec<f32>> {
    let val = data.get(path, field).ok()?;
    extract_float_array_from_value(val.as_ref())
}

/// Extract a flat f32 array from various Value types.
fn extract_float_array_from_value(val: &Value) -> Option<Vec<f32>> {
    match val {
        Value::Vec3f(v) | Value::Vec2f(v) | Value::FloatVec(v) => Some(v.clone()),
        Value::Vec3d(v) | Value::Vec2d(v) | Value::DoubleVec(v) => {
            Some(v.iter().map(|&d| d as f32).collect())
        }
        _ => None,
    }
}

/// Get an int array (IntVec).
fn get_int_array(data: &mut dyn AbstractData, path: &sdf::Path, field: &str) -> Option<Vec<i32>> {
    let val = data.get(path, field).ok()?;
    match val.as_ref() {
        Value::IntVec(v) => Some(v.clone()),
        _ => None,
    }
}

// ============================================================================
// Path Utilities
// ============================================================================

/// Construct a child prim path.
fn make_child_path(parent: &sdf::Path, child_name: &str) -> sdf::Path {
    let parent_str = parent.as_str();
    if parent_str == "/" {
        sdf::Path::new(&format!("/{}", child_name)).expect("valid child path")
    } else {
        sdf::Path::new(&format!("{}/{}", parent_str, child_name)).expect("valid child path")
    }
}

/// Construct a property path from a prim path and property name.
fn make_property_path(prim: &sdf::Path, prop: &str) -> sdf::Path {
    sdf::Path::new(&format!("{}.{}", prim.as_str(), prop)).expect("valid property path")
}

/// Get the prim's type name (e.g., "Mesh", "Material", "Xform", "Camera").
fn get_prim_type(data: &mut dyn AbstractData, path: &sdf::Path) -> Option<String> {
    get_token(data, path, "typeName")
}

/// Get children of a prim.
fn get_prim_children(data: &mut dyn AbstractData, path: &sdf::Path) -> Vec<String> {
    get_token_vec(data, path, "primChildren")
}

// ============================================================================
// Conversion Pipeline
// ============================================================================

fn convert_scene(data: &mut dyn AbstractData) -> Result<UsdLoadResult> {
    let mut scene = SceneData::new();
    let root = sdf::Path::abs_root();

    // Phase 1: Collect all materials (pre-pass)
    let mut material_map: HashMap<String, FaceMaterialHandle> = HashMap::new();
    collect_materials_recursive(data, &root, &mut material_map, &mut scene);

    // Phase 2: Build node hierarchy with meshes, lights, cameras
    let mut camera = None;
    let mut fallback_material_id: Option<FaceMaterialHandle> = None;
    let children = get_prim_children(data, &root);
    for child_name in &children {
        let child_path = make_child_path(&root, child_name);
        build_node_recursive(
            data,
            &child_path,
            &material_map,
            &mut scene,
            None,
            &mut camera,
            &mut fallback_material_id,
        )?;
    }

    Ok(UsdLoadResult { scene, camera })
}

// ============================================================================
// Material Collection
// ============================================================================

/// Recursively walk the prim tree to find and create all materials.
fn collect_materials_recursive(
    data: &mut dyn AbstractData,
    path: &sdf::Path,
    material_map: &mut HashMap<String, FaceMaterialHandle>,
    scene: &mut SceneData,
) {
    let children = get_prim_children(data, path);
    for child_name in &children {
        let child_path = make_child_path(path, child_name);
        let type_name = get_prim_type(data, &child_path);

        if type_name.as_deref() == Some("Material") {
            let material = extract_material(data, &child_path);
            let mat_id = scene.add_face_material(material);
            material_map.insert(child_path.as_str().to_string(), mat_id);
        }

        // Recurse into children (materials can be nested under Scope, etc.)
        collect_materials_recursive(data, &child_path, material_map, scene);
    }
}

/// Extract material properties from a Material prim.
fn extract_material(data: &mut dyn AbstractData, mat_path: &sdf::Path) -> FaceMaterial {
    let mut material = FaceMaterial::new();

    // Find the UsdPreviewSurface shader child
    let children = get_prim_children(data, mat_path);
    for child_name in &children {
        let shader_path = make_child_path(mat_path, child_name);
        let type_name = get_prim_type(data, &shader_path);

        if type_name.as_deref() != Some("Shader") {
            continue;
        }

        // Check if this is a UsdPreviewSurface
        let shader_id = get_shader_token(data, &shader_path, "info:id");
        if shader_id.as_deref() != Some("UsdPreviewSurface") {
            continue;
        }

        // Extract shader inputs
        if let Some(color) = get_shader_color3f(data, &shader_path, "inputs:diffuseColor") {
            material = material.with_base_color_factor(color);
        }
        if let Some(metallic) = get_shader_float(data, &shader_path, "inputs:metallic") {
            material = material.with_metallic_factor(metallic);
        }
        if let Some(roughness) = get_shader_float(data, &shader_path, "inputs:roughness") {
            material = material.with_roughness_factor(roughness);
        }

        break; // Use the first UsdPreviewSurface found
    }

    material
}

/// Get a token value from a shader property.
fn get_shader_token(
    data: &mut dyn AbstractData,
    shader_path: &sdf::Path,
    input: &str,
) -> Option<String> {
    let prop_path = make_property_path(shader_path, input);
    get_token(data, &prop_path, "default")
}

/// Get a color3f from a shader input.
fn get_shader_color3f(
    data: &mut dyn AbstractData,
    shader_path: &sdf::Path,
    input: &str,
) -> Option<RgbaColor> {
    let prop_path = make_property_path(shader_path, input);
    let val = data.get(&prop_path, "default").ok()?;
    match val.as_ref() {
        Value::Vec3f(v) if v.len() >= 3 => Some(RgbaColor {
            r: v[0],
            g: v[1],
            b: v[2],
            a: 1.0,
        }),
        _ => None,
    }
}

/// Get a float from a shader input.
fn get_shader_float(
    data: &mut dyn AbstractData,
    shader_path: &sdf::Path,
    input: &str,
) -> Option<f32> {
    let prop_path = make_property_path(shader_path, input);
    let val = data.get(&prop_path, "default").ok()?;
    match val.as_ref() {
        Value::Float(f) => Some(*f),
        Value::Double(d) => Some(*d as f32),
        _ => None,
    }
}

// ============================================================================
// Node Hierarchy + Mesh/Light/Camera Extraction
// ============================================================================

/// Recursively build the scene node tree from the USD prim hierarchy.
fn build_node_recursive(
    data: &mut dyn AbstractData,
    prim_path: &sdf::Path,
    material_map: &HashMap<String, FaceMaterialHandle>,
    scene: &mut SceneData,
    parent: Option<NodeId>,
    camera_out: &mut Option<PositionedCamera>,
    fallback_material_id: &mut Option<FaceMaterialHandle>,
) -> Result<()> {
    let type_name = get_prim_type(data, prim_path).unwrap_or_default();
    let name = prim_path
        .as_str()
        .rsplit('/')
        .next()
        .map(|s| s.to_string());

    // Extract transform
    let transform = extract_transform(data, prim_path);

    match type_name.as_str() {
        "Mesh" => {
            let mesh_entries = extract_mesh(data, prim_path, material_map, scene, fallback_material_id);

            if mesh_entries.len() == 1 {
                let (mesh, material) = mesh_entries.into_iter().next().unwrap();
                let instance = Instance::new(mesh).with_face_material(material);
                let node_id = scene.add_instance_node(
                    parent, instance, name, transform, NodeFlags::NONE
                )?.id();
                recurse_children(data, prim_path, material_map, scene, node_id, camera_out, fallback_material_id)?;
            } else if mesh_entries.len() > 1 {
                // Split mesh: create group node, then instance children
                let group_id = scene.add_node(parent, name, transform, NodeFlags::NONE)?.id();
                for (i, (mesh, material)) in mesh_entries.into_iter().enumerate() {
                    scene.add_instance_node(
                        Some(group_id),
                        Instance::new(mesh).with_face_material(material),
                        Some(format!("chunk_{}", i)),
                        Transform::IDENTITY,
                        NodeFlags::NONE
                    )?;
                }
                recurse_children(data, prim_path, material_map, scene, group_id, camera_out, fallback_material_id)?;
            }
        }
        "Camera" => {
            if camera_out.is_none() {
                *camera_out = extract_camera(data, prim_path);
            }
        }
        "DistantLight" | "RectLight" | "SphereLight" | "DiskLight" => {
            extract_light(data, prim_path, &type_name, &transform.position, scene);
        }
        "Material" | "Shader" => {
            // Already handled in material collection phase
        }
        _ => {
            // Xform, Scope, or unknown → create a transform node and recurse
            let node_id = scene.add_node(parent, name, transform, NodeFlags::NONE)?.id();
            recurse_children(data, prim_path, material_map, scene, node_id, camera_out, fallback_material_id)?;
        }
    }

    Ok(())
}

/// Recurse into children of a prim.
fn recurse_children(
    data: &mut dyn AbstractData,
    prim_path: &sdf::Path,
    material_map: &HashMap<String, FaceMaterialHandle>,
    scene: &mut SceneData,
    parent_node: NodeId,
    camera_out: &mut Option<PositionedCamera>,
    fallback_material_id: &mut Option<FaceMaterialHandle>,
) -> Result<()> {
    let children = get_prim_children(data, prim_path);
    for child_name in &children {
        let child_path = make_child_path(prim_path, child_name);
        build_node_recursive(
            data,
            &child_path,
            material_map,
            scene,
            Some(parent_node),
            camera_out,
            fallback_material_id,
        )?;
    }
    Ok(())
}

// ============================================================================
// Transform Extraction
// ============================================================================

/// Extract and compose transform from USD xform ops.
fn extract_transform(
    data: &mut dyn AbstractData,
    prim_path: &sdf::Path,
) -> Transform {
    let default = Transform::IDENTITY;

    // Try getting xformOpOrder from the property spec
    let op_order_path = make_property_path(prim_path, "xformOpOrder");
    let mut ops = get_token_vec(data, &op_order_path, "default");

    // Also try getting from the prim directly
    if ops.is_empty() {
        ops = get_token_vec(data, prim_path, "xformOpOrder");
    }

    if ops.is_empty() {
        return default;
    }

    compose_xform_ops(data, prim_path, &ops)
}

/// Compose xform operations into a final transform matrix, then decompose.
fn compose_xform_ops(
    data: &mut dyn AbstractData,
    prim_path: &sdf::Path,
    ops: &[String],
) -> Transform {
    let mut matrix = Matrix4::identity();

    for op in ops {
        let prop_path = make_property_path(prim_path, op);

        if op.contains("transform") || op.contains("Transform") {
            if let Some(m) = get_matrix4_from_prop(data, &prop_path) {
                matrix = matrix * m;
            }
        } else if op.contains("translate") || op.contains("Translate") {
            if let Some(t) = get_vec3_from_prop(data, &prop_path) {
                matrix = matrix * Matrix4::from_translation(t);
            }
        } else if op.contains("rotateXYZ") {
            if let Some(r) = get_vec3_from_prop(data, &prop_path) {
                let rx = Matrix4::from(Matrix3::from_angle_x(Deg(r.x)));
                let ry = Matrix4::from(Matrix3::from_angle_y(Deg(r.y)));
                let rz = Matrix4::from(Matrix3::from_angle_z(Deg(r.z)));
                matrix = matrix * rx * ry * rz;
            }
        } else if op.contains("rotateX") && !op.contains("rotateXY") {
            if let Some(angle) = get_float_from_prop(data, &prop_path) {
                matrix = matrix * Matrix4::from(Matrix3::from_angle_x(Deg(angle)));
            }
        } else if op.contains("rotateY") && !op.contains("rotateYZ") {
            if let Some(angle) = get_float_from_prop(data, &prop_path) {
                matrix = matrix * Matrix4::from(Matrix3::from_angle_y(Deg(angle)));
            }
        } else if op.contains("rotateZ") {
            if let Some(angle) = get_float_from_prop(data, &prop_path) {
                matrix = matrix * Matrix4::from(Matrix3::from_angle_z(Deg(angle)));
            }
        } else if op.contains("scale") || op.contains("Scale") {
            if let Some(s) = get_vec3_from_prop(data, &prop_path) {
                matrix = matrix * Matrix4::from_nonuniform_scale(s.x, s.y, s.z);
            }
        } else {
            log::debug!("Unknown xform op: {}", op);
        }
    }

    decompose_matrix(&matrix)
}

/// Get Matrix4 from a property spec's default value.
/// USD stores matrices row-major; cgmath uses column-major, so we transpose.
fn get_matrix4_from_prop(data: &mut dyn AbstractData, prop_path: &sdf::Path) -> Option<Matrix4> {
    let val = data.get(prop_path, "default").ok()?;
    match val.as_ref() {
        Value::Matrix4d(v) if v.len() >= 16 => {
            // USD row-major → cgmath column-major (transpose)
            #[rustfmt::skip]
            let m = Matrix4::new(
                v[0] as f32,  v[4] as f32,  v[8] as f32,  v[12] as f32,
                v[1] as f32,  v[5] as f32,  v[9] as f32,  v[13] as f32,
                v[2] as f32,  v[6] as f32,  v[10] as f32, v[14] as f32,
                v[3] as f32,  v[7] as f32,  v[11] as f32, v[15] as f32,
            );
            Some(m)
        }
        _ => None,
    }
}

/// Get Vec3 from a property spec's default value.
fn get_vec3_from_prop(data: &mut dyn AbstractData, prop_path: &sdf::Path) -> Option<Vector3> {
    let val = data.get(prop_path, "default").ok()?;
    match val.as_ref() {
        Value::Vec3f(v) if v.len() >= 3 => Some(Vector3::new(v[0], v[1], v[2])),
        Value::Vec3d(v) if v.len() >= 3 => {
            Some(Vector3::new(v[0] as f32, v[1] as f32, v[2] as f32))
        }
        _ => None,
    }
}

/// Get float from a property spec's default value.
fn get_float_from_prop(data: &mut dyn AbstractData, prop_path: &sdf::Path) -> Option<f32> {
    let val = data.get(prop_path, "default").ok()?;
    match val.as_ref() {
        Value::Float(f) => Some(*f),
        Value::Double(d) => Some(*d as f32),
        _ => None,
    }
}

// ============================================================================
// Mesh Extraction
// ============================================================================

/// Extract mesh geometry from a Mesh prim.
/// Returns a Vec of (mesh, material) pairs (one per material binding found).
fn extract_mesh(
    data: &mut dyn AbstractData,
    mesh_path: &sdf::Path,
    material_map: &HashMap<String, FaceMaterialHandle>,
    scene: &mut SceneData,
    fallback_material_id: &mut Option<FaceMaterialHandle>,
) -> Vec<(MeshHandle, FaceMaterialHandle)> {
    let material_id = get_material_binding(data, mesh_path, material_map)
        .unwrap_or_else(|| {
            fallback_material_id
                .get_or_insert_with(|| scene.add_face_material(FaceMaterial::default()))
                .clone()
        });

    // Read geometry attributes
    let points_path = make_property_path(mesh_path, "points");
    let points = match get_float_array(data, &points_path, "default") {
        Some(p) if !p.is_empty() => p,
        _ => return Vec::new(),
    };

    let counts_path = make_property_path(mesh_path, "faceVertexCounts");
    let face_vertex_counts = get_int_array(data, &counts_path, "default");

    let indices_path = make_property_path(mesh_path, "faceVertexIndices");
    let face_vertex_indices = get_int_array(data, &indices_path, "default");

    // Read normals
    let normals_path = make_property_path(mesh_path, "normals");
    let normals = get_float_array(data, &normals_path, "default");

    // Read UVs - try various primvar names
    let uvs = try_read_uvs(data, mesh_path);

    // Build triangulated geometry
    let num_points = points.len() / 3;
    let (vertices, indices) =
        if let (Some(counts), Some(face_indices)) = (face_vertex_counts, face_vertex_indices) {
            triangulate_mesh(&points, &normals, &uvs, &counts, &face_indices)
        } else {
            // No face data — treat points as individual vertices
            let verts: Vec<Vertex> = (0..num_points)
                .map(|i| {
                    let position = [points[i * 3], points[i * 3 + 1], points[i * 3 + 2]];
                    let normal = normals
                        .as_ref()
                        .and_then(|n| {
                            if i * 3 + 2 < n.len() {
                                Some([n[i * 3], n[i * 3 + 1], n[i * 3 + 2]])
                            } else {
                                None
                            }
                        })
                        .unwrap_or([0.0, 1.0, 0.0]);
                    let tex_coords = uvs
                        .as_ref()
                        .and_then(|u| {
                            if i * 2 + 1 < u.len() {
                                Some([u[i * 2], u[i * 2 + 1], 0.0])
                            } else {
                                None
                            }
                        })
                        .unwrap_or([0.0, 0.0, 0.0]);
                    Vertex {
                        position,
                        normal,
                        tex_coords,
                    }
                })
                .collect();
            let tri_indices: Vec<u32> = (0..verts.len() as u32).collect();
            (verts, tri_indices)
        };

    if vertices.is_empty() {
        return Vec::new();
    }

    let primitive = MeshPrimitive { primitive_type: PrimitiveType::TriangleList, indices };
    let mesh_id = scene.add_mesh(Mesh::from_raw(vertices, vec![primitive]));
    vec![(mesh_id, material_id)]
}

/// Try reading UV coordinates from various primvar names.
fn try_read_uvs(data: &mut dyn AbstractData, mesh_path: &sdf::Path) -> Option<Vec<f32>> {
    for name in &[
        "primvars:st",
        "primvars:UVMap",
        "primvars:uv",
        "primvars:st0",
    ] {
        let uv_path = make_property_path(mesh_path, name);
        if let Some(uvs) = get_float_array(data, &uv_path, "default")
            && !uvs.is_empty() {
                return Some(uvs);
            }
    }
    None
}

/// Get the material binding for a mesh prim, returning `None` when no material is bound.
fn get_material_binding(
    data: &mut dyn AbstractData,
    mesh_path: &sdf::Path,
    material_map: &HashMap<String, FaceMaterialHandle>,
) -> Option<FaceMaterialHandle> {
    let rel_path = make_property_path(mesh_path, "material:binding");

    ["targetPaths", "default"].iter().find_map(|field| {
        let val = data.get(&rel_path, field).ok()?;
        let path_str = extract_first_path_from_value(val.as_ref())?;
        material_map.get(&path_str).cloned()
    })
}

/// Try to extract a path string from various Value types used for relationships.
fn extract_first_path_from_value(val: &Value) -> Option<String> {
    match val {
        Value::PathListOp(list_op) => list_op
            .explicit_items
            .first()
            .or_else(|| list_op.prepended_items.first())
            .or_else(|| list_op.appended_items.first())
            .map(|p| p.as_str().to_string()),
        // Some implementations store targets differently
        Value::Token(s) | Value::String(s) | Value::AssetPath(s) => Some(s.clone()),
        _ => None,
    }
}

/// Triangulate USD face data (n-gon support via fan triangulation).
///
/// USD meshes use faceVertexCounts + faceVertexIndices to describe n-gon faces.
/// This function converts them to a triangle list with proper vertex attributes.
fn triangulate_mesh(
    points: &[f32],              // flat [x,y,z, ...] per vertex
    normals: &Option<Vec<f32>>,  // flat [x,y,z, ...] may be faceVarying or vertex
    uvs: &Option<Vec<f32>>,      // flat [u,v, ...] may be faceVarying or vertex
    face_vertex_counts: &[i32],  // number of vertices per face
    face_vertex_indices: &[i32], // vertex indices for all faces
) -> (Vec<Vertex>, Vec<u32>) {
    let num_points = points.len() / 3;
    let total_face_verts: usize = face_vertex_counts.iter().map(|&c| c.max(0) as usize).sum();

    // Determine if normals/UVs are faceVarying (one per face vertex) or vertex (one per point)
    let normals_face_varying = normals
        .as_ref()
        .map(|n| n.len() / 3 == total_face_verts && total_face_verts != num_points)
        .unwrap_or(false);
    let uvs_face_varying = uvs
        .as_ref()
        .map(|u| u.len() / 2 == total_face_verts && total_face_verts != num_points)
        .unwrap_or(false);

    if !normals_face_varying && !uvs_face_varying {
        // Per-vertex interpolation: build one Vertex per point, emit triangle indices
        let vertices: Vec<Vertex> = (0..num_points)
            .map(|i| {
                let position = [points[i * 3], points[i * 3 + 1], points[i * 3 + 2]];
                let normal = normals
                    .as_ref()
                    .and_then(|n| {
                        if i * 3 + 2 < n.len() {
                            Some([n[i * 3], n[i * 3 + 1], n[i * 3 + 2]])
                        } else {
                            None
                        }
                    })
                    .unwrap_or([0.0, 1.0, 0.0]);
                let tex_coords = uvs
                    .as_ref()
                    .and_then(|u| {
                        if i * 2 + 1 < u.len() {
                            Some([u[i * 2], u[i * 2 + 1], 0.0])
                        } else {
                            None
                        }
                    })
                    .unwrap_or([0.0, 0.0, 0.0]);
                Vertex {
                    position,
                    normal,
                    tex_coords,
                }
            })
            .collect();

        // Fan-triangulate each face
        let mut tri_indices: Vec<u32> = Vec::new();
        let mut idx_offset = 0usize;
        for &count in face_vertex_counts {
            let n = count.max(0) as usize;
            if n >= 3 && idx_offset + n <= face_vertex_indices.len() {
                let v0 = face_vertex_indices[idx_offset] as u32;
                for j in 1..n - 1 {
                    tri_indices.push(v0);
                    tri_indices.push(face_vertex_indices[idx_offset + j] as u32);
                    tri_indices.push(face_vertex_indices[idx_offset + j + 1] as u32);
                }
            }
            idx_offset += n;
        }

        (vertices, tri_indices)
    } else {
        // Face-varying: emit unique vertices per face corner
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut tri_indices: Vec<u32> = Vec::new();
        let mut fv_offset = 0usize;

        for &count in face_vertex_counts {
            let n = count.max(0) as usize;
            if n < 3 || fv_offset + n > face_vertex_indices.len() {
                fv_offset += n;
                continue;
            }

            let face_start = vertices.len() as u32;

            for j in 0..n {
                let vi = face_vertex_indices[fv_offset + j] as usize;
                let position = if vi * 3 + 2 < points.len() {
                    [points[vi * 3], points[vi * 3 + 1], points[vi * 3 + 2]]
                } else {
                    [0.0, 0.0, 0.0]
                };

                let normal = if normals_face_varying {
                    normals
                        .as_ref()
                        .and_then(|n| {
                            let idx = fv_offset + j;
                            if idx * 3 + 2 < n.len() {
                                Some([n[idx * 3], n[idx * 3 + 1], n[idx * 3 + 2]])
                            } else {
                                None
                            }
                        })
                        .unwrap_or([0.0, 1.0, 0.0])
                } else {
                    normals
                        .as_ref()
                        .and_then(|n| {
                            if vi * 3 + 2 < n.len() {
                                Some([n[vi * 3], n[vi * 3 + 1], n[vi * 3 + 2]])
                            } else {
                                None
                            }
                        })
                        .unwrap_or([0.0, 1.0, 0.0])
                };

                let tex_coords = if uvs_face_varying {
                    uvs.as_ref()
                        .and_then(|u| {
                            let idx = fv_offset + j;
                            if idx * 2 + 1 < u.len() {
                                Some([u[idx * 2], u[idx * 2 + 1], 0.0])
                            } else {
                                None
                            }
                        })
                        .unwrap_or([0.0, 0.0, 0.0])
                } else {
                    uvs.as_ref()
                        .and_then(|u| {
                            if vi * 2 + 1 < u.len() {
                                Some([u[vi * 2], u[vi * 2 + 1], 0.0])
                            } else {
                                None
                            }
                        })
                        .unwrap_or([0.0, 0.0, 0.0])
                };

                vertices.push(Vertex {
                    position,
                    normal,
                    tex_coords,
                });
            }

            // Fan triangulate this face
            for j in 1..n - 1 {
                tri_indices.push(face_start);
                tri_indices.push(face_start + j as u32);
                tri_indices.push(face_start + j as u32 + 1);
            }

            fv_offset += n;
        }

        (vertices, tri_indices)
    }
}

// ============================================================================
// Camera Extraction
// ============================================================================

fn extract_camera(data: &mut dyn AbstractData, cam_path: &sdf::Path) -> Option<PositionedCamera> {
    let focal_length =
        get_float_from_prop(data, &make_property_path(cam_path, "focalLength"))?;
    let h_aperture =
        get_float_from_prop(data, &make_property_path(cam_path, "horizontalAperture"))
            .unwrap_or(36.0);
    let v_aperture =
        get_float_from_prop(data, &make_property_path(cam_path, "verticalAperture"))
            .unwrap_or(24.0);

    let fovy_rad = 2.0 * (v_aperture / (2.0 * focal_length)).atan();
    let fovy = fovy_rad.to_degrees();
    let aspect = h_aperture / v_aperture;

    // Read clipping planes
    let clipping_path = make_property_path(cam_path, "clippingRange");
    let (znear, zfar) = get_float_array(data, &clipping_path, "default")
        .and_then(|v| {
            if v.len() >= 2 {
                Some((v[0], v[1]))
            } else {
                None
            }
        })
        .unwrap_or((0.1, 10000.0));

    // Use transform position as eye point
    let Transform { position, .. } = extract_transform(data, cam_path);

    Some(PositionedCamera {
        eye: position,
        target: Point3::new(position.x, position.y, position.z - 1.0),
        up: Vector3::new(0.0, 1.0, 0.0),
        aspect,
        fovy,
        znear,
        zfar,
        ortho: false,
    })
}

// ============================================================================
// Light Extraction
// ============================================================================

fn extract_light(
    data: &mut dyn AbstractData,
    light_path: &sdf::Path,
    type_name: &str,
    position: &Point3,
    scene: &mut SceneData,
) {
    let intensity =
        get_float_from_prop(data, &make_property_path(light_path, "inputs:intensity"))
            .unwrap_or(1.0);

    let color_path = make_property_path(light_path, "inputs:color");
    let color = get_float_array(data, &color_path, "default")
        .and_then(|v| {
            if v.len() >= 3 {
                Some(RgbaColor { r: v[0], g: v[1], b: v[2], a: 1.0 })
            } else {
                None
            }
        })
        .unwrap_or(RgbaColor::WHITE);

    let (light, transform) = match type_name {
        "DistantLight" => {
            // Rotate so local -Z aligns with default downward direction (0, -1, 0)
            let rotation = Quaternion::from_angle_x(Deg(-90.0_f32));
            let t = Transform::new(Point3::new(0.0, 0.0, 0.0), rotation, Vector3::new(1.0, 1.0, 1.0));
            (Light::directional(color, intensity), t)
        }
        "SphereLight" | "DiskLight" | "RectLight" => {
            (Light::point(color, intensity), Transform::from_position(*position))
        }
        _ => return,
    };

    if let Ok(node) = scene.add_node(None, None, transform, NodeFlags::NONE) {
        scene.set_node_payload(node.id(), NodePayload::Light(light));
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_usd_extension() {
        assert!(is_usd_extension("usd"));
        assert!(is_usd_extension("usda"));
        assert!(is_usd_extension("usdc"));
        assert!(is_usd_extension("usdz"));
        assert!(is_usd_extension("USD"));
        assert!(is_usd_extension("USDZ"));
        assert!(!is_usd_extension("glb"));
        assert!(!is_usd_extension("fbx"));
    }

    #[test]
    fn test_triangulate_simple_triangle() {
        let points = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let counts = vec![3];
        let indices = vec![0, 1, 2];

        let (verts, tri_indices) = triangulate_mesh(&points, &None, &None, &counts, &indices);
        assert_eq!(verts.len(), 3);
        assert_eq!(tri_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_triangulate_quad() {
        let points = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let counts = vec![4];
        let indices = vec![0, 1, 2, 3];

        let (verts, tri_indices) = triangulate_mesh(&points, &None, &None, &counts, &indices);
        assert_eq!(verts.len(), 4);
        // Fan triangulation of quad → 2 triangles
        assert_eq!(tri_indices.len(), 6);
        assert_eq!(tri_indices, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn test_triangulate_mixed_faces() {
        // One triangle + one quad
        let points = vec![
            0.0, 0.0, 0.0, // 0
            1.0, 0.0, 0.0, // 1
            0.5, 1.0, 0.0, // 2
            2.0, 0.0, 0.0, // 3
            3.0, 0.0, 0.0, // 4
            3.0, 1.0, 0.0, // 5
            2.0, 1.0, 0.0, // 6
        ];
        let counts = vec![3, 4];
        let indices = vec![0, 1, 2, 3, 4, 5, 6];

        let (verts, tri_indices) = triangulate_mesh(&points, &None, &None, &counts, &indices);
        assert_eq!(verts.len(), 7);
        assert_eq!(tri_indices.len(), 9); // 1 tri + 2 tris from quad
    }

    #[test]
    fn test_triangulate_with_face_varying_uvs() {
        let points = vec![
            0.0, 0.0, 0.0, // 0
            1.0, 0.0, 0.0, // 1
            1.0, 1.0, 0.0, // 2
            0.0, 1.0, 0.0, // 3
        ];
        let counts = vec![4];
        let indices = vec![0, 1, 2, 3];
        // Face-varying UVs: 4 UV values (one per face vertex)
        let uvs = Some(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

        let (verts, tri_indices) = triangulate_mesh(&points, &None, &uvs, &counts, &indices);
        // Face-varying → expanded to 4 unique vertices
        assert_eq!(verts.len(), 4);
        assert_eq!(tri_indices.len(), 6);
        // Check UV assignment
        assert_eq!(verts[0].tex_coords, [0.0, 0.0, 0.0]);
        assert_eq!(verts[1].tex_coords, [1.0, 0.0, 0.0]);
        assert_eq!(verts[2].tex_coords, [1.0, 1.0, 0.0]);
        assert_eq!(verts[3].tex_coords, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_make_child_path() {
        let root = sdf::Path::abs_root();
        let child = make_child_path(&root, "World");
        assert_eq!(child.as_str(), "/World");

        let grandchild = make_child_path(&child, "Mesh");
        assert_eq!(grandchild.as_str(), "/World/Mesh");
    }

    #[test]
    fn test_make_property_path() {
        let prim = sdf::Path::new("/World/Mesh").unwrap();
        let prop = make_property_path(&prim, "points");
        assert_eq!(prop.as_str(), "/World/Mesh.points");
    }
}
