use std::path::Path;
use duck_engine_common::{InnerSpace, Matrix4, Point3, SquareMatrix, Vector3};
use duck_engine_scene::camera::PositionedCamera;
use duck_engine_scene::resource::{
    AlphaMode, FaceMaterial, FaceMaterialHandle, Instance, MaterialFlags, Mesh, MeshHandle,
    MeshPrimitive, NodeFlags, NodeId, PrimitiveType, Texture, TextureHandle, Vertex,
};
use duck_engine_scene::SceneData;

/// A loaded primitive from a glTF mesh: the scene mesh and its material.
type LoadedPrimitive = (MeshHandle, FaceMaterialHandle);

/// Maps glTF mesh indices to their loaded primitives.
///
/// Each glTF mesh can contain multiple primitives (e.g. different parts with different materials).
/// Each primitive is loaded as a separate scene mesh, so a single glTF mesh index maps to
/// potentially multiple (mesh, material) pairs.
type GltfMeshMap = Vec<Vec<LoadedPrimitive>>;

/// Result of loading a glTF scene, containing the scene and optional camera.
pub struct GltfLoadResult {
    /// The loaded scene with meshes, materials, and scene tree
    pub scene: SceneData,
    /// Camera from the glTF file, if one was defined
    pub camera: Option<PositionedCamera>,
}

/// Loads vertex data from a glTF primitive.
///
/// Reads positions, normals, and texture coordinates from the primitive's attributes.
fn load_vertices(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
) -> anyhow::Result<Vec<Vertex>> {
    // Create a reader for this primitive
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    // Read positions (required)
    let positions = reader
        .read_positions()
        .ok_or_else(|| anyhow::anyhow!("Primitive missing positions"))?;

    // Read normals (optional - lines and points don't need them)
    let normals = reader.read_normals();

    // Read texture coordinates (optional, default to [0, 0] if missing)
    let tex_coords = reader.read_tex_coords(0);

    // Build vertex array by zipping iterators
    let vertices = match (normals, tex_coords) {
        (Some(normals), Some(tex_coords)) => {
            // Has normals and texture coordinates
            let tex_coords_f32 = tex_coords.into_f32();
            positions
                .zip(normals)
                .zip(tex_coords_f32)
                .map(|((position, normal), tex_coords)| Vertex {
                    position,
                    normal,
                    tex_coords: [tex_coords[0], tex_coords[1], 0.0],
                })
                .collect()
        }
        (Some(normals), None) => {
            // Has normals but no texture coordinates
            positions
                .zip(normals)
                .map(|(position, normal)| Vertex {
                    position,
                    normal,
                    tex_coords: [0.0, 0.0, 0.0],
                })
                .collect()
        }
        (None, Some(tex_coords)) => {
            // Has texture coordinates but no normals
            let tex_coords_f32 = tex_coords.into_f32();
            positions
                .zip(tex_coords_f32)
                .map(|(position, tex_coords)| Vertex {
                    position,
                    normal: [0.0, 0.0, 0.0],
                    tex_coords: [tex_coords[0], tex_coords[1], 0.0],
                })
                .collect()
        }
        (None, None) => {
            // No normals or texture coordinates (lines/points)
            positions
                .map(|position| Vertex {
                    position,
                    normal: [0.0, 0.0, 0.0],
                    tex_coords: [0.0, 0.0, 0.0],
                })
                .collect()
        }
    };

    Ok(vertices)
}

/// Loads index data from a glTF primitive as u32 indices.
///
/// Returns u32 indices which are later converted to u16 (splitting the mesh
/// into chunks if the vertex count exceeds the u16 limit).
fn load_indices(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
) -> anyhow::Result<Vec<u32>> {
    // Create a reader for this primitive
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    // Read indices as u32 (will be converted to u16 later, with splitting if needed)
    let indices = reader
        .read_indices()
        .ok_or_else(|| anyhow::anyhow!("Primitive missing indices"))?
        .into_u32()
        .collect();

    Ok(indices)
}

/// Maps a glTF primitive mode to our PrimitiveType.
/// Returns None for unsupported modes.
fn map_primitive_mode(mode: gltf::mesh::Mode) -> Option<PrimitiveType> {
    match mode {
        gltf::mesh::Mode::Triangles => Some(PrimitiveType::TriangleList),
        gltf::mesh::Mode::Lines => Some(PrimitiveType::LineList),
        gltf::mesh::Mode::Points => Some(PrimitiveType::PointList),
        _ => None, // TriangleFan, TriangleStrip not supported
    }
}

/// Gets the material for a glTF primitive, returning `None` when no material applies.
fn get_material_for_primitive(
    primitive: &gltf::Primitive,
    primitive_type: PrimitiveType,
    material_map: &[FaceMaterialHandle],
) -> Option<FaceMaterialHandle> {
    match primitive_type {
        PrimitiveType::TriangleList => primitive
            .material()
            .index()
            .and_then(|idx| material_map.get(idx).cloned()),
        PrimitiveType::LineList | PrimitiveType::PointList => None,
    }
}

/// Tries to resolve an external URI image to a file path.
fn resolve_image_path(
    image_index: usize,
    document: &gltf::Document,
    base_path: Option<&Path>,
) -> Option<std::path::PathBuf> {
    let base = base_path?;
    let img = document.images().nth(image_index)?;
    let gltf::image::Source::Uri { uri, .. } = img.source() else { return None };

    // Skip data: URIs - those are embedded
    if uri.starts_with("data:") {
        return None;
    }

    let path = base.join(uri);
    path.exists().then_some(path)
}

/// Decodes glTF image pixel data into a DynamicImage.
fn decode_gltf_image(gltf_image: &gltf::image::Data) -> anyhow::Result<image::DynamicImage> {
    use gltf::image::Format;
    use image::DynamicImage;

    let w = gltf_image.width;
    let h = gltf_image.height;
    let pixels = &gltf_image.pixels;
    let make_err = || anyhow::anyhow!("Pixel data size mismatch for {:?} image", gltf_image.format);

    match gltf_image.format {
        Format::R8 => {
            let img = image::GrayImage::from_raw(w, h, pixels.clone()).ok_or_else(make_err)?;
            Ok(DynamicImage::ImageLuma8(img))
        }
        Format::R8G8 => {
            let img = image::GrayAlphaImage::from_raw(w, h, pixels.clone()).ok_or_else(make_err)?;
            Ok(DynamicImage::ImageLumaA8(img))
        }
        Format::R8G8B8 => {
            let img = image::RgbImage::from_raw(w, h, pixels.clone()).ok_or_else(make_err)?;
            Ok(DynamicImage::ImageRgb8(img))
        }
        Format::R8G8B8A8 => {
            let img = image::RgbaImage::from_raw(w, h, pixels.clone()).ok_or_else(make_err)?;
            Ok(DynamicImage::ImageRgba8(img))
        }
        Format::R16 => {
            let img = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(
                w, h, bytemuck::cast_vec(pixels.clone()),
            ).ok_or_else(make_err)?;
            Ok(DynamicImage::ImageLuma16(img))
        }
        Format::R16G16 => {
            let img = image::ImageBuffer::<image::LumaA<u16>, _>::from_raw(
                w, h, bytemuck::cast_vec(pixels.clone()),
            ).ok_or_else(make_err)?;
            Ok(DynamicImage::ImageLumaA16(img))
        }
        Format::R16G16B16 => {
            let img = image::ImageBuffer::<image::Rgb<u16>, _>::from_raw(
                w, h, bytemuck::cast_vec(pixels.clone()),
            ).ok_or_else(make_err)?;
            Ok(DynamicImage::ImageRgb16(img))
        }
        Format::R16G16B16A16 => {
            let img = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(
                w, h, bytemuck::cast_vec(pixels.clone()),
            ).ok_or_else(make_err)?;
            Ok(DynamicImage::ImageRgba16(img))
        }
        Format::R32G32B32FLOAT => {
            let img = image::ImageBuffer::<image::Rgb<f32>, _>::from_raw(
                w, h, bytemuck::cast_vec(pixels.clone()),
            ).ok_or_else(make_err)?;
            Ok(DynamicImage::ImageRgb32F(img))
        }
        Format::R32G32B32A32FLOAT => {
            let img = image::ImageBuffer::<image::Rgba<f32>, _>::from_raw(
                w, h, bytemuck::cast_vec(pixels.clone()),
            ).ok_or_else(make_err)?;
            Ok(DynamicImage::ImageRgba32F(img))
        }
    }
}

/// Loads an image from glTF image data and adds it as a texture to the scene.
///
/// External URI references become path-based textures; embedded images are
/// decoded into memory.
fn load_gltf_texture(
    gltf_image: &gltf::image::Data,
    image_index: usize,
    document: &gltf::Document,
    base_path: Option<&Path>,
    scene: &mut SceneData,
) -> anyhow::Result<TextureHandle> {
    // For external files, use path-based texture (lazy loading, preserves original format)
    if let Some(path) = resolve_image_path(image_index, document, base_path) {
        let texture = Texture::from_path(path);
        return Ok(scene.add_texture(texture));
    }

    let texture = Texture::from_image(decode_gltf_image(gltf_image)?);
    Ok(scene.add_texture(texture))
}

/// Loads a material from a glTF material definition.
///
/// Extracts full PBR data including base color, metallic-roughness, and normal maps.
/// Returns the handle of the created material.
fn load_material(
    gltf_material: &gltf::Material,
    images: &[gltf::image::Data],
    document: &gltf::Document,
    base_path: Option<&Path>,
    scene: &mut SceneData,
) -> anyhow::Result<FaceMaterialHandle> {
    let pbr = gltf_material.pbr_metallic_roughness();

    // Start with a new material
    let mut material = FaceMaterial::new();

    // Base color factor (always present, defaults to white in glTF)
    let base_color = pbr.base_color_factor();
    material = material.with_base_color_factor(duck_engine_scene::common::RgbaColor {
        r: base_color[0],
        g: base_color[1],
        b: base_color[2],
        a: base_color[3],
    });

    // Base color texture (optional)
    if let Some(tex_info) = pbr.base_color_texture() {
        let image_index = tex_info.texture().source().index();
        let texture_id = load_gltf_texture(&images[image_index], image_index, document, base_path, scene)?;
        material = material.with_base_color_texture(texture_id);
    }

    // Metallic and roughness factors
    material = material
        .with_metallic_factor(pbr.metallic_factor())
        .with_roughness_factor(pbr.roughness_factor());

    // Metallic-roughness texture (optional)
    // glTF packs roughness in G channel and metallic in B channel
    if let Some(tex_info) = pbr.metallic_roughness_texture() {
        let image_index = tex_info.texture().source().index();
        let texture_id = load_gltf_texture(&images[image_index], image_index, document, base_path, scene)?;
        material = material.with_metallic_roughness_texture(texture_id);
    }

    // Normal map (from material, not from PBR extension)
    if let Some(normal_tex) = gltf_material.normal_texture() {
        let image_index = normal_tex.texture().source().index();
        let texture_id = load_gltf_texture(&images[image_index], image_index, document, base_path, scene)?;
        material = material
            .with_normal_texture(texture_id)
            .with_normal_scale(normal_tex.scale());
    }

    // Double-sided rendering
    if gltf_material.double_sided() {
        let flags = material.flags() | MaterialFlags::DOUBLE_SIDED;
        material = material.with_flags(flags);
    }

    // Alpha mode. OPAQUE is set explicitly rather than left at the default:
    // the spec says the alpha channel is ignored, which is exactly what the
    // explicit mode means, whereas the default would infer blending from a
    // sub-1.0 base color factor.
    match gltf_material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => {
            material = material.with_alpha_mode(AlphaMode::Opaque);
        }
        gltf::material::AlphaMode::Mask => {
            material = material
                .with_alpha_mode(AlphaMode::Mask)
                .with_alpha_cutoff(gltf_material.alpha_cutoff().unwrap_or(0.5));
        }
        gltf::material::AlphaMode::Blend => {
            material = material.with_alpha_mode(AlphaMode::Blend);
        }
    }

    // Approximate KHR_materials_transmission by mapping transmission to alpha blending.
    // A correct implementation would require multi-pass rendering to capture the scene
    // behind the transmissive surface, so we map transmission to alpha instead.
    if let Some(transmission) = gltf_material.transmission() {
        let factor = transmission.transmission_factor();
        if factor > 0.0 {
            let mut color = material.base_color_factor();
            color.a *= f32::max(1.0 - factor, 0.1);
            material = material
                .with_base_color_factor(color)
                .with_alpha_mode(AlphaMode::Blend);
        }
    }

    Ok(scene.add_face_material(material))
}

/// Converts a glTF transform to a 4x4 matrix.
fn transform_to_matrix(transform: &gltf::scene::Transform) -> Matrix4 {
    Matrix4::from(transform.clone().matrix())
}

/// Decomposes a glTF transform into position, rotation, and scale.
fn decompose_transform(transform: &gltf::scene::Transform) -> duck_engine_scene::common::Transform {
    duck_engine_scene::common::decompose_matrix(&transform_to_matrix(transform))
}

/// Extracts camera data from a glTF camera node.
///
/// Returns a [`PositionedCamera`] with pose derived from `world_transform` if the
/// node contains a camera.
fn extract_camera_from_node(
    gltf_node: &gltf::Node,
    world_transform: Matrix4,
    aspect: f32,
) -> Option<PositionedCamera> {

    let gltf_camera = gltf_node.camera()?;

    // Extract camera position from world transform (translation column)
    let eye = Point3::new(
        world_transform[3][0],
        world_transform[3][1],
        world_transform[3][2],
    );

    // Extract forward direction from world transform.
    // In glTF, cameras look down -Z in their local space;
    // the third column of the rotation matrix is the local Z axis.
    let local_z = Vector3::new(
        world_transform[2][0],
        world_transform[2][1],
        world_transform[2][2],
    ).normalize();
    let forward = -local_z;

    // Extract up direction from world transform (Y axis)
    let up = Vector3::new(
        world_transform[1][0],
        world_transform[1][1],
        world_transform[1][2],
    ).normalize();

    // Target is 1 unit in front of the camera (focus_distance defaults to 1.0)
    let target = eye + forward;

    match gltf_camera.projection() {
        gltf::camera::Projection::Perspective(persp) => {
            let fovy = persp.yfov().to_degrees();
            let znear = persp.znear();
            let zfar = persp.zfar().unwrap_or(1000.0);

            Some(PositionedCamera {
                eye,
                target,
                up,
                aspect,
                fovy,
                znear,
                zfar,
                ortho: false,
            })
        }
        gltf::camera::Projection::Orthographic(ortho_cam) => {
            // glTF orthographic uses xmag/ymag as half-extents.
            // We derive an equivalent fovy from the distance and ymag:
            //   ymag = distance * tan(fovy/2)  →  fovy = 2 * atan(ymag / distance)
            let ymag = ortho_cam.ymag();
            let znear = ortho_cam.znear();
            let zfar = ortho_cam.zfar();
            let distance = (eye - target).magnitude();
            let fovy = if distance > 0.0 {
                2.0 * (ymag / distance).atan().to_degrees()
            } else {
                45.0
            };

            Some(PositionedCamera {
                eye,
                target,
                up,
                aspect,
                fovy,
                znear,
                zfar,
                ortho: true,
            })
        }
    }
}

/// Recursively searches for a camera in the glTF scene tree.
///
/// Returns the first camera found along with its world transform.
fn find_camera_recursive(
    gltf_node: &gltf::Node,
    parent_transform: Matrix4,
    aspect: f32,
) -> Option<PositionedCamera> {
    let local_transform = transform_to_matrix(&gltf_node.transform());
    let world_transform = parent_transform * local_transform;

    // Check if this node has a camera
    if let Some(camera) = extract_camera_from_node(gltf_node, world_transform, aspect) {
        return Some(camera);
    }

    // Recurse into children
    for child in gltf_node.children() {
        if let Some(camera) = find_camera_recursive(&child, world_transform, aspect) {
            return Some(camera);
        }
    }

    None
}

/// Loads a glTF scene from a file.
///
/// This is the main entry point for loading glTF files into the engine.
/// Returns a `GltfLoadResult` containing the scene and an optional camera
/// if one was defined in the glTF file.
///
/// # Arguments
/// * `path` - Path to the glTF file
/// * `aspect` - Aspect ratio to use for the camera (if found)
pub fn load_gltf_scene_from_path<P: AsRef<Path>>(
    path: P,
    aspect: f32,
) -> anyhow::Result<GltfLoadResult> {
    let path = path.as_ref();
    let base_path = path.parent().map(|p| p.to_path_buf());
    let (document, buffers, images) = gltf::import(path)?;
    load_gltf_from_data(document, buffers, images, base_path.as_deref(), aspect)
}

/// Loads a glTF scene from a byte slice.
///
/// This is useful for loading glTF data that has been read from memory,
/// such as when loading files in a web browser via JavaScript.
/// Returns a `GltfLoadResult` containing the scene and an optional camera
/// if one was defined in the glTF file.
///
/// # Arguments
/// * `data` - The glTF file contents as a byte slice
/// * `aspect` - Aspect ratio to use for the camera (if found)
pub fn load_gltf_scene_from_slice(
    data: &[u8],
    aspect: f32,
) -> anyhow::Result<GltfLoadResult> {
    let (document, buffers, images) = gltf::import_slice(data)?;
    load_gltf_from_data(document, buffers, images, None, aspect)
}

/// Parsed glTF data, ready for asset loading. Produced by [`parse_gltf`].
pub struct ParsedGltf {
    pub document: gltf::Document,
    pub buffers: Vec<gltf::buffer::Data>,
    pub images: Vec<gltf::image::Data>,
    pub base_path: Option<std::path::PathBuf>,
}

/// Phase 1: Parse glTF from bytes.
///
/// Validates the file and decodes embedded images. This is typically the most
/// expensive phase for glTF files with embedded textures.
pub fn parse_gltf(data: &[u8]) -> anyhow::Result<ParsedGltf> {
    let (document, buffers, images) = gltf::import_slice(data)?;
    Ok(ParsedGltf {
        document,
        buffers,
        images,
        base_path: None,
    })
}

/// Phase 1 (path variant): Parse glTF from a file path.
///
/// Like [`parse_gltf`] but resolves external texture references relative to
/// the file's directory.
pub fn parse_gltf_from_path(path: &Path) -> anyhow::Result<ParsedGltf> {
    let base_path = path.parent().map(|p| p.to_path_buf());
    let (document, buffers, images) = gltf::import(path)?;
    Ok(ParsedGltf {
        document,
        buffers,
        images,
        base_path,
    })
}

/// Phase 2: Load materials, textures, and meshes from parsed glTF data into a scene.
///
/// Creates all materials (with their textures) and meshes, returning the maps
/// needed by [`build_gltf_scene`] to wire up the node hierarchy.
pub fn load_gltf_assets(
    parsed: &ParsedGltf,
    scene: &mut SceneData,
) -> anyhow::Result<(Vec<FaceMaterialHandle>, GltfMeshMap)> {
    let base_path = parsed.base_path.as_deref();

    // Load all materials (which also loads their textures)
    let mut material_map: Vec<FaceMaterialHandle> = Vec::new();
    for material in parsed.document.materials() {
        let mat = load_material(&material, &parsed.images, &parsed.document, base_path, scene)?;
        material_map.push(mat);
    }

    // Load all meshes
    let mut mesh_map: GltfMeshMap = Vec::new();
    let mut fallback_material: Option<FaceMaterialHandle> = None;
    for mesh in parsed.document.meshes() {
        let mut primitives_data = Vec::new();

        for primitive in mesh.primitives() {
            let Some(primitive_type) = map_primitive_mode(primitive.mode()) else {
                log::warn!(
                    "Skipping unsupported primitive mode {:?} in mesh {}",
                    primitive.mode(),
                    mesh.name().unwrap_or("unnamed")
                );
                continue;
            };

            let vertices = load_vertices(&primitive, &parsed.buffers)?;
            let indices_u32 = load_indices(&primitive, &parsed.buffers)?;

            let material = get_material_for_primitive(&primitive, primitive_type, &material_map)
                .unwrap_or_else(|| {
                    fallback_material
                        .get_or_insert_with(|| scene.add_face_material(FaceMaterial::default()))
                        .clone()
                });

            let primitive = MeshPrimitive { primitive_type, indices: indices_u32 };
            let mesh = scene.add_mesh(Mesh::from_raw(vertices, vec![primitive]));
            primitives_data.push((mesh, material));
        }

        mesh_map.push(primitives_data);
    }

    Ok((material_map, mesh_map))
}

/// Phase 3: Build the scene hierarchy (nodes, camera, lights) from parsed glTF.
///
/// Requires the mesh map from [`load_gltf_assets`]. Returns the camera if one
/// was defined in the glTF file.
pub fn build_gltf_scene(
    parsed: &ParsedGltf,
    scene: &mut SceneData,
    mesh_map: &GltfMeshMap,
    aspect: f32,
) -> anyhow::Result<Option<PositionedCamera>> {
    use std::collections::HashMap;

    let mut camera: Option<PositionedCamera> = None;

    if let Some(gltf_scene) = parsed
        .document
        .default_scene()
        .or_else(|| parsed.document.scenes().next())
    {
        let mut node_map: HashMap<usize, duck_engine_scene::resource::NodeId> = HashMap::new();

        for gltf_node in gltf_scene.nodes() {
            load_node_recursive(&gltf_node, None, scene, mesh_map, &mut node_map)?;

            if camera.is_none() {
                camera = find_camera_recursive(
                    &gltf_node,
                    Matrix4::identity(),
                    aspect,
                );
            }
        }
    }

    Ok(camera)
}

/// Internal helper to load a glTF scene from parsed data.
fn load_gltf_from_data(
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
    base_path: Option<&Path>,
    aspect: f32,
) -> anyhow::Result<GltfLoadResult> {
    let parsed = ParsedGltf {
        document,
        buffers,
        images,
        base_path: base_path.map(|p| p.to_path_buf()),
    };

    let mut scene = SceneData::new();
    let (_material_map, mesh_map) = load_gltf_assets(&parsed, &mut scene)?;
    let camera = build_gltf_scene(&parsed, &mut scene, &mesh_map, aspect)?;

    Ok(GltfLoadResult { scene, camera })
}

/// Recursively loads a glTF node and its children.
fn load_node_recursive(
    gltf_node: &gltf::Node,
    parent: Option<duck_engine_scene::resource::NodeId>,
    scene: &mut SceneData,
    mesh_map: &[Vec<LoadedPrimitive>],
    node_map: &mut std::collections::HashMap<usize, NodeId>,
) -> anyhow::Result<()> {

    // Decompose transform
    let transform = decompose_transform(&gltf_node.transform());

    // Extract node name from glTF
    let name = gltf_node.name().map(|s| s.to_string());

    // Create node(s) and track the parent node ID for child recursion
    let node_id = if let Some(mesh) = gltf_node.mesh() {
        let primitives = &mesh_map[mesh.index()];

        if primitives.is_empty() {
            // This is likely composed solely of primitives we don't support, like triangle fan
            scene.add_node(parent, name, transform, NodeFlags::NONE)?.id()
        } else if primitives.len() > 1 {
            // Multi-primitive mesh: create parent node and child nodes for each primitive
            let parent_node_id = scene.add_node(parent, name, transform, NodeFlags::NONE)?.id();

            // Create child nodes for each primitive with identity transform
            for (mesh, material) in primitives {
                scene.add_instance_node(
                    Some(parent_node_id),
                    Instance::new(mesh.clone()).with_face_material(material.clone()),
                    None,
                    duck_engine_scene::common::Transform::IDENTITY,
                    NodeFlags::NONE
                )?;
            }

            parent_node_id
        } else {
            // Single primitive: create a single instance node
            let (mesh, material) = &primitives[0];
            let instance = Instance::new(mesh.clone()).with_face_material(material.clone());
            scene.add_instance_node(parent, instance, name, transform, NodeFlags::NONE)?.id()
        }
    } else {
        // No mesh, just a transform node
        scene.add_node(parent, name, transform, NodeFlags::NONE)?.id()
    };

    node_map.insert(gltf_node.index(), node_id);

    // Recurse for all children
    for child in gltf_node.children() {
        load_node_recursive(&child, Some(node_id), scene, mesh_map, node_map)?;
    }

    Ok(())
}