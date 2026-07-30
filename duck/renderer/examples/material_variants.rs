//! Headless render of every surface-material permutation, one sphere each.
//!
//! Exercises the unified surface shader across its full matrix: lit/unlit, the
//! optional base-color / normal / metallic-roughness textures, and the
//! line/point primitives — each of which compiles a distinct WESL variant whose
//! bind-group layout is derived from the same config. There are no fallback
//! textures: a material binds exactly the textures it declares.
//!
//! Also covers alpha-mode resolution: the last two columns set only a
//! sub-1.0 color alpha and must come out blended, while column 0 is the same
//! blue at full alpha for comparison.
//!
//! Doubles as a smoke test — if any variant's shader or layout is wrong, the
//! render fails. Run with `cargo run --example material_variants -p duck-engine-renderer`.

use duck_engine_common::{Point3, Vector3};
use duck_engine_renderer::Renderer;
use duck_engine_renderer::scene::{
    AlphaMode, FaceMaterial, Instance, Light, LineMaterial, MaterialFlags, Mesh, NodePayload,
    PointMaterial, PositionedCamera, PrimitiveType, Scene, Texture, TextureId,
    common::{RgbaColor, Transform},
};
use duck_engine_scene::NodeFlags;

/// Add a 2×2 solid-color texture and return its id.
fn solid_texture(scene: &mut Scene, rgba: [u8; 4]) -> TextureId {
    let pixels: Vec<u8> = rgba.iter().copied().cycle().take(2 * 2 * 4).collect();
    scene.add_texture(Texture::from_rgba8(2, 2, pixels))
}

fn main() -> anyhow::Result<()> {
    let (width, height) = (640u32, 320u32);
    let mut renderer = pollster::block_on(Renderer::new_headless(width, height));

    let mut scene = Scene::new();
    let tris = scene.add_mesh(Mesh::sphere(0.35, 24, 16, PrimitiveType::TriangleList));
    let lines = scene.add_mesh(Mesh::sphere(0.35, 16, 10, PrimitiveType::LineList));
    let points = scene.add_mesh(Mesh::sphere(0.35, 12, 8, PrimitiveType::PointList));

    let base = solid_texture(&mut scene, [210, 90, 70, 255]);
    let normal = solid_texture(&mut scene, [128, 128, 255, 255]); // flat tangent-space normal
    let metal_rough = solid_texture(&mut scene, [255, 200, 40, 255]); // G=rough, B=metal

    // Each closure spawns one sphere of the given mesh at column `col` (centered).
    let place = |scene: &mut Scene, mesh, col: i32, name: &str| -> anyhow::Result<()> {
        let x = (col as f32 - 3.5) * 0.78;
        scene.add_instance_node(
            None,
            mesh,
            Some(name.to_string()),
            Transform::from_position(Point3::new(x, 0.0, 0.0)),
            NodeFlags::NONE,
        )?;
        Ok(())
    };

    // --- Lit face variants ---------------------------------------------------
    let m_factor = scene.add_face_material(
        FaceMaterial::new().with_base_color_factor(RgbaColor { r: 0.2, g: 0.6, b: 0.9, a: 1.0 }),
    );
    place(&mut scene, Instance::new(tris).with_face_material(m_factor), 0, "lit-factor")?;

    let m_base = scene.add_face_material(FaceMaterial::new().with_base_color_texture(base));
    place(&mut scene, Instance::new(tris).with_face_material(m_base), 1, "lit-base")?;

    let m_all = scene.add_face_material(
        FaceMaterial::new()
            .with_base_color_texture(base)
            .with_normal_texture(normal)
            .with_metallic_roughness_texture(metal_rough),
    );
    place(&mut scene, Instance::new(tris).with_face_material(m_all), 2, "lit-all-textures")?;

    // --- Unlit face: tinted base-color texture, blended (the "cursor" case) ---
    let m_unlit = scene.add_face_material(
        FaceMaterial::new()
            .with_base_color_texture(base)
            .with_base_color_factor(RgbaColor { r: 1.0, g: 1.0, b: 0.0, a: 1.0 })
            .with_alpha_mode(AlphaMode::Blend)
            .with_flags(MaterialFlags::DO_NOT_LIGHT | MaterialFlags::DOUBLE_SIDED),
    );
    place(&mut scene, Instance::new(tris).with_face_material(m_unlit), 3, "unlit-textured")?;

    // --- Line + point materials, with and without a base-color texture --------
    let line_plain = scene.add_line_material(LineMaterial::new(RgbaColor::WHITE));
    place(&mut scene, Instance::new(lines).with_line_material(line_plain), 4, "line-plain")?;

    let point_tex = scene
        .add_point_material(PointMaterial::new(RgbaColor::WHITE).with_base_color_texture(base));
    place(&mut scene, Instance::new(points).with_point_material(point_tex), 5, "point-textured")?;

    // --- Inferred blending: alpha alone, no explicit AlphaMode ---------------
    // Both leave the mode at the default `Auto`, so the sub-1.0 alpha is what
    // selects the blended pipeline. Compare column 6 against column 0, which is
    // the same blue at full alpha.
    let m_auto_blend = scene.add_face_material(
        FaceMaterial::new().with_base_color_factor(RgbaColor { r: 0.2, g: 0.6, b: 0.9, a: 0.3 }),
    );
    place(&mut scene, Instance::new(tris).with_face_material(m_auto_blend), 6, "auto-blend-face")?;

    let line_translucent =
        scene.add_line_material(LineMaterial::new(RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 0.3 }));
    place(&mut scene, Instance::new(lines).with_line_material(line_translucent), 7, "auto-blend-line")?;

    // A white directional light (its direction is the node's -Z axis).
    let light = scene
        .add_node(None, Some("Light".to_string()), Default::default(), NodeFlags::NONE)
        .unwrap();
    scene.set_node_payload(
        light,
        NodePayload::Light(Light::directional(RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }, 2.5)),
    );

    let camera = PositionedCamera {
        eye: Point3::new(0.0, 0.0, 4.0),
        target: Point3::new(0.0, 0.0, 0.0),
        up: Vector3::new(0.0, 1.0, 0.0),
        aspect: width as f32 / height as f32,
        fovy: 45.0,
        znear: 0.1,
        zfar: 100.0,
        ortho: false,
    };

    let image = renderer.render_scene_to_image(&camera, &mut scene, None)?;
    image.save("material_variants.png")?;
    println!("Saved material_variants.png ({width}×{height}) — all surface variants compiled");
    Ok(())
}
