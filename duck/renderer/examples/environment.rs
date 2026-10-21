//! Headless render of spheres lit only by an environment map, baked once in
//! compute shaders and once on the CPU.
//!
//! Rows are metallic (top) and dielectric (bottom); roughness rises from 0 to
//! 1 left to right. Saves `environment_compute.png` and `environment_cpu.png`
//! and prints each bake's time and the difference between the two images. The
//! compute bake is skipped on adapters without compute shaders.
//!
//! Run with `cargo run --release --example environment -p duck-engine-renderer [HDR]`;
//! the HDR defaults to `assets/studio_small_09_4k.hdr`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use duck_engine_common::{Point3, Real, Vector3};
use duck_engine_renderer::ibl::load_hdr_from_path;
use duck_engine_renderer::scene::common::{RgbaColor, Transform};
use duck_engine_renderer::scene::resource::{FaceMaterial, Instance, Mesh, NodeFlags, PrimitiveType};
use duck_engine_renderer::scene::{PositionedCamera, Projection, Scene, SceneData};
use duck_engine_renderer::{Gpu, GpuCapabilities, GpuOptions, RenderContext, Renderer, SceneResources};

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 440;
const COLUMNS: usize = 5;

/// Two rows of spheres, roughness 0 → 1 across, under the environment at `hdr`.
fn spheres(hdr: &Path) -> anyhow::Result<SceneData> {
    let mut scene = SceneData::new();
    let mesh = scene.add_mesh(Mesh::sphere(0.42, 64, 32, PrimitiveType::TriangleList));
    let rows = [
        (1.0, RgbaColor { r: 0.95, g: 0.95, b: 0.95, a: 1.0 }),
        (0.0, RgbaColor { r: 0.8, g: 0.25, b: 0.2, a: 1.0 }),
    ];
    for (row, (metallic, color)) in rows.into_iter().enumerate() {
        for column in 0..COLUMNS {
            let roughness = column as f32 / (COLUMNS - 1) as f32;
            let material = scene.add_face_material(
                FaceMaterial::new()
                    .with_base_color_factor(color)
                    .with_metallic_factor(metallic)
                    .with_roughness_factor(roughness),
            );
            let position = Point3::new(column as Real - 2.0, 0.5 - row as Real, 0.0);
            scene.add_instance_node(
                None,
                Instance::new(mesh.clone()).with_face_material(material),
                Some(format!("sphere {row} {column}")),
                Transform::from_position(position),
                NodeFlags::NONE,
            )?;
        }
    }
    let environment = scene.add_environment_map_from_hdr_path(hdr);
    scene.set_active_environment_map(Some(environment));
    Ok(scene)
}

fn main() -> anyhow::Result<()> {
    let hdr = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        [env!("CARGO_MANIFEST_DIR"), "..", "..", "assets", "studio_small_09_4k.hdr"]
            .iter()
            .collect()
    });
    let (gpu, caps) = pollster::block_on(Gpu::headless(GpuOptions::default()))?;

    let start = Instant::now();
    load_hdr_from_path(&hdr)?;
    println!("Decoding {} alone: {:.0} ms", hdr.display(), start.elapsed().as_secs_f64() * 1e3);

    let camera = PositionedCamera {
        eye: Point3::new(0.0, 0.0, 6.0),
        target: Point3::new(0.0, 0.0, 0.0),
        up: Vector3::new(0.0, 1.0, 0.0),
        aspect: WIDTH as Real / HEIGHT as Real,
        projection: Projection::Perspective { fovy: 25.0, znear: 0.1, zfar: 100.0 },
    };

    let bakers = [("compute", caps), ("cpu", GpuCapabilities { has_compute: false, ..caps })];
    let mut images = Vec::new();
    for (name, caps) in bakers {
        if !caps.has_compute && name == "compute" {
            println!("compute: skipped, the adapter has no compute shaders");
            continue;
        }
        let mut ctx = RenderContext::new(gpu.clone(), wgpu::TextureFormat::Rgba8UnormSrgb, 1, caps);
        let mut shared = SceneResources::new(&ctx);
        let mut renderer = Renderer::new(&mut ctx, WIDTH, HEIGHT);
        let mut data = spheres(&hdr)?;

        let start = Instant::now();
        shared.prepare(&mut ctx, &mut data)?;
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        let elapsed = start.elapsed();

        let image = renderer.render_scene_to_image(
            &mut ctx,
            &mut shared,
            &mut Scene::new(data),
            &camera,
            &[],
            None,
        )?;
        let file = format!("environment_{name}.png");
        image.save(&file)?;
        println!("{name}: decode + bake + upload {:.0} ms, saved {file}", elapsed.as_secs_f64() * 1e3);
        images.push(image);
    }

    if let [compute, cpu] = &images[..] {
        let differences: Vec<u8> = compute
            .pixels()
            .zip(cpu.pixels())
            .flat_map(|(a, b)| (0..3).map(move |c| a[c].abs_diff(b[c])))
            .collect();
        let mean = differences.iter().map(|&d| d as f64).sum::<f64>() / differences.len() as f64;
        let max = differences.iter().max().copied().unwrap_or(0);
        let over = differences.iter().filter(|&&d| d > 8).count() as f64 / differences.len() as f64;
        println!(
            "compute vs cpu: mean {mean:.2}/255, max {max}/255, {:.2}% of channels differ by more than 8",
            over * 100.0
        );
    }
    Ok(())
}
