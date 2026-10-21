//! Build script for the renderer crate.
//!
//! Bakes the BRDF integration lookup table with the library's own
//! `src/ibl/cpu/brdf.rs`: 512x512 texels of (scale, bias, 0, 1) as
//! little-endian f32. The renderer converts it to Rgba16Float at upload.

use std::env;
use std::fs;
use std::path::Path;

#[allow(dead_code)]
#[path = "src/ibl/cpu/sampling.rs"]
mod sampling;
#[path = "src/ibl/cpu/brdf.rs"]
mod brdf;

const LUT_SIZE: u32 = 512;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/ibl/cpu/sampling.rs");
    println!("cargo:rerun-if-changed=src/ibl/cpu/brdf.rs");

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("brdf_lut.bin");

    // 4 f32 channels per pixel, stored as little-endian bytes
    let bytes_per_pixel = 4 * 4; // 4 channels * 4 bytes per f32
    let total_bytes = (LUT_SIZE * LUT_SIZE) as usize * bytes_per_pixel;
    let mut data = vec![0u8; total_bytes];

    for y in 0..LUT_SIZE {
        for x in 0..LUT_SIZE {
            let n_dot_v = ((x as f32) + 0.5) / (LUT_SIZE as f32);
            let roughness = ((y as f32) + 0.5) / (LUT_SIZE as f32);
            let clamped_n_dot_v = n_dot_v.max(0.001);

            let (scale, bias) = brdf::integrate_brdf(clamped_n_dot_v, roughness);

            let pixel_offset = ((y * LUT_SIZE + x) as usize) * bytes_per_pixel;
            data[pixel_offset..pixel_offset + 4].copy_from_slice(&scale.to_le_bytes());
            data[pixel_offset + 4..pixel_offset + 8].copy_from_slice(&bias.to_le_bytes());
            data[pixel_offset + 8..pixel_offset + 12].copy_from_slice(&0.0f32.to_le_bytes());
            data[pixel_offset + 12..pixel_offset + 16].copy_from_slice(&1.0f32.to_le_bytes());
        }
    }

    fs::write(&out_path, &data).expect("Failed to write BRDF LUT");
}
