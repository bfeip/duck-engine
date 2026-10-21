//! Diffuse irradiance by direct cosine convolution.

use std::f32::consts::PI;

use super::cube::{texel_direction, texel_solid_angle, texels, CubeImage, Rgb};
use super::sampling::{dot, scale, Vec3};

/// Face size of the environment level convolved.
const SOURCE_SIZE: u32 = 16;

/// The irradiance cube of `environment` with `size`² faces, divided by π so
/// that a uniform radiance L bakes to L.
pub(super) fn irradiance(environment: &CubeImage, size: u32) -> CubeImage {
    let level = environment.level_at_most(SOURCE_SIZE);
    let source_size = environment.level_size(level);
    // Each source texel as (direction, radiance × solid angle / π).
    let sources: Vec<(Vec3, Rgb)> = texels(source_size)
        .zip(environment.level(level))
        .map(|((face, x, y), &radiance)| {
            let weight = texel_solid_angle(source_size, x, y) / PI;
            (texel_direction(source_size, face, x, y), scale(radiance, weight))
        })
        .collect();

    CubeImage::from_fn(size, |normal| {
        sources.iter().fold([0.0; 3], |sum, &(dir, flux)| {
            let cos = dot(normal, dir);
            if cos > 0.0 {
                [sum[0] + flux[0] * cos, sum[1] + flux[1] * cos, sum[2] + flux[2] * cos]
            } else {
                sum
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-2;

    #[test]
    fn uniform_radiance_bakes_to_itself() {
        let environment = CubeImage::from_fn(32, |_| [0.5, 1.0, 2.0]).with_mips();
        let irradiance = irradiance(&environment, 8);
        for texel in irradiance.level(0) {
            assert!(
                (0..3).all(|c| (texel[c] / [0.5, 1.0, 2.0][c] - 1.0).abs() < EPSILON),
                "{texel:?}"
            );
        }
    }

    #[test]
    fn upper_hemisphere_lights_up_from_above() {
        let environment =
            CubeImage::from_fn(32, |d| if d[1] > 0.0 { [1.0; 3] } else { [0.0; 3] }).with_mips();
        let irradiance = irradiance(&environment, 8);
        for ((face, x, y), texel) in texels(8).zip(irradiance.level(0)) {
            // A uniformly lit upper hemisphere: E/π = (1 + n·y) / 2.
            let analytic = 0.5 * (1.0 + texel_direction(8, face, x, y)[1]);
            assert!(
                (texel[0] - analytic).abs() < 2.0 * EPSILON,
                "texel ({face}, {x}, {y}): {} vs {analytic}",
                texel[0]
            );
        }
    }
}
