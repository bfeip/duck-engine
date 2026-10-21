//! Specular prefiltering by filtered importance sampling.

use std::f32::consts::PI;

use super::cube::{level_from_fn, CubeImage};
use super::sampling::{
    add, d_ggx, hammersley, importance_sample_ggx, scale, tangent_basis, to_world, Vec3,
};

/// GGX samples per texel. Each reads the environment's mip chain at the level
/// matching its share of the lobe, which stands in for many more samples.
const SAMPLE_COUNT: u32 = 64;

/// `environment` prefiltered into `levels` levels of roughness 0 to 1, the
/// first a copy of `environment`'s own.
///
/// `environment` needs its full mip chain.
pub(super) fn prefilter(environment: &CubeImage, levels: u32) -> CubeImage {
    let size = environment.size();
    let texel_solid_angle = 4.0 * PI / (6 * size * size) as f32;
    let mut out = vec![environment.level(0).to_vec()];
    for level in 1..levels {
        let samples = lobe(level as f32 / (levels - 1) as f32, texel_solid_angle);
        out.push(level_from_fn(size >> level, |normal| {
            let basis = tangent_basis(normal);
            let (color, weight) = samples.iter().fold(([0.0; 3], 0.0), |(color, weight), s| {
                let radiance = environment.sample(to_world(s.dir, basis, normal), s.lod);
                (add(color, scale(radiance, s.n_dot_l)), weight + s.n_dot_l)
            });
            scale(color, 1.0 / weight)
        }));
    }
    CubeImage::from_levels(size, out)
}

/// A reflected direction in tangent space (normal = view = +Z), its cosine
/// weight, and the environment level to read it from.
struct LobeSample {
    dir: Vec3,
    n_dot_l: f32,
    lod: f32,
}

/// The samples of a GGX lobe of `roughness` that land above the horizon.
fn lobe(roughness: f32, texel_solid_angle: f32) -> Vec<LobeSample> {
    (0..SAMPLE_COUNT)
        .filter_map(|i| {
            let h = importance_sample_ggx(hammersley(i, SAMPLE_COUNT), roughness);
            let n_dot_h = h[2];
            // Reflect the view (+Z) about h.
            let dir = [2.0 * n_dot_h * h[0], 2.0 * n_dot_h * h[1], 2.0 * n_dot_h * h[2] - 1.0];
            let n_dot_l = dir[2];
            (n_dot_l > 0.0).then(|| {
                // pdf(l) = D·(n·h) / (4·(v·h)), and v·h = n·h here.
                let pdf = d_ggx(n_dot_h, roughness) / 4.0;
                let sample_solid_angle = 1.0 / (SAMPLE_COUNT as f32 * pdf);
                let lod = 0.5 * (sample_solid_angle / texel_solid_angle).log2() + 1.0;
                LobeSample { dir, n_dot_l, lod: lod.max(0.0) }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    #[test]
    fn levels_halve_and_keep_uniform_radiance() {
        let environment = CubeImage::from_fn(16, |_| [0.25, 1.0, 3.0]).with_mips();
        let prefiltered = prefilter(&environment, 4);
        assert_eq!(prefiltered.level_count(), 4);
        for level in 0..4 {
            let texels = prefiltered.level(level);
            assert_eq!(texels.len(), 6 * (16usize >> level).pow(2));
            for texel in texels {
                assert!(
                    (0..3).all(|c| (texel[c] - [0.25, 1.0, 3.0][c]).abs() < EPSILON),
                    "level {level}: {texel:?}"
                );
            }
        }
    }

    #[test]
    fn rougher_levels_read_coarser_mips() {
        let texel_solid_angle = 4.0 * PI / (6 * 128 * 128) as f32;
        let mean_lod = |roughness| {
            let samples = lobe(roughness, texel_solid_angle);
            samples.iter().map(|s| s.lod).sum::<f32>() / samples.len() as f32
        };
        let lods = [0.25, 0.5, 0.75, 1.0].map(mean_lod);
        assert!(lods.windows(2).all(|w| w[0] < w[1]), "{lods:?}");
    }
}
