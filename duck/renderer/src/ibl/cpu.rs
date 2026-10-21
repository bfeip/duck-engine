//! IBL baking on the CPU.

#[cfg(test)]
mod brdf;
mod cube;
mod equirect;
mod irradiance;
mod prefilter;
mod sampling;

pub use cube::{CubeImage, Rgb};

use crate::ibl::{HdrImage, IRRADIANCE_CUBEMAP_SIZE, PREFILTERED_CUBEMAP_SIZE, PREFILTERED_MIP_LEVELS};

/// The cubemaps PBR shading samples, baked on the CPU.
#[derive(Debug, Clone)]
pub struct IblMaps {
    /// Diffuse irradiance over π, [`IRRADIANCE_CUBEMAP_SIZE`]² faces.
    pub irradiance: CubeImage,
    /// Specular radiance, [`PREFILTERED_CUBEMAP_SIZE`]² faces, one level per
    /// roughness step over [`PREFILTERED_MIP_LEVELS`].
    pub prefiltered: CubeImage,
}

impl IblMaps {
    /// Bake the maps for an equirectangular environment.
    pub fn from_equirect(hdr: &HdrImage) -> Self {
        let environment =
            equirect::cube_from_equirect(hdr, PREFILTERED_CUBEMAP_SIZE).with_mips();
        Self {
            irradiance: irradiance::irradiance(&environment, IRRADIANCE_CUBEMAP_SIZE),
            prefiltered: prefilter::prefilter(&environment, PREFILTERED_MIP_LEVELS),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-2;

    #[test]
    fn uniform_environment_bakes_uniform_maps() {
        let hdr = HdrImage { width: 64, height: 32, data: [0.5, 1.0, 2.0].repeat(64 * 32) };
        let maps = IblMaps::from_equirect(&hdr);

        assert_eq!(maps.irradiance.size(), IRRADIANCE_CUBEMAP_SIZE);
        assert_eq!(maps.irradiance.level_count(), 1);
        assert_eq!(maps.prefiltered.size(), PREFILTERED_CUBEMAP_SIZE);
        assert_eq!(maps.prefiltered.level_count(), PREFILTERED_MIP_LEVELS);

        let levels = std::iter::once(maps.irradiance.level(0))
            .chain((0..PREFILTERED_MIP_LEVELS).map(|l| maps.prefiltered.level(l)));
        for texels in levels {
            for texel in texels {
                assert!(
                    (0..3).all(|c| (texel[c] / [0.5, 1.0, 2.0][c] - 1.0).abs() < EPSILON),
                    "{texel:?}"
                );
            }
        }
    }
}
