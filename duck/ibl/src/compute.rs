//! IBL baking in compute shaders.

mod equirect;
mod irradiance;
mod prefilter;

use equirect::EquirectToCubePipeline;
use irradiance::IrradiancePipeline;
use prefilter::PrefilterPipeline;

use crate::render_core::{Gpu, GpuTexture};
use crate::{HdrImage, IblTextures, CUBEMAP_FORMAT};

/// Face size of the cube the HDR is resampled onto before convolution.
const ENVIRONMENT_CUBEMAP_SIZE: u32 = 512;

/// The compute pipelines that bake an environment.
pub struct ComputeBaker {
    equirect: EquirectToCubePipeline,
    irradiance: IrradiancePipeline,
    prefilter: PrefilterPipeline,
}

impl ComputeBaker {
    /// Compile the pipelines.
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            equirect: EquirectToCubePipeline::new(&gpu.device),
            irradiance: IrradiancePipeline::new(&gpu.device),
            prefilter: PrefilterPipeline::new(&gpu.device),
        }
    }

    /// Bake the textures for an equirectangular environment.
    pub fn bake(&self, gpu: &Gpu, hdr: &HdrImage) -> IblTextures {
        let (device, queue) = (&gpu.device, &gpu.queue);
        let environment = self.equirect.convert(device, queue, hdr);
        IblTextures {
            irradiance: self.irradiance.generate(device, queue, &environment),
            prefiltered: self.prefilter.generate(device, queue, &environment),
        }
    }
}

/// One mip level of `cube` viewed as a six-layer array, for a compute shader
/// to write.
fn storage_view(cube: &GpuTexture, mip_level: u32) -> wgpu::TextureView {
    cube.texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("Cubemap Storage View Mip {}", mip_level)),
        format: Some(CUBEMAP_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: mip_level,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    })
}
