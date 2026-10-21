//! IBL baking in compute shaders.

mod equirect;
mod irradiance;
mod prefilter;

use equirect::EquirectToCubePipeline;
use irradiance::IrradiancePipeline;
use prefilter::PrefilterPipeline;

use crate::ibl::{HdrImage, IblTextures, CUBEMAP_FORMAT};
use crate::render_core::GpuTexture;

/// The compute pipelines that bake an environment.
pub(crate) struct ComputeBaker {
    equirect: EquirectToCubePipeline,
    irradiance: IrradiancePipeline,
    prefilter: PrefilterPipeline,
}

impl ComputeBaker {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            equirect: EquirectToCubePipeline::new(device),
            irradiance: IrradiancePipeline::new(device),
            prefilter: PrefilterPipeline::new(device),
        }
    }

    /// Bake the textures for an equirectangular environment.
    pub fn bake(&self, device: &wgpu::Device, queue: &wgpu::Queue, hdr: &HdrImage) -> IblTextures {
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
