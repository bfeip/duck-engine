use crate::{Gpu, GpuTexture};

/// Every multisample count wgpu recognizes, highest first.
const SAMPLE_COUNTS: [u32; 5] = [16, 8, 4, 2, 1];

/// The highest multisample count `adapter` supports for every format in
/// `formats`. Always at least 1, which needs no support.
#[must_use]
pub fn highest_supported_sample_count(
    adapter: &wgpu::Adapter,
    formats: impl IntoIterator<Item = wgpu::TextureFormat>,
) -> u32 {
    let flags: Vec<_> = formats
        .into_iter()
        .map(|format| adapter.get_texture_format_features(format).flags)
        .collect();
    SAMPLE_COUNTS
        .into_iter()
        .find(|&count| flags.iter().all(|f| f.sample_count_supported(count)))
        .unwrap_or(1)
}

/// The fixed parameters of a render target: size, color format, MSAA level.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TargetConfig {
    pub size: (u32, u32),
    pub format: wgpu::TextureFormat,
    /// MSAA sample count (1 = no MSAA).
    pub sample_count: u32,
}

/// Which shared attachments a rendering stack needs. Everything is opt-in:
/// a workflow that needs no depth buffer pays for none.
#[derive(Clone, Copy, Default)]
pub struct TargetFeatures {
    pub depth: bool,
}

/// Shared size-dependent frame attachments: the depth buffer and, when MSAA
/// is active, the multisampled color attachment that resolves to the final
/// target. Recreated on [`resize`](Self::resize).
pub struct FrameTargets {
    config: TargetConfig,
    features: TargetFeatures,
    depth: Option<GpuTexture>,
    /// `Some` iff `sample_count > 1`.
    msaa_color: Option<GpuTexture>,
}

impl FrameTargets {
    #[must_use] 
    pub fn new(gpu: &Gpu, config: TargetConfig, features: TargetFeatures) -> Self {
        let mut targets = Self { config, features, depth: None, msaa_color: None };
        targets.create_attachments(gpu);
        targets
    }

    fn create_attachments(&mut self, gpu: &Gpu) {
        let (width, height) = self.config.size;
        self.depth = self.features.depth.then(|| {
            GpuTexture::depth(&gpu.device, width, height, self.config.sample_count, "depth_texture")
        });
        self.msaa_color = (self.config.sample_count > 1).then(|| {
            GpuTexture::color_attachment(
                &gpu.device, width, height, self.config.format, self.config.sample_count,
                "msaa_color_attachment",
            )
        });
    }

    /// Recreate all attachments at a new size. Ignores zero dimensions.
    pub fn resize(&mut self, gpu: &Gpu, size: (u32, u32)) {
        if size.0 == 0 || size.1 == 0 {
            return;
        }
        self.config.size = size;
        self.create_attachments(gpu);
    }

    #[must_use] 
    pub const fn config(&self) -> TargetConfig {
        self.config
    }

    #[must_use] 
    pub const fn size(&self) -> (u32, u32) {
        self.config.size
    }

    #[must_use] 
    pub const fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    #[must_use] 
    pub const fn sample_count(&self) -> u32 {
        self.config.sample_count
    }

    /// The shared depth buffer view for this frame.
    ///
    /// Pass this as the `depth_stencil_attachment` view in a render pass
    /// descriptor to depth-test against geometry drawn by earlier passes.
    ///
    /// # Panics
    ///
    /// Panics if depth was not requested in [`TargetFeatures`] at creation.
    #[must_use] 
    pub const fn depth_view(&self) -> &wgpu::TextureView {
        &self
            .depth
            .as_ref()
            .expect("FrameTargets created without TargetFeatures::depth")
            .view
    }

    /// Returns `(render_view, resolve_target)` for a render pass that may use MSAA.
    ///
    /// When MSAA is active, the pass should render into `render_view` (the
    /// multisampled attachment) and resolve into `target` (typically the
    /// swapchain). When MSAA is inactive, renders directly into `target` with
    /// no resolve step.
    #[must_use] 
    pub const fn color_views<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
    ) -> (&'a wgpu::TextureView, Option<&'a wgpu::TextureView>) {
        match &self.msaa_color {
            Some(msaa) => (&msaa.view, Some(target)),
            None => (target, None),
        }
    }
}
