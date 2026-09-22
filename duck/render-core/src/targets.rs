use crate::{Gpu, GpuTexture, MaskChannels};

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

/// What an auxiliary attachment holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AuxKind {
    Color(wgpu::TextureFormat),
    Mask(MaskChannels),
    Depth,
}

impl AuxKind {
    #[must_use]
    pub const fn format(self) -> wgpu::TextureFormat {
        match self {
            Self::Color(format) => format,
            Self::Mask(channels) => channels.format(),
            Self::Depth => GpuTexture::DEPTH_FORMAT,
        }
    }
}

/// A named attachment a pass needs beyond the shared depth and color targets.
///
/// This is how one pass hands a texture to another: both declare the same
/// target, the host allocates exactly one for that name and resizes it with
/// the frame, and neither pass owns or rebuilds it. Two passes naming the same
/// target must describe it identically, or the workflow rejects the second one
/// with [`AuxConflict`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AuxTarget {
    pub name: &'static str,
    pub kind: AuxKind,
    /// Allocate at the frame's sample count rather than single-sampled.
    pub multisampled: bool,
    /// When multisampled, also allocate a single-sample texture to resolve
    /// into. Color and mask only — depth attachments cannot be resolved.
    pub resolved: bool,
    /// The readable texture needs `TEXTURE_BINDING`.
    pub sampled: bool,
}

impl AuxTarget {
    #[must_use]
    pub const fn new(name: &'static str, kind: AuxKind) -> Self {
        Self { name, kind, multisampled: false, resolved: false, sampled: false }
    }

    /// Allocate at the frame's sample count, with no resolve target.
    #[must_use]
    pub fn with_multisampled(mut self) -> Self {
        self.multisampled = true;
        self
    }

    /// Allocate at the frame's sample count and resolve into a single-sample
    /// texture, which is then what [`AuxAttachment::sampled_view`] returns.
    #[must_use]
    pub fn with_resolved(mut self) -> Self {
        self.multisampled = true;
        self.resolved = true;
        self
    }

    /// Make the readable texture bindable.
    #[must_use]
    pub fn with_sampled(mut self) -> Self {
        self.sampled = true;
        self
    }
}

/// Two passes declared the same auxiliary target with different descriptions.
#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("conflicting declarations for auxiliary target '{name}': {first:?} vs {second:?}")]
pub struct AuxConflict {
    pub name: &'static str,
    pub first: AuxTarget,
    pub second: AuxTarget,
}

/// Which shared attachments a rendering stack needs. Everything is opt-in:
/// a workflow that needs no depth buffer pays for none.
///
/// Each pass declares its own via
/// [`Pass::target_features`](crate::Pass::target_features); a
/// [`Workflow`](crate::Workflow) keeps the [`union`](Self::union) of its
/// passes' declarations, and the host allocates from that.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct TargetFeatures {
    depth: bool,
    sampled_depth: bool,
    aux: Vec<AuxTarget>,
}

impl TargetFeatures {
    /// No attachments at all — for a pass that only reads.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// The shared depth buffer.
    #[must_use]
    pub fn depth() -> Self {
        Self { depth: true, ..Self::default() }
    }

    #[must_use]
    pub fn with_depth(mut self) -> Self {
        self.depth = true;
        self
    }

    /// The depth attachment must also be bindable, for passes that read it.
    /// Not every backend can — see
    /// [`GpuCapabilities::samples_depth_textures`](crate::GpuCapabilities::samples_depth_textures)
    /// — so [`FrameTargets::sampled_depth_view`] reports what the targets
    /// actually got, and a pass that needs it should skip itself when it is
    /// `None`.
    #[must_use]
    pub fn with_sampled_depth(mut self) -> Self {
        self.depth = true;
        self.sampled_depth = true;
        self
    }

    #[must_use]
    pub fn with_aux(mut self, target: AuxTarget) -> Self {
        self.aux.push(target);
        self
    }

    #[must_use]
    pub const fn needs_depth(&self) -> bool {
        self.depth
    }

    #[must_use]
    pub const fn needs_sampled_depth(&self) -> bool {
        self.sampled_depth
    }

    #[must_use]
    pub fn aux_targets(&self) -> &[AuxTarget] {
        &self.aux
    }

    /// Merge `other` into `self`: flags OR together, auxiliary targets are
    /// deduplicated by name.
    ///
    /// An identical redeclaration is the normal case — a producing pass and its
    /// consumer both name the target they share.
    ///
    /// # Errors
    ///
    /// Returns [`AuxConflict`] when both sides name one target differently.
    pub fn union(mut self, other: &Self) -> Result<Self, AuxConflict> {
        self.depth |= other.depth;
        self.sampled_depth |= other.sampled_depth;
        for target in &other.aux {
            match self.aux.iter().find(|existing| existing.name == target.name) {
                Some(existing) if existing != target => {
                    return Err(AuxConflict {
                        name: target.name,
                        first: *existing,
                        second: *target,
                    });
                }
                Some(_) => {}
                None => self.aux.push(*target),
            }
        }
        Ok(self)
    }
}

/// One allocated auxiliary attachment, owned by [`FrameTargets`] and shared by
/// every pass that declared it.
pub struct AuxAttachment {
    desc: AuxTarget,
    texture: GpuTexture,
    /// `Some` only when the target resolves and MSAA is active.
    resolve: Option<GpuTexture>,
}

impl AuxAttachment {
    fn new(device: &wgpu::Device, desc: AuxTarget, size: (u32, u32), frame_samples: u32) -> Self {
        let (width, height) = size;
        let samples = if desc.multisampled { frame_samples } else { 1 };
        let resolving = desc.resolved && samples > 1;

        let make = |count: u32, sampled: bool, label: &str| match desc.kind {
            AuxKind::Depth => GpuTexture::depth(device, width, height, count, sampled, label),
            kind => {
                GpuTexture::attachment(device, width, height, kind.format(), count, sampled, label)
            }
        };

        // A multisampled attachment that resolves is a resolve source only; the
        // resolve target is what carries TEXTURE_BINDING.
        let texture = make(samples, desc.sampled && !resolving, desc.name);
        let resolve = resolving
            .then(|| make(1, desc.sampled, &format!("{} (resolved)", desc.name)));

        Self { desc, texture, resolve }
    }

    /// `(attachment, resolve_target)` for a render pass color attachment,
    /// mirroring [`FrameTargets::color_views`].
    #[must_use]
    pub fn attachment(&self) -> (&wgpu::TextureView, Option<&wgpu::TextureView>) {
        (&self.texture.view, self.resolve.as_ref().map(|t| &t.view))
    }

    /// The attachment view itself, ignoring any resolve target. This is what a
    /// depth-stencil attachment binds.
    #[must_use]
    pub const fn view(&self) -> &wgpu::TextureView {
        &self.texture.view
    }

    /// The view to sample: the resolve target when this attachment resolves,
    /// otherwise the attachment itself.
    #[must_use]
    pub fn sampled_view(&self) -> &wgpu::TextureView {
        self.resolve.as_ref().map_or(&self.texture.view, |t| &t.view)
    }

    /// A sampler for [`sampled_view`](Self::sampled_view).
    #[must_use]
    pub fn sampler(&self) -> &wgpu::Sampler {
        self.resolve.as_ref().map_or(&self.texture.sampler, |t| &t.sampler)
    }

    /// How this attachment was declared.
    #[must_use]
    pub const fn desc(&self) -> &AuxTarget {
        &self.desc
    }
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
    /// One entry per distinct name in `features`, in declaration order.
    aux: Vec<AuxAttachment>,
}

impl FrameTargets {
    #[must_use]
    pub fn new(gpu: &Gpu, config: TargetConfig, features: TargetFeatures) -> Self {
        let mut targets =
            Self { config, features, depth: None, msaa_color: None, aux: Vec::new() };
        targets.create_attachments(gpu);
        targets
    }

    fn create_attachments(&mut self, gpu: &Gpu) {
        let (width, height) = self.config.size;
        self.depth = self.features.depth.then(|| {
            GpuTexture::depth(
                &gpu.device,
                width,
                height,
                self.config.sample_count,
                self.features.sampled_depth,
                "depth_texture",
            )
        });
        self.msaa_color = (self.config.sample_count > 1).then(|| {
            GpuTexture::color_attachment(
                &gpu.device, width, height, self.config.format, self.config.sample_count,
                "msaa_color_attachment",
            )
        });
        self.aux = self
            .features
            .aux
            .iter()
            .map(|desc| {
                AuxAttachment::new(&gpu.device, *desc, self.config.size, self.config.sample_count)
            })
            .collect();
    }

    /// The attachments this frame was built for.
    #[must_use]
    pub const fn features(&self) -> &TargetFeatures {
        &self.features
    }

    /// Reallocate every attachment against a new set of features, keeping the
    /// current size and format.
    pub fn set_features(&mut self, gpu: &Gpu, features: TargetFeatures) {
        self.features = features;
        self.create_attachments(gpu);
    }

    /// The auxiliary attachment declared under `name`, or `None` when no pass
    /// in the active workflow declared one.
    #[must_use]
    pub fn aux(&self, name: &str) -> Option<&AuxAttachment> {
        self.aux.iter().find(|a| a.desc.name == name)
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

    /// The depth view for passes that *read* depth, or `None` when the depth
    /// attachment was not created bindable. Such a pass has no depth to read
    /// and should skip itself.
    #[must_use]
    pub fn sampled_depth_view(&self) -> Option<&wgpu::TextureView> {
        if !self.features.sampled_depth {
            return None;
        }
        self.depth.as_ref().map(|depth| &depth.view)
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
