mod batching;
mod bind_group_layouts;
mod custom_pipeline;
mod lights;
mod material_cache;
mod mesh;
pub mod pass;
mod pass_context;
mod pipeline;
mod prepare;
mod scene_bindings;
mod surface_config;
mod texture;
pub mod workflow;

pub use batching::{
    BatchKey, BatchMaterial, DrawBatch, DrawData, InstanceTransform, SubGeomBatch,
};
pub use bind_group_layouts::BindGroupLayouts;
pub use custom_pipeline::CustomPipelineBuilder;
pub use mesh::{instance_buffer_layout, vertex_buffer_layout};
pub use pass_context::{DrawOptions, SceneFrame, SceneFrames, ScenePass, SceneWorkflow};
pub use pipeline::PipelineCacheKey;
pub use scene_bindings::SceneBindingRefs;
pub use surface_config::{MaterialTextureSlot, SurfaceConfig, TexturePresence};
pub use workflow::HiddenLineConfig;

use anyhow::Result;

use crate::{
    highlight_query::HighlightQuery,
    ibl::IblResources,
    render_core::{
        GenCache, Gpu, GpuCapabilities, GpuTexture, MaskChannels, RenderHost, TargetConfig,
        WorkflowGuard,
        highest_supported_sample_count,
    },
    rgba_to_wgpu_color,
    scene::{
        PositionedCamera,
        PositionedLight,
        Scene,
        SceneData,
        SceneProperties,
        common::RgbaColor,
        resource::{MeshId, TextureId}
    },
    shaders::ShaderGenerator
};

use material_cache::MaterialCache;
use mesh::MeshGpuResources;
use pipeline::MaterialPipelineCache;
use scene_bindings::{CameraBinding, LightsBinding};

/// The color format scene passes render at, given a final presentation format.
///
/// Scene shaders write linear color and rely on the target to apply the sRGB
/// transfer encode, so the scene target is always the sRGB variant. Formats with
/// no sRGB variant are returned unchanged.
pub fn scene_color_format(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    format.add_srgb_suffix()
}

/// Device-scoped render state: the GPU handles, the target configuration, and
/// the caches that depend only on those — bind group layouts, material
/// pipelines, and the WESL shader generator.
///
/// One `RenderContext` serves every scene and every view rendering at its
/// format and sample count; nothing here is per-scene, so scenes drawn at the
/// same target configuration share one set of compiled pipelines.
///
/// Create the per-scene caches from it with [`SceneResources::new`], and the
/// per-view state with [`Renderer::new`].
pub struct RenderContext {
    gpu: Gpu,
    format: wgpu::TextureFormat,
    sample_count: u32,
    capabilities: GpuCapabilities,

    layouts: BindGroupLayouts,
    /// Material pipelines, their layouts, and the shader generator.
    pipelines: MaterialPipelineCache,
}

impl RenderContext {
    /// Creates the render context for one device and target configuration.
    ///
    /// `format` and `sample_count` are baked into every pipeline built here, so
    /// all renderers over this context render at that configuration;
    /// [`Renderer::preferred_sample_count`] probes a suitable count. `format` is
    /// promoted to its sRGB variant by [`scene_color_format`].
    /// `capabilities` are the adapter's, as returned by
    /// [`Gpu`](crate::render_core::Gpu) acquisition; they decide which
    /// attachments and passes are available.
    pub fn new(
        gpu: Gpu,
        format: wgpu::TextureFormat,
        sample_count: u32,
        capabilities: GpuCapabilities,
    ) -> Self {
        let format = scene_color_format(format);
        let layouts = BindGroupLayouts::new(&gpu.device);
        let pipelines =
            MaterialPipelineCache::new(&layouts, ShaderGenerator::new(), sample_count, format);

        Self { gpu, format, sample_count, capabilities, layouts, pipelines }
    }

    /// The GPU handle pair, cloneable for sharing the device/queue.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.gpu.device
    }

    /// The wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.gpu.queue
    }

    /// The target texture format every renderer over this context draws at.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// The MSAA sample count every renderer over this context draws at.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// What the adapter behind this context can do.
    pub fn capabilities(&self) -> GpuCapabilities {
        self.capabilities
    }

    /// Whether the device supports compute shaders; without it environment maps
    /// are not processed.
    pub fn has_compute(&self) -> bool {
        self.capabilities.has_compute
    }

    /// Compile a user-supplied WESL shader with access to all engine shader modules.
    ///
    /// Engine modules available for import: `package::common`, `package::camera`,
    /// `package::lighting`, `package::constants`, `package::vertex`, `package::pbr`.
    pub fn compile_user_wesl(&self, source: &str) -> anyhow::Result<wgpu::ShaderModule> {
        crate::shaders::compile_user_wesl(&self.gpu.device, source)
    }

    /// Create a pipeline builder pre-configured with the engine's standard vertex
    /// and instance buffer layouts, target format, and MSAA sample count.
    ///
    /// Camera (group 0) and lights (group 1) bind group layouts are included by
    /// default. See [`CustomPipelineBuilder`] for the full configuration API.
    pub fn custom_pipeline_builder(&self) -> CustomPipelineBuilder<'_> {
        CustomPipelineBuilder::new(
            &self.gpu.device,
            self.format,
            self.sample_count,
            &self.layouts.camera,
            &self.layouts.light,
        )
    }

    /// The shared bind group layouts: camera, lights, flat color, and IBL.
    pub fn layouts(&self) -> &BindGroupLayouts {
        &self.layouts
    }

    /// Everything needed to construct a render pass, for a target of `size`.
    ///
    /// This is the one interface every pass constructor takes — built-in and
    /// user-written alike. [`Renderer::pass_builder`] is the usual way to get
    /// one, since it supplies its own target configuration.
    pub fn pass_builder(&mut self, size: (u32, u32)) -> PassBuilder<'_> {
        let config = TargetConfig { size, format: self.format, sample_count: self.sample_count };
        PassBuilder { ctx: self, config }
    }
}

/// Everything a render pass constructor needs: the device, the target
/// configuration it will render at, the adapter's capabilities, the shared bind
/// group layouts, and the engine shader library.
///
/// Obtained from [`Renderer::pass_builder`] or [`RenderContext::pass_builder`].
/// Passing one builder rather than a handful of positional arguments is what
/// lets built-in and user-written passes be constructed the same way, and so
/// composed in the same [`SceneWorkflow`].
pub struct PassBuilder<'a> {
    ctx: &'a mut RenderContext,
    config: TargetConfig,
}

impl PassBuilder<'_> {
    /// The GPU handle pair.
    pub fn gpu(&self) -> &Gpu {
        &self.ctx.gpu
    }

    /// The wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.ctx.gpu.device
    }

    /// The wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.ctx.gpu.queue
    }

    /// Size, color format and sample count the pass will render at.
    pub fn config(&self) -> TargetConfig {
        self.config
    }

    /// The target color format. Pipelines built here must use it.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// The MSAA sample count (1 = no MSAA), baked into every pipeline.
    pub fn sample_count(&self) -> u32 {
        self.config.sample_count
    }

    /// What the adapter can do. A pass whose pipeline needs an optional
    /// capability should check here and build no pipeline without it, then skip
    /// itself from [`Pass::is_active`](crate::render_core::Pass::is_active).
    pub fn capabilities(&self) -> GpuCapabilities {
        self.ctx.capabilities
    }

    /// The shared bind group layouts: camera, lights, flat color, and IBL.
    pub fn layouts(&self) -> &BindGroupLayouts {
        &self.ctx.layouts
    }

    /// The engine shader library and the device, for compiling a built-in
    /// shader variant:
    ///
    /// ```ignore
    /// let (shaders, device) = builder.shaders();
    /// let module = shaders.generate_outline_mask_shader(device)?;
    /// ```
    ///
    /// Handed back as a pair because generating a variant needs both, and
    /// borrowing them from the builder one at a time would alias.
    pub fn shaders(&mut self) -> (&mut ShaderGenerator, &wgpu::Device) {
        (self.ctx.pipelines.shader_generator_mut(), &self.ctx.gpu.device)
    }

    /// Compile a user-supplied WESL shader against the engine shader modules.
    ///
    /// # Errors
    ///
    /// Returns the WESL compilation error.
    pub fn compile_wesl(&self, source: &str) -> Result<wgpu::ShaderModule> {
        crate::shaders::compile_user_wesl(&self.ctx.gpu.device, source)
    }

    /// A pipeline builder pre-configured with this target's format and sample
    /// count and the engine's standard vertex and instance buffer layouts.
    pub fn pipeline(&self) -> CustomPipelineBuilder<'_> {
        CustomPipelineBuilder::new(
            &self.ctx.gpu.device,
            self.config.format,
            self.config.sample_count,
            &self.ctx.layouts.camera,
            &self.ctx.layouts.light,
        )
    }
}

/// GPU state for one scene: the resource caches generation-synced to a
/// [`SceneData`] — meshes, textures, material bind groups, and processed
/// environment maps.
///
/// One `SceneResources` serves any number of [`Renderer`]s (views) of the same
/// scene; create one per distinct scene. It owns the destructive
/// [`prepare`](Self::prepare), which must run exactly once per scene per frame.
///
/// Nothing here depends on the target configuration — pipelines and layouts
/// live in the [`RenderContext`] this was created from, and every call that
/// touches them takes it back.
pub struct SceneResources {
    /// Per-material bind groups, generation-synced to the scene.
    materials: MaterialCache,
    ibl_resources: IblResources,

    // Per-object geometry GPU resources, generation-synced to the scene.
    gpu_meshes: GenCache<MeshId, MeshGpuResources>,
    gpu_textures: GenCache<TextureId, GpuTexture>,
}

impl SceneResources {
    /// Creates the GPU caches for one scene, drawn through `ctx`.
    pub fn new(ctx: &RenderContext) -> Self {
        let ibl_resources = IblResources::new(
            &ctx.gpu.device,
            &ctx.gpu.queue,
            &ctx.layouts.ibl,
            ctx.capabilities.has_compute,
        );

        Self {
            materials: MaterialCache::new(),
            ibl_resources,
            gpu_meshes: GenCache::new(),
            gpu_textures: GenCache::new(),
        }
    }

    /// Clear all scene-specific GPU resources.
    ///
    /// Call this when the scene is cleared or replaced to ensure stale GPU
    /// buffers (vertex data, textures, material bind groups) are not reused.
    /// Pipelines are not scene state and are retained.
    pub fn clear(&mut self) {
        self.gpu_meshes.clear();
        self.gpu_textures.clear();
        self.materials.clear();
    }
}

/// Per-view render state: the frame targets (depth/MSAA), the active workflow,
/// and the camera and lights uniforms for one view of a scene.
///
/// The device, target configuration, and pipelines live in [`RenderContext`];
/// the scene's resource caches live in [`SceneResources`], shared by every
/// renderer over the same scene. The renderer borrows both per call.
pub struct Renderer {
    /// Core dispatch: owns the GPU handles, frame targets, the active
    /// workflow, and headless readback.
    host: RenderHost<SceneFrames>,
    /// This view's camera uniform and bind group.
    camera: CameraBinding,
    /// This view's lights uniform: scene lights plus the view's extra
    /// (camera-space) lights, re-uploaded every frame.
    lights: LightsBinding,
    background_color: wgpu::Color,
}

impl Renderer {
    /// The highest MSAA sample count this renderer can use on `adapter` with
    /// `surface_format`, or 1 when multisampling is unavailable.
    #[must_use]
    pub fn preferred_sample_count(
        adapter: &wgpu::Adapter,
        surface_format: wgpu::TextureFormat,
    ) -> u32 {
        let formats = [scene_color_format(surface_format), GpuTexture::DEPTH_FORMAT]
            .into_iter()
            .chain(MaskChannels::ALL.into_iter().map(MaskChannels::format));
        highest_supported_sample_count(adapter, formats)
    }

    /// Create a renderer at the given target size, drawing through `ctx`.
    ///
    /// Format and sample count come from `ctx` — every renderer over a
    /// `RenderContext` renders with the same pipeline configuration. A renderer
    /// is not bound to a scene: pass the scene's [`SceneResources`] per render
    /// call.
    pub fn new(ctx: &mut RenderContext, width: u32, height: u32) -> Self {
        let workflow = workflow::shaded(&mut ctx.pass_builder((width, height)));
        Self::with_workflow(ctx, width, height, workflow)
    }

    /// Create a renderer running `workflow` from the start, rather than
    /// installing the default one and immediately replacing it.
    ///
    /// Build the workflow from this context's [`pass_builder`](RenderContext::pass_builder)
    /// at the same size.
    pub fn with_workflow(
        ctx: &mut RenderContext,
        width: u32,
        height: u32,
        workflow: SceneWorkflow,
    ) -> Self {
        let config = TargetConfig {
            size: (width, height),
            format: ctx.format,
            sample_count: ctx.sample_count,
        };

        let camera = CameraBinding::new(&ctx.gpu.device, &ctx.layouts.camera);
        let lights = LightsBinding::new(&ctx.gpu.device, &ctx.layouts.light);

        // Attachments come from the workflow's passes, so a workflow that needs
        // no readable depth buffer does not make every view pay for one.
        let host = RenderHost::new(ctx.gpu.clone(), config, workflow);

        Self {
            host,
            camera,
            lights,
            background_color: wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 },
        }
    }

    /// The wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.host.gpu().device
    }

    /// The wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.host.gpu().queue
    }

    /// The current render target size as (width, height) in pixels.
    pub fn size(&self) -> (u32, u32) {
        self.host.targets().size()
    }

    /// The target texture format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.host.targets().format()
    }

    /// The MSAA sample count (1 = no MSAA).
    pub fn sample_count(&self) -> u32 {
        self.host.targets().sample_count()
    }

    /// Set the color the frame is cleared to before geometry draws.
    pub fn set_background_color(&mut self, color: RgbaColor) {
        self.background_color = rgba_to_wgpu_color(color);
    }

    /// Replace the active rendering workflow.
    ///
    /// The new workflow takes effect immediately on the next frame, with the
    /// frame attachments reallocated to match its passes. The previous workflow
    /// and all its GPU resources are dropped; the material pipelines cached in
    /// [`RenderContext`] are retained across workflow swaps.
    pub fn set_workflow(&mut self, workflow: SceneWorkflow) {
        self.host.set_workflow(workflow);
    }

    /// The active rendering workflow.
    pub fn workflow(&self) -> &SceneWorkflow {
        self.host.workflow()
    }

    /// Edit the active workflow in place: insert a pass next to a known one,
    /// swap one out, drop one, or retune one through
    /// [`Workflow::pass_mut`](crate::render_core::Workflow::pass_mut).
    ///
    /// Frame attachments are reconciled with the edited pass list when the
    /// returned guard drops.
    pub fn workflow_mut(&mut self) -> WorkflowGuard<'_, SceneFrames> {
        self.host.workflow_mut()
    }

    /// A [`PassBuilder`] configured for this renderer's target size, format and
    /// MSAA settings — the way to construct a pass that will go into this
    /// renderer's workflow.
    pub fn pass_builder<'a>(&self, ctx: &'a mut RenderContext) -> PassBuilder<'a> {
        ctx.pass_builder(self.host.targets().size())
    }

    /// Resize the render target to `new_size` (width, height) in pixels.
    ///
    /// Recreates the depth/MSAA attachments and lets the active workflow's
    /// passes recreate their size-dependent resources.
    pub fn resize(&mut self, new_size: (u32, u32)) {
        self.host.resize(new_size);
    }

    /// Render the scene to an RGBA image, at the renderer's current size.
    ///
    /// This is the primary API for headless rendering, and unlike
    /// [`render_scene_to_view`](Self::render_scene_to_view) it is
    /// self-contained: it locks the scene, [`prepare`](SceneResources::prepare)s
    /// `shared`, and submits the GPU work itself — so it counts as that
    /// scene's `prepare` for the frame.
    ///
    /// `lights` are resolved against `scene` and `camera` and composed after
    /// the scene's own lights; pass `&[]` for scene lighting only.
    pub fn render_scene_to_image(
        &mut self,
        ctx: &mut RenderContext,
        shared: &mut SceneResources,
        scene: &mut Scene,
        camera: &PositionedCamera,
        lights: &[PositionedLight],
        highlight: Option<&dyn HighlightQuery>,
    ) -> Result<image::RgbaImage> {
        // Lock scene for duration of rendering
        let mut scene = scene.lock();

        shared.prepare(ctx, &mut scene)?;

        let size = self.host.targets().size();
        self.camera.write(&self.host.gpu().queue, camera);
        let draw_data = DrawData::new(&scene, camera, size, highlight);
        let lights = lights::resolve_lights(lights, &scene, camera);
        self.lights.write(&self.host.gpu().queue, &lights);

        // Build the frame from disjoint field borrows of `self`, `shared`, and
        // `ctx`, then hand it to the host's readback path, which owns the
        // offscreen target and the encoder/submit. IBL resolution is inlined
        // (not a helper) so the borrow is of the `ibl_resources` field alone,
        // leaving `host` borrowable.
        let ibl_bind_group = scene
            .active_environment_map()
            .and_then(|env_id| shared.ibl_resources.get_processed(env_id))
            .map(|processed| &processed.bind_group);
        let mut frame = SceneFrame {
            scene: &scene,
            draw: &draw_data,
            gpu_meshes: &shared.gpu_meshes,
            bindings: SceneBindingRefs {
                camera: &self.camera.bind_group,
                lights: &self.lights.bind_group,
                ibl: ibl_bind_group,
            },
            scene_props: SceneProperties { has_ibl: ibl_bind_group.is_some() },
            projection: camera.projection,
            materials: &shared.materials,
            pipelines: &mut ctx.pipelines,
            background_color: self.background_color,
        };
        let pixels = self.host.render_to_rgba(&mut frame)?;

        image::RgbaImage::from_raw(pixels.width, pixels.height, pixels.data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image from rendered data"))
    }

    /// Render the scene into `view`, recording into `encoder`.
    ///
    /// The encoder is not submitted — the caller is responsible for that.
    /// `shared` must have been [`prepare`](SceneResources::prepare)d for this
    /// frame; the caller holds the scene lock for the duration of the render.
    ///
    /// `lights` are resolved against `scene` and `camera` and composed after
    /// the scene's own lights; pass `&[]` for scene lighting only. A non-empty
    /// `highlight` renders selection outlines and sub-geometry highlights.
    pub fn render_scene_to_view(
        &mut self,
        ctx: &mut RenderContext,
        shared: &mut SceneResources,
        scene: &SceneData,
        camera: &PositionedCamera,
        lights: &[PositionedLight],
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        highlight: Option<&dyn HighlightQuery>,
    ) -> Result<()> {
        let size = self.host.targets().size();

        self.camera.write(&self.host.gpu().queue, camera);

        // Collect, sort, and partition draw batches for this frame
        let draw_data = DrawData::new(scene, camera, size, highlight);
        let lights = lights::resolve_lights(lights, scene, camera);
        self.lights.write(&self.host.gpu().queue, &lights);

        // Build the frame from disjoint field borrows of `self`, `shared`, and
        // `ctx`. Because the frame borrows only the scene subsystems and the
        // pipeline cache (not `host`), `&mut self.host` in `render` coexists
        // with the frame's borrows. IBL resolution is inlined so the borrow is
        // of the `ibl_resources` field alone.
        let ibl_bind_group = scene
            .active_environment_map()
            .and_then(|env_id| shared.ibl_resources.get_processed(env_id))
            .map(|processed| &processed.bind_group);
        let mut frame = SceneFrame {
            scene,
            draw: &draw_data,
            gpu_meshes: &shared.gpu_meshes,
            bindings: SceneBindingRefs {
                camera: &self.camera.bind_group,
                lights: &self.lights.bind_group,
                ibl: ibl_bind_group,
            },
            scene_props: SceneProperties { has_ibl: ibl_bind_group.is_some() },
            projection: camera.projection,
            materials: &shared.materials,
            pipelines: &mut ctx.pipelines,
            background_color: self.background_color,
        };
        self.host.render(encoder, view, &mut frame);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::TextureFormat;

    #[test]
    fn scene_color_format_promotes_to_srgb() {
        assert_eq!(scene_color_format(TextureFormat::Rgba8Unorm), TextureFormat::Rgba8UnormSrgb);
        assert_eq!(scene_color_format(TextureFormat::Bgra8Unorm), TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn scene_color_format_is_idempotent() {
        assert_eq!(
            scene_color_format(TextureFormat::Rgba8UnormSrgb),
            TextureFormat::Rgba8UnormSrgb
        );
    }

    #[test]
    fn scene_color_format_passes_through_formats_without_an_srgb_variant() {
        assert_eq!(scene_color_format(TextureFormat::Rgba16Float), TextureFormat::Rgba16Float);
    }
}
