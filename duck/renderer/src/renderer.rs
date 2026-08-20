mod batching;
mod bind_group_layouts;
mod custom_pipeline;
mod material_cache;
mod mesh;
mod pass_context;
mod pipeline;
mod prepare;
mod scene_bindings;
mod scene_pass;
mod surface_config;
mod texture;
mod workflow;

pub use batching::{
    BatchKey, BatchMaterial, DrawBatch, DrawData, InstanceTransform, ResolvedLight, SubGeomBatch,
};
pub use custom_pipeline::CustomPipelineBuilder;
pub use mesh::{instance_buffer_layout, vertex_buffer_layout};
pub use pass_context::{SceneFrame, SceneFrames, SceneRenderPass, SceneWorkflow};
pub use scene_bindings::SceneBindingRefs;
pub use workflow::{HiddenLineConfig, HiddenLineWorkflow, ShadedWorkflow};

use anyhow::Result;

use crate::{
    highlight_query::HighlightQuery,
    ibl::IblResources,
    render_core::{
        GenCache, Gpu, GpuTexture, MaskChannels, RenderHost, TargetConfig, TargetFeatures,
        highest_supported_sample_count,
    },
    rgba_to_wgpu_color,
    scene::{
        PositionedCamera,
        Scene,
        SceneData,
        SceneProperties,
        common::RgbaColor,
        resource::{MeshId, TextureId}
    },
    shaders::ShaderGenerator
};

use bind_group_layouts::BindGroupLayouts;
use material_cache::MaterialCache;
use mesh::MeshGpuResources;
use pipeline::MaterialPipelineCache;
use scene_bindings::{CameraBinding, LightsBinding};

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
    has_compute: bool,

    layouts: BindGroupLayouts,
    /// Material pipelines, their layouts, and the shader generator.
    pipelines: MaterialPipelineCache,
}

impl RenderContext {
    /// Creates the render context for one device and target configuration.
    ///
    /// `format` and `sample_count` are baked into every pipeline built here, so
    /// all renderers over this context render at that configuration;
    /// [`Renderer::preferred_sample_count`] probes a suitable count.
    /// `has_compute` reports compute shader availability, as returned by
    /// [`Gpu`](crate::render_core::Gpu) acquisition; without it, environment
    /// map processing is skipped.
    pub fn new(
        gpu: Gpu,
        format: wgpu::TextureFormat,
        sample_count: u32,
        has_compute: bool,
    ) -> Self {
        let layouts = BindGroupLayouts::new(&gpu.device);
        let pipelines =
            MaterialPipelineCache::new(&layouts, ShaderGenerator::new(), sample_count, format);

        Self { gpu, format, sample_count, has_compute, layouts, pipelines }
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

    /// Whether the device supports compute shaders; without it environment maps
    /// are not processed.
    pub fn has_compute(&self) -> bool {
        self.has_compute
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

    /// Get the bind group layout for the camera uniform (group 0).
    ///
    /// Prefer [`custom_pipeline_builder`](Self::custom_pipeline_builder) for
    /// building custom pipelines — this method is a lower-level escape hatch.
    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layouts.camera
    }

    /// Get the bind group layout for the lights uniform (group 1).
    ///
    /// Prefer [`custom_pipeline_builder`](Self::custom_pipeline_builder) for
    /// building custom pipelines — this method is a lower-level escape hatch.
    pub fn lights_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layouts.light
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
            ctx.has_compute,
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
        let downlevel = adapter.get_downlevel_capabilities().flags;
        if !downlevel.contains(wgpu::DownlevelFlags::MULTISAMPLED_SHADING) {
            return 1;
        }
        let formats = [surface_format, GpuTexture::DEPTH_FORMAT]
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
        let config = TargetConfig {
            size: (width, height),
            format: ctx.format,
            sample_count: ctx.sample_count,
        };

        let camera = CameraBinding::new(&ctx.gpu.device, &ctx.layouts.camera);
        let lights = LightsBinding::new(&ctx.gpu.device, &ctx.layouts.light);

        let shaded_workflow = ShadedWorkflow::new(
            &ctx.gpu.device,
            config,
            &ctx.layouts.camera,
            &ctx.layouts.light,
            &ctx.layouts.color,
            ctx.pipelines.shader_generator_mut(),
        );

        let host = RenderHost::new(
            ctx.gpu.clone(),
            config,
            TargetFeatures { depth: true },
            Box::new(shaded_workflow),
        );

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
    /// The new workflow takes effect immediately on the next frame. The previous
    /// workflow and all its GPU resources are dropped; the material pipelines
    /// cached in [`RenderContext`] are retained across workflow swaps.
    pub fn set_workflow(&mut self, workflow: Box<SceneWorkflow>) {
        self.host.set_workflow(workflow);
    }

    /// Create a new [`ShadedWorkflow`] configured for this renderer's target and
    /// MSAA settings. Pass to [`set_workflow`](Self::set_workflow) to activate it.
    pub fn shaded_workflow(&self, ctx: &mut RenderContext) -> ShadedWorkflow {
        ShadedWorkflow::new(
            &ctx.gpu.device,
            self.host.targets().config(),
            &ctx.layouts.camera,
            &ctx.layouts.light,
            &ctx.layouts.color,
            ctx.pipelines.shader_generator_mut(),
        )
    }

    /// Create a new [`HiddenLineWorkflow`] configured for this renderer's target and
    /// MSAA settings. Pass to [`set_workflow`](Self::set_workflow) to activate it.
    pub fn hidden_line_workflow(
        &self,
        ctx: &mut RenderContext,
        config: HiddenLineConfig,
    ) -> HiddenLineWorkflow {
        HiddenLineWorkflow::new(
            &ctx.gpu.device,
            self.host.targets().format(),
            self.host.targets().sample_count(),
            &ctx.layouts.camera,
            &ctx.layouts.light,
            &ctx.layouts.color,
            ctx.pipelines.shader_generator_mut(),
            config,
        )
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
    /// `extra_lights` are composed after the scene's lights (e.g. camera-space
    /// headlights); pass `&[]` for scene lighting only.
    pub fn render_scene_to_image(
        &mut self,
        ctx: &mut RenderContext,
        shared: &mut SceneResources,
        scene: &mut Scene,
        camera: &PositionedCamera,
        extra_lights: &[ResolvedLight],
        highlight: Option<&dyn HighlightQuery>,
    ) -> Result<image::RgbaImage> {
        // Lock scene for duration of rendering
        let mut scene = scene.lock();

        shared.prepare(ctx, &mut scene)?;

        let size = self.host.targets().size();
        self.camera.write(&self.host.gpu().queue, camera);
        let draw_data = DrawData::new(&scene, camera, size, highlight);
        self.lights.write(&self.host.gpu().queue, draw_data.lights(), extra_lights);

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
    /// `extra_lights` are composed after the scene's lights (e.g. camera-space
    /// headlights); pass `&[]` for scene lighting only. A non-empty `highlight`
    /// renders selection outlines and sub-geometry highlights.
    pub fn render_scene_to_view(
        &mut self,
        ctx: &mut RenderContext,
        shared: &mut SceneResources,
        scene: &SceneData,
        camera: &PositionedCamera,
        extra_lights: &[ResolvedLight],
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        highlight: Option<&dyn HighlightQuery>,
    ) -> Result<()> {
        let size = self.host.targets().size();

        self.camera.write(&self.host.gpu().queue, camera);

        // Collect, sort, and partition draw batches for this frame
        let draw_data = DrawData::new(scene, camera, size, highlight);
        self.lights.write(&self.host.gpu().queue, draw_data.lights(), extra_lights);

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
            materials: &shared.materials,
            pipelines: &mut ctx.pipelines,
            background_color: self.background_color,
        };
        self.host.render(encoder, view, &mut frame);

        Ok(())
    }
}
