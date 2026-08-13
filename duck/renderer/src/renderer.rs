mod batching;
mod bind_group_layouts;
mod custom_pipeline;
mod material_system;
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
use material_system::MaterialSystem;
use mesh::MeshGpuResources;
use scene_bindings::{CameraBinding, LightsBinding};

/// Shared GPU state for one scene: resource caches generation-synced to a
/// [`SceneData`], plus the bind group layouts and pipeline caches every
/// renderer over that scene draws from.
///
/// One `SceneResources` serves any number of [`Renderer`]s (views) of the same
/// scene; create one per distinct scene. It owns the destructive
/// [`prepare`](Self::prepare), which must run exactly once per scene per frame.
///
/// All renderers sharing a `SceneResources` render at its target format and
/// MSAA sample count — both are baked into the shared pipeline caches.
pub struct SceneResources {
    gpu: Gpu,
    format: wgpu::TextureFormat,
    sample_count: u32,

    layouts: BindGroupLayouts,
    /// Materials: pipelines, shader generator, per-material bind groups, fallbacks.
    materials: MaterialSystem,
    ibl_resources: IblResources,

    // Per-object geometry GPU resources, generation-synced to the scene.
    gpu_meshes: GenCache<MeshId, MeshGpuResources>,
    gpu_textures: GenCache<TextureId, GpuTexture>,
}

impl SceneResources {
    /// Creates the shared GPU state for one scene.
    ///
    /// `format` and `sample_count` fix the target configuration for every
    /// [`Renderer`] created from this value;
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
        let ibl_resources = IblResources::new(&gpu.device, &gpu.queue, &layouts.ibl, has_compute);
        let materials = MaterialSystem::new(&layouts, ShaderGenerator::new(), sample_count, format);

        Self {
            gpu,
            format,
            sample_count,
            layouts,
            materials,
            ibl_resources,
            gpu_meshes: GenCache::new(),
            gpu_textures: GenCache::new(),
        }
    }

    /// The wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.gpu.device
    }

    /// The wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.gpu.queue
    }

    /// The target texture format shared by all renderers over this scene.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// The MSAA sample count shared by all renderers over this scene.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Clear all scene-specific GPU resources.
    ///
    /// Call this when the scene is cleared or replaced to ensure stale GPU
    /// buffers (vertex data, textures, material bind groups) are not reused.
    pub fn clear(&mut self) {
        self.gpu_meshes.clear();
        self.gpu_textures.clear();
        self.materials.clear();
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

/// Per-view render state: the frame targets (depth/MSAA), the active workflow,
/// and the camera and lights uniforms for one view of a scene.
///
/// Scene resource caches live in [`SceneResources`], shared by every renderer
/// over the same scene; the renderer borrows them per call.
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

    /// Create a renderer drawing from `shared` at the given target size.
    ///
    /// Format and sample count come from `shared` — every renderer over a
    /// `SceneResources` renders with the same pipeline configuration.
    pub fn new(shared: &mut SceneResources, width: u32, height: u32) -> Self {
        let config = TargetConfig {
            size: (width, height),
            format: shared.format,
            sample_count: shared.sample_count,
        };

        let camera = CameraBinding::new(&shared.gpu.device, &shared.layouts.camera);
        let lights = LightsBinding::new(&shared.gpu.device, &shared.layouts.light);

        let shaded_workflow = ShadedWorkflow::new(
            &shared.gpu.device,
            config,
            &shared.layouts.camera,
            &shared.layouts.light,
            &shared.layouts.color,
            shared.materials.shader_generator_mut(),
        );

        let host = RenderHost::new(
            shared.gpu.clone(),
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
    /// cached in [`SceneResources`] are retained across workflow swaps.
    pub fn set_workflow(&mut self, workflow: Box<SceneWorkflow>) {
        self.host.set_workflow(workflow);
    }

    /// Create a new [`ShadedWorkflow`] configured for this renderer's target and
    /// MSAA settings. Pass to [`set_workflow`](Self::set_workflow) to activate it.
    pub fn shaded_workflow(&self, shared: &mut SceneResources) -> ShadedWorkflow {
        ShadedWorkflow::new(
            &shared.gpu.device,
            self.host.targets().config(),
            &shared.layouts.camera,
            &shared.layouts.light,
            &shared.layouts.color,
            shared.materials.shader_generator_mut(),
        )
    }

    /// Create a new [`HiddenLineWorkflow`] configured for this renderer's target and
    /// MSAA settings. Pass to [`set_workflow`](Self::set_workflow) to activate it.
    pub fn hidden_line_workflow(
        &self,
        shared: &mut SceneResources,
        config: HiddenLineConfig,
    ) -> HiddenLineWorkflow {
        HiddenLineWorkflow::new(
            &shared.gpu.device,
            self.host.targets().format(),
            self.host.targets().sample_count(),
            &shared.layouts.camera,
            &shared.layouts.light,
            &shared.layouts.color,
            shared.materials.shader_generator_mut(),
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
        shared: &mut SceneResources,
        scene: &mut Scene,
        camera: &PositionedCamera,
        extra_lights: &[ResolvedLight],
        highlight: Option<&dyn HighlightQuery>,
    ) -> Result<image::RgbaImage> {
        // Lock scene for duration of rendering
        let mut scene = scene.lock();

        shared.prepare(&mut scene)?;

        let size = self.host.targets().size();
        self.camera.write(&self.host.gpu().queue, camera);
        let draw_data = DrawData::new(&scene, camera, size, highlight);
        self.lights.write(&self.host.gpu().queue, draw_data.lights(), extra_lights);

        // Build the frame from disjoint field borrows of `self` and `shared`,
        // then hand it to the host's readback path, which owns the offscreen
        // target and the encoder/submit. IBL resolution is inlined (not a
        // helper) so the borrow is of the `ibl_resources` field alone, leaving
        // `materials`/`host` borrowable.
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
            materials: &mut shared.materials,
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

        // Build the frame from disjoint field borrows of `self` and `shared`.
        // Because the frame borrows only the scene subsystems (not `host`),
        // `&mut self.host` in `render` coexists with the frame's
        // `&mut shared.materials` and shared borrows. IBL resolution is inlined
        // so the borrow is of the `ibl_resources` field alone, leaving
        // `materials`/`host` borrowable.
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
            materials: &mut shared.materials,
            background_color: self.background_color,
        };
        self.host.render(encoder, view, &mut frame);

        Ok(())
    }
}
