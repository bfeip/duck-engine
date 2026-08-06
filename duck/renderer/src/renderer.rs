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

pub use batching::{DrawBatch, DrawData};
pub use custom_pipeline::CustomPipelineBuilder;
pub use mesh::{instance_buffer_layout, vertex_buffer_layout};
pub use pass_context::{SceneFrame, SceneFrames, SceneRenderPass, SceneWorkflow};
pub use pipeline::MaterialPipelineCache;
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
        SceneProperties,
        common::RgbaColor,
        resource::{MeshId, TextureId}
    },
    shaders::ShaderGenerator
};

use bind_group_layouts::BindGroupLayouts;
use material_system::MaterialSystem;
use mesh::MeshGpuResources;
use scene_bindings::SceneBindings;

pub struct Renderer {
    /// Core dispatch: owns the GPU handles, shared frame targets, the active
    /// workflow, and headless readback. The scene subsystems below are reached
    /// alongside `&mut host` via disjoint field borrows when building a frame.
    host: RenderHost<SceneFrames>,
    background_color: wgpu::Color,

    // Grouped resources
    layouts: BindGroupLayouts,
    /// Scene-level camera + lights uniforms and bind groups.
    bindings: SceneBindings,

    // Subsystems
    /// Materials: pipelines, shader generator, per-material bind groups, fallbacks.
    materials: MaterialSystem,
    ibl_resources: IblResources,

    // Per-object geometry GPU resources, generation-synced to the scene.
    gpu_meshes: GenCache<MeshId, MeshGpuResources>,
    gpu_textures: GenCache<TextureId, GpuTexture>,
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

    /// Create a new renderer from pre-created device and queue.
    ///
    /// The caller (typically Viewer) is responsible for creating the wgpu instance,
    /// adapter, device, queue, and surface. The renderer only needs the device, queue,
    /// surface format, dimensions, and MSAA sample count.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
        has_compute: bool,
    ) -> Self {
        let gpu = Gpu::new(device, queue);
        let config = TargetConfig {
            size: (width, height),
            format: surface_format,
            sample_count,
        };

        // Create grouped resources
        let layouts = BindGroupLayouts::new(&gpu.device);
        let bindings = SceneBindings::new(&gpu.device, &layouts);
        let ibl_resources = IblResources::new(&gpu.device, &gpu.queue, &layouts.ibl, has_compute);

        // The shader generator is shared between the workflow (for pass-specific
        // shaders) and the material system (for material shaders). Build the
        // workflow first, then hand the generator to the material system.
        let mut shader_generator = ShaderGenerator::new();
        let shaded_workflow = ShadedWorkflow::new(
            &gpu.device,
            config,
            &layouts.camera,
            &layouts.light,
            &layouts.color,
            &mut shader_generator,
        );

        let materials = MaterialSystem::new(
            &layouts,
            shader_generator,
            sample_count,
            surface_format,
        );

        let host = RenderHost::new(
            gpu,
            config,
            TargetFeatures { depth: true },
            Box::new(shaded_workflow),
        );

        Self {
            host,
            layouts,
            bindings,
            materials,
            ibl_resources,
            gpu_meshes: GenCache::new(),
            gpu_textures: GenCache::new(),
            background_color: wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 },
        }
    }

    /// Create a new renderer for headless/offscreen rendering.
    ///
    /// This creates its own wgpu instance, adapter, device, and queue without
    /// requiring a surface. Useful for generating still images, thumbnails,
    /// or server-side rendering.
    ///
    /// Uses `Rgba8UnormSrgb` format and disables MSAA for simplicity.
    pub async fn new_headless(width: u32, height: u32) -> Self {
        let (gpu, caps) = Gpu::headless()
            .await
            .expect("Failed to create GPU for headless rendering");

        let surface_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let sample_count = 1; // No MSAA for headless

        Self::new(gpu.device, gpu.queue, surface_format, width, height, sample_count, caps.has_compute)
    }

    /// Get a reference to the wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.host.gpu().device
    }

    /// Get a reference to the wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.host.gpu().queue
    }

    /// Get the current viewport size as (width, height).
    pub fn size(&self) -> (u32, u32) {
        self.host.targets().size()
    }

    /// Get the surface texture format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.host.targets().format()
    }

    /// Get the MSAA sample count (1 = no MSAA, 4 = 4× MSAA).
    pub fn sample_count(&self) -> u32 {
        self.host.targets().sample_count()
    }

    /// Compile a user-supplied WESL shader with access to all engine shader modules.
    ///
    /// Engine modules available for import: `package::common`, `package::camera`,
    /// `package::lighting`, `package::constants`, `package::vertex`, `package::pbr`.
    /// See `docs/custom-passes.md` for the full module reference.
    pub fn compile_user_wesl(&self, source: &str) -> anyhow::Result<wgpu::ShaderModule> {
        crate::shaders::compile_user_wesl(&self.host.gpu().device, source)
    }

    /// Create a pipeline builder pre-configured with the engine's standard vertex
    /// and instance buffer layouts, surface format, and MSAA sample count.
    ///
    /// Camera (group 0) and lights (group 1) bind group layouts are included by
    /// default. See [`CustomPipelineBuilder`] for the full configuration API.
    pub fn custom_pipeline_builder(&self) -> CustomPipelineBuilder<'_> {
        CustomPipelineBuilder::new(
            &self.host.gpu().device,
            self.host.targets().format(),
            self.host.targets().sample_count(),
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

    pub fn set_background_color(&mut self, color: RgbaColor) {
        self.background_color = rgba_to_wgpu_color(color);
    }

    /// Replace the active rendering workflow.
    ///
    /// The new workflow takes effect immediately on the next frame. The previous
    /// workflow and all its GPU resources are dropped. The [`MaterialPipelineCache`] is
    /// retained across workflow swaps.
    pub fn set_workflow(&mut self, workflow: Box<SceneWorkflow>) {
        self.host.set_workflow(workflow);
    }

    /// Create a new [`ShadedWorkflow`] configured for this renderer's device, format,
    /// and MSAA settings. Pass to [`set_workflow`](Self::set_workflow) to activate it.
    pub fn shaded_workflow(&mut self) -> ShadedWorkflow {
        ShadedWorkflow::new(
            &self.host.gpu().device,
            self.host.targets().config(),
            &self.layouts.camera,
            &self.layouts.light,
            &self.layouts.color,
            self.materials.shader_generator_mut(),
        )
    }

    /// Create a new [`HiddenLineWorkflow`] configured for this renderer's device, format,
    /// and MSAA settings. Pass to [`set_workflow`](Self::set_workflow) to activate it.
    pub fn hidden_line_workflow(&mut self, config: HiddenLineConfig) -> HiddenLineWorkflow {
        HiddenLineWorkflow::new(
            &self.host.gpu().device,
            self.host.targets().format(),
            self.host.targets().sample_count(),
            &self.layouts.camera,
            &self.layouts.light,
            &self.layouts.color,
            self.materials.shader_generator_mut(),
            config,
        )
    }

    /// Clear all scene-specific GPU resources.
    ///
    /// Call this when the scene is cleared or replaced to ensure stale GPU
    /// buffers (vertex data, textures, material bind groups) are not reused.
    pub fn clear_gpu_resources(&mut self) {
        self.gpu_meshes.clear();
        self.gpu_textures.clear();
        self.materials.clear();
        self.bindings.invalidate_lights();
    }

    pub fn resize(&mut self, new_size: (u32, u32)) {
        self.host.resize(new_size);
    }

    /// Render the scene to an image buffer.
    ///
    /// This is the primary API for headless rendering. It renders the scene
    /// from the given camera viewpoint and returns the result as an RGBA image.
    ///
    /// The renderer must have been created with dimensions matching the desired
    /// output size, or resized beforehand.
    pub fn render_scene_to_image(
        &mut self,
        camera: &PositionedCamera,
        scene: &mut Scene,
        highlight: Option<&dyn HighlightQuery>,
    ) -> Result<image::RgbaImage> {
        // Lock scene for duration of rendering
        let mut scene = scene.lock();

        self.prepare_scene(&mut scene)?;

        let size = self.host.targets().size();
        self.bindings.write_camera(&self.host.gpu().queue, camera);
        let draw_data = DrawData::new(&scene, camera, size, highlight);

        // Build the frame from disjoint field borrows, then hand it to the host's
        // readback path, which owns the offscreen target and the encoder/submit.
        // IBL resolution is inlined (not a `&self` helper) so the borrow is of the
        // `ibl_resources` field alone, leaving `materials`/`host` borrowable.
        let ibl_bind_group = scene
            .active_environment_map()
            .and_then(|env_id| self.ibl_resources.get_processed(env_id))
            .map(|processed| &processed.bind_group);
        let mut frame = SceneFrame {
            scene: &scene,
            draw: &draw_data,
            gpu_meshes: &self.gpu_meshes,
            bindings: self.bindings.refs(ibl_bind_group),
            scene_props: SceneProperties { has_ibl: ibl_bind_group.is_some() },
            materials: &mut self.materials,
            background_color: self.background_color,
        };
        let pixels = self.host.render_to_rgba(&mut frame)?;

        image::RgbaImage::from_raw(pixels.width, pixels.height, pixels.data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image from rendered data"))
    }

    /// Render the scene to a specific view, updating uniforms and drawing all batches.
    /// If a highlight is provided and not empty, highlight outlines will be rendered.
    /// The encoder is not submitted - the caller is responsible for that.
    pub fn render_scene_to_view(
        &mut self,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        camera_override: Option<&PositionedCamera>,
        scene: &Scene,
        highlight: Option<&dyn HighlightQuery>,
    ) -> Result<()> {
        // Lock scene for the duration of the render
        let scene = scene.lock();

        let size = self.host.targets().size();
        let owned: PositionedCamera;
        let camera: &PositionedCamera = match camera_override {
            Some(c) => c,
            None => {
                let aspect = size.0 as f32 / size.1 as f32;
                owned = scene
                    .active_camera_positioned(aspect)
                    .ok_or_else(|| anyhow::anyhow!("No camera_override provided and scene has no active camera"))?;
                &owned
            }
        };

        self.bindings.write_camera(&self.host.gpu().queue, camera);

        // Collect, sort, and partition draw batches for this frame
        let draw_data = DrawData::new(&scene, camera, size, highlight);

        // Build the frame from disjoint field borrows. Because the frame borrows
        // only the scene subsystems (not `host`), `&mut self.host` in `render`
        // coexists with the frame's `&mut self.materials` and shared borrows.
        // IBL resolution is inlined so the borrow is of the `ibl_resources` field
        // alone, leaving `materials`/`host` borrowable.
        let ibl_bind_group = scene
            .active_environment_map()
            .and_then(|env_id| self.ibl_resources.get_processed(env_id))
            .map(|processed| &processed.bind_group);
        let mut frame = SceneFrame {
            scene: &scene,
            draw: &draw_data,
            gpu_meshes: &self.gpu_meshes,
            bindings: self.bindings.refs(ibl_bind_group),
            scene_props: SceneProperties { has_ibl: ibl_bind_group.is_some() },
            materials: &mut self.materials,
            background_color: self.background_color,
        };
        self.host.render(encoder, view, &mut frame);

        Ok(())
    }

}
