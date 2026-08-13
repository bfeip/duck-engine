use crate::render_core::GpuTexture;

use super::mesh::{instance_buffer_layout, vertex_buffer_layout};

/// Builder for creating a `wgpu::RenderPipeline` that uses the engine's standard
/// vertex and instance buffer layouts.
///
/// Obtain one via [`SceneResources::custom_pipeline_builder`](super::SceneResources::custom_pipeline_builder).
///
/// # Bind group defaults
///
/// By default the builder includes both the camera (group 0) and lights (group 1)
/// bind group layouts in the pipeline layout. If your shader imports
/// `package::camera` or `package::lighting`, keep these included — the group
/// indices in the compiled shader must match the pipeline layout.
///
/// Call [`without_camera`](Self::without_camera) or
/// [`without_lights`](Self::without_lights) only when your shader does **not**
/// import the corresponding engine module and does not bind that group at all.
/// Removing a group shifts all subsequent group indices down by one, which will
/// cause a mismatch if the shader still references the original index.
pub struct CustomPipelineBuilder<'a> {
    device: &'a wgpu::Device,
    surface_format: wgpu::TextureFormat,
    sample_count: u32,
    camera_bgl: Option<&'a wgpu::BindGroupLayout>,
    lights_bgl: Option<&'a wgpu::BindGroupLayout>,
    extra_bgls: Vec<&'a wgpu::BindGroupLayout>,
    shader_module: Option<&'a wgpu::ShaderModule>,
    vs_entry: &'a str,
    fs_entry: &'a str,
    topology: wgpu::PrimitiveTopology,
    blend: wgpu::BlendState,
    cull_mode: Option<wgpu::Face>,
    depth_write: bool,
    label: Option<&'a str>,
}

impl<'a> CustomPipelineBuilder<'a> {
    pub(super) fn new(
        device: &'a wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        camera_bgl: &'a wgpu::BindGroupLayout,
        lights_bgl: &'a wgpu::BindGroupLayout,
    ) -> Self {
        Self {
            device,
            surface_format,
            sample_count,
            camera_bgl: Some(camera_bgl),
            lights_bgl: Some(lights_bgl),
            extra_bgls: Vec::new(),
            shader_module: None,
            vs_entry: "vs_main",
            fs_entry: "fs_main",
            topology: wgpu::PrimitiveTopology::TriangleList,
            blend: wgpu::BlendState::REPLACE,
            cull_mode: Some(wgpu::Face::Back),
            depth_write: true,
            label: None,
        }
    }

    /// Set the shader module and entry point names.
    ///
    /// This is required — [`build`](Self::build) will panic if not called.
    pub fn shader(mut self, module: &'a wgpu::ShaderModule, vs: &'a str, fs: &'a str) -> Self {
        self.shader_module = Some(module);
        self.vs_entry = vs;
        self.fs_entry = fs;
        self
    }

    /// Exclude the camera bind group (group 0) from the pipeline layout.
    ///
    /// Only use this when your shader does **not** import `package::camera`.
    /// Excluding camera shifts the lights group from index 1 to index 0, which
    /// will break any shader that imports `package::lighting`.
    pub fn without_camera(mut self) -> Self {
        self.camera_bgl = None;
        self
    }

    /// Exclude the lights bind group from the pipeline layout.
    ///
    /// Only use this when your shader does **not** import `package::lighting`.
    pub fn without_lights(mut self) -> Self {
        self.lights_bgl = None;
        self
    }

    /// Add an extra bind group layout appended after camera and lights.
    ///
    /// The first call adds at group 2 (or 0/1 if camera/lights were excluded).
    /// Subsequent calls increment the group index by one each time.
    pub fn extra_bind_group(mut self, bgl: &'a wgpu::BindGroupLayout) -> Self {
        self.extra_bgls.push(bgl);
        self
    }

    /// Set the primitive topology. Defaults to `TriangleList`.
    pub fn topology(mut self, topology: wgpu::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Set the fragment blend state. Defaults to `BlendState::REPLACE` (opaque).
    pub fn blend(mut self, blend: wgpu::BlendState) -> Self {
        self.blend = blend;
        self
    }

    /// Set the back-face cull mode. Defaults to `Some(Face::Back)`.
    pub fn cull_mode(mut self, cull_mode: Option<wgpu::Face>) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    /// Enable or disable depth writes. Defaults to `true`.
    pub fn depth_write(mut self, enabled: bool) -> Self {
        self.depth_write = enabled;
        self
    }

    /// Set a debug label for the pipeline.
    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// Build the `wgpu::RenderPipeline`.
    ///
    /// # Panics
    ///
    /// Panics if [`shader`](Self::shader) was not called.
    pub fn build(self) -> wgpu::RenderPipeline {
        let module = self.shader_module
            .expect("CustomPipelineBuilder::shader() must be called before build()");

        let mut bgls: Vec<&wgpu::BindGroupLayout> = Vec::new();
        if let Some(cam) = self.camera_bgl { bgls.push(cam); }
        if let Some(lit) = self.lights_bgl { bgls.push(lit); }
        bgls.extend(self.extra_bgls.iter());

        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: self.label,
            bind_group_layouts: &bgls,
            push_constant_ranges: &[],
        });

        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: self.label,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some(self.vs_entry),
                buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(self.fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_format,
                    blend: Some(self.blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: self.topology,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: self.cull_mode,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: GpuTexture::DEPTH_FORMAT,
                depth_write_enabled: self.depth_write,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: self.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    }
}
