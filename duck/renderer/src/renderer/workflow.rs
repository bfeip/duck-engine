use crate::render_core::{FrameTargets, Gpu, RenderWorkflow, TargetConfig};
use crate::scene::resource::PrimitiveType;
use crate::scene::common::RgbaColor;

use super::pass_context::{SceneFrame, SceneFrames, SceneRenderPass};
use super::scene_pass::{
    FlatColorPass, FlatColorPassDesc,
    MainPass, OverlayPass, SilhouetteEdgesPass, SubGeomHighlightPass, outline_passes,
};
use crate::shaders::ShaderGenerator;

/// The default shaded rendering workflow.
///
/// Runs the standard pass sequence: scene faces, the highlight outline mask,
/// scene lines and points, the outline composite, overlay (always-on-top)
/// geometry, and sub-geometry highlights. Custom passes can be injected via
/// [`ShadedWorkflow::set_passes`].
pub struct ShadedWorkflow {
    passes: Vec<Box<dyn SceneRenderPass>>,
}

impl ShadedWorkflow {
    pub(super) fn new(
        device: &wgpu::Device,
        config: TargetConfig,
        camera_bgl: &wgpu::BindGroupLayout,
        lights_bgl: &wgpu::BindGroupLayout,
        material_color_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut ShaderGenerator,
    ) -> Self {
        let (width, height) = config.size;
        let sample_count = config.sample_count;
        let (outline_mask, outline_composite) =
            outline_passes(device, config, camera_bgl, shader_generator);
        Self {
            passes: vec![
                Box::new(MainPass::faces()),
                // Between the two geometry passes: the mask depth-tests against
                // faces only.
                Box::new(outline_mask),
                Box::new(MainPass::lines_and_points()),
                // After lines and points, so nothing cuts the outline band.
                Box::new(outline_composite),
                Box::new(OverlayPass::new(device, width, height, sample_count)),
                // Sub-geometry highlights draw on top of the node outlines.
                Box::new(SubGeomHighlightPass::new(
                    device, config.format, sample_count,
                    camera_bgl, lights_bgl, material_color_bgl, shader_generator,
                )),
            ],
        }
    }

    /// Replace the pass list. Passes execute in order; each receives the same
    /// [`SceneFrame`] and can skip itself by returning `false` from
    /// [`SceneRenderPass::is_active`].
    pub fn set_passes(&mut self, passes: Vec<Box<dyn SceneRenderPass>>) {
        self.passes = passes;
    }
}

impl RenderWorkflow<SceneFrames> for ShadedWorkflow {
    fn name(&self) -> &'static str { "Shaded" }

    fn resize(&mut self, gpu: &Gpu, targets: &FrameTargets) {
        for pass in &mut self.passes {
            pass.resize(gpu, targets);
        }
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        for pass in &mut self.passes {
            if pass.is_active(frame) {
                pass.execute(gpu, targets, encoder, view, frame);
            }
        }
    }
}

/// Color configuration for [`HiddenLineWorkflow`].
#[derive(Clone, Debug)]
pub struct HiddenLineConfig {
    /// Background / face color (also used as the clear color).
    pub face_color: RgbaColor,
    /// Color of lines that are visible (in front of solid geometry).
    pub visible_line_color: RgbaColor,
    /// Color of lines that are occluded (behind solid geometry).
    pub hidden_line_color: RgbaColor,
}

impl Default for HiddenLineConfig {
    fn default() -> Self {
        Self {
            face_color: RgbaColor::WHITE,
            visible_line_color: RgbaColor::BLACK,
            hidden_line_color: RgbaColor { r: 0.6, g: 0.6, b: 0.6, a: 1.0 },
        }
    }
}

/// Hidden-line rendering workflow.
///
/// Renders scene geometry as solid faces with silhouette edges detected from
/// the depth buffer, plus explicit `LineList` primitives in two flat colors:
/// one for occluded lines and one for visible lines. All colors are configured
/// via [`HiddenLineConfig`].
///
/// Pass sequence:
/// 1. Solid — clear to face color, render all triangles, write depth.
/// 2. Silhouette — fullscreen depth-discontinuity edge detection.
/// 3. Occluded lines — hidden line color where depth compare is `Greater`.
/// 4. Visible lines — visible line color where depth compare is `LessEqual`.
pub struct HiddenLineWorkflow {
    solid_pass: FlatColorPass,
    silhouette_pass: SilhouetteEdgesPass,
    occluded_pass: FlatColorPass,
    visible_pass: FlatColorPass,
}

impl HiddenLineWorkflow {
    pub(super) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        camera_bgl: &wgpu::BindGroupLayout,
        lights_bgl: &wgpu::BindGroupLayout,
        material_color_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut ShaderGenerator,
        config: HiddenLineConfig,
    ) -> Self {
        let solid_pass = FlatColorPass::new(
            device, surface_format, sample_count,
            camera_bgl, lights_bgl, material_color_bgl, shader_generator,
            FlatColorPassDesc {
                label: "Hidden Line Solid",
                cull_mode: Some(wgpu::Face::Back),
                depth_compare: wgpu::CompareFunction::Less,
                depth_write: true,
                // Push faces slightly away so coplanar edges pass depth test.
                depth_bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
                clear_color: Some(crate::rgba_to_wgpu_color(config.face_color)),
                primitive_filter: PrimitiveType::TriangleList,
                color: config.face_color,
            },
        );
        let silhouette_pass = SilhouetteEdgesPass::new(device, surface_format, sample_count, shader_generator);

        let mut make_line_pass = |label, depth_compare, color| FlatColorPass::new(
            device, surface_format, sample_count,
            camera_bgl, lights_bgl, material_color_bgl, shader_generator,
            FlatColorPassDesc {
                label,
                cull_mode: None,
                depth_compare,
                depth_write: false,
                depth_bias: Default::default(),
                clear_color: None,
                primitive_filter: PrimitiveType::LineList,
                color,
            },
        );
        let occluded_pass = make_line_pass(
            "Hidden Line Occluded",
            wgpu::CompareFunction::Greater,
            config.hidden_line_color,
        );
        let visible_pass = make_line_pass(
            "Hidden Line Visible",
            wgpu::CompareFunction::LessEqual,
            config.visible_line_color,
        );

        Self { solid_pass, silhouette_pass, occluded_pass, visible_pass }
    }
}

impl RenderWorkflow<SceneFrames> for HiddenLineWorkflow {
    fn name(&self) -> &'static str { "Hidden Line" }

    fn resize(&mut self, gpu: &Gpu, targets: &FrameTargets) {
        self.solid_pass.resize(gpu, targets);
        self.silhouette_pass.resize(gpu, targets);
        self.occluded_pass.resize(gpu, targets);
        self.visible_pass.resize(gpu, targets);
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        self.solid_pass.execute(gpu, targets, encoder, view, frame);
        self.silhouette_pass.execute(gpu, targets, encoder, view, frame);
        let has_lines = frame.draw.all_batches().iter().any(|b| b.primitive_type == PrimitiveType::LineList);
        if has_lines {
            self.occluded_pass.execute(gpu, targets, encoder, view, frame);
            self.visible_pass.execute(gpu, targets, encoder, view, frame);
        }
    }
}
