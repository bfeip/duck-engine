use crate::abi;
use crate::render_core::{FrameTargets, Gpu};

use crate::render_core::GpuTexture;

use super::super::scene_bindings::{CameraBinding, CameraUniform};
use super::super::pass_context::{SceneFrame, SceneRenderPass};
use super::main_pass::{PrimitiveFilter, draw_batches};

/// Draws each sub-view in its own region of the surface, after the main view.
///
/// For every [`SubViewDraw`](super::super::batching) the pass:
/// - writes that sub-view's camera into a dedicated slot buffer,
/// - opens a render pass that *loads* the existing color (so the main view shows
///   through) and *clears* this pass's private depth buffer,
/// - confines drawing to the sub-view's pixel rectangle via `set_viewport` +
///   `set_scissor_rect`,
/// - binds the slot camera at group 0 and draws the subtree's batches.
///
/// Because `LoadOp::Clear` clears the whole depth attachment (WebGPU has no
/// sub-rect clear), the viewport+scissor confine all draws to the sub-rect; the
/// shared depth buffer is intentionally not reused so sub-views depth-test only
/// among their own geometry.
///
/// Limitations (v1): sub-views share the main scene's lights and IBL; the region
/// is not cleared to a background, so gaps between sub-view geometry reveal the
/// main view beneath.
pub(crate) struct SubViewPass {
    depth: GpuTexture,
    camera_layout: wgpu::BindGroupLayout,
    cameras: Vec<CameraBinding>,
}

impl SubViewPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
        camera_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        Self {
            depth: GpuTexture::depth(device, width, height, sample_count, "sub_view_depth_texture"),
            camera_layout: camera_layout.clone(),
            cameras: Vec::new(),
        }
    }

    /// Ensures at least `count` camera slots exist.
    fn ensure_cameras(&mut self, device: &wgpu::Device, count: usize) {
        while self.cameras.len() < count {
            self.cameras.push(CameraBinding::new(device, &self.camera_layout));
        }
    }
}

impl SceneRenderPass for SubViewPass {
    fn is_active(&self, frame: &SceneFrame<'_>) -> bool {
        frame.draw.has_sub_views()
    }

    fn resize(&mut self, gpu: &Gpu, targets: &FrameTargets) {
        let (w, h) = targets.size();
        self.depth = GpuTexture::depth(&gpu.device, w, h, targets.sample_count(), "sub_view_depth_texture");
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        let sub_views = frame.draw.sub_views();
        self.ensure_cameras(&gpu.device, sub_views.len());

        let (surface_w, surface_h) = targets.size();
        for (i, sv) in sub_views.iter().enumerate() {
            // Write this sub-view's camera into its dedicated slot buffer.
            let uniform = CameraUniform::from_positioned_camera(&sv.camera);
            gpu.queue
                .write_buffer(&self.cameras[i].buffer, 0, bytemuck::cast_slice(&[uniform]));

            let (x, y, w, h) = sv.rect.to_pixels(surface_w, surface_h);
            let (color_view, resolve_target) = targets.color_views(view);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sub-View Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
            render_pass.set_scissor_rect(x, y, w, h);

            // Bind this sub-view's camera at group 0; share lights/IBL with the scene.
            render_pass.set_bind_group(abi::GROUP_CAMERA, &self.cameras[i].bind_group, &[]);
            render_pass.set_bind_group(abi::GROUP_LIGHTS, frame.bindings.lights, &[]);
            if let Some(ibl) = frame.bindings.ibl {
                render_pass.set_bind_group(abi::GROUP_IBL, ibl, &[]);
            }

            draw_batches(gpu, &mut render_pass, &sv.batches, frame, true, PrimitiveFilter::All);
        }
    }
}
