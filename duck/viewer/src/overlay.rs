//! Screen-space shapes drawn over views, such as a rubber band.

use std::ops::Range;

use bytemuck::{Pod, Zeroable};

use crate::scene::common::{rgba_to_array, RgbaColor};
use crate::view::PixelRect;

/// A rectangle drawn over a view, in view-local pixels (Y-down).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlayRect {
    /// The top-left corner.
    pub min: (f32, f32),
    /// The bottom-right corner.
    pub max: (f32, f32),
    /// Interior color, linear with straight alpha.
    pub fill_color: RgbaColor,
    /// Border color, linear with straight alpha.
    pub stroke_color: RgbaColor,
    /// Border width in pixels, inside the rectangle.
    pub stroke_width: f32,
}

/// Screen-space shapes drawn over one view, above everything it renders.
/// Operators fill it each frame through
/// [`Operator::overlay`](crate::operator::Operator::overlay).
#[derive(Debug, Clone, Default)]
pub struct ScreenOverlay {
    rects: Vec<OverlayRect>,
}

impl ScreenOverlay {
    /// Adds a rectangle.
    pub fn add_rect(&mut self, rect: OverlayRect) {
        self.rects.push(rect);
    }

    /// The rectangles, in the order they were added.
    pub fn rects(&self) -> &[OverlayRect] {
        &self.rects
    }

    /// Whether there is nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.rects.is_empty()
    }
}

/// One [`OverlayRect`] as `overlay.wgsl` reads it, from an instance vertex
/// buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
struct RectInstance {
    /// The rectangle's corners, moved from view-local to target pixels.
    min: [f32; 2],
    max: [f32; 2],
    /// The rectangle's colors, premultiplied.
    fill_color: [f32; 4],
    stroke_color: [f32; 4],
    stroke_width: f32,
    /// The whole target's width and height in pixels, by which the corners
    /// map to clip space. The same for every instance, but carried in each so
    /// the pipeline needs no bind group.
    target_size: [f32; 2],
}

impl RectInstance {
    fn new(rect: &OverlayRect, view: PixelRect, target_size: (u32, u32)) -> Self {
        let (x, y) = (view.x as f32, view.y as f32);
        Self {
            min: [rect.min.0 + x, rect.min.1 + y],
            max: [rect.max.0 + x, rect.max.1 + y],
            fill_color: rgba_to_array(rect.fill_color.premultiplied()),
            stroke_color: rgba_to_array(rect.stroke_color.premultiplied()),
            stroke_width: rect.stroke_width,
            target_size: [target_size.0 as f32, target_size.1 as f32],
        }
    }
}

/// Draws views' [`ScreenOverlay`]s over the composited target.
pub(crate) struct OverlayPainter {
    pipeline: wgpu::RenderPipeline,
    /// Grown to fit the largest frame so far.
    instances: Option<wgpu::Buffer>,
}

impl OverlayPainter {
    /// A painter for targets of `format`, which must be sRGB: colors are
    /// blended in linear space.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Screen Overlay Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Screen Overlay Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let attributes = wgpu::vertex_attr_array![
            0 => Float32x2, // min
            1 => Float32x2, // max
            2 => Float32x4, // fill_color
            3 => Float32x4, // stroke_color
            4 => Float32,   // stroke_width
            5 => Float32x2, // target_size
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Screen Overlay Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<RectInstance>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &attributes,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Self { pipeline, instances: None }
    }

    /// Draw each view's overlay over `target`, a `target_size` texture view,
    /// clipped to the view's rect. Records nothing when there is nothing to
    /// draw.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        target_size: (u32, u32),
        overlays: &[(PixelRect, ScreenOverlay)],
    ) {
        let mut instances = Vec::new();
        let mut batches: Vec<(PixelRect, Range<u32>)> = Vec::new();
        for (view, overlay) in overlays {
            let start = instances.len() as u32;
            instances.extend(overlay.rects().iter().map(|r| RectInstance::new(r, *view, target_size)));
            batches.push((*view, start..instances.len() as u32));
        }
        if instances.is_empty() {
            return;
        }

        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        let buffer = match &self.instances {
            Some(buffer) if buffer.size() >= bytes.len() as u64 => buffer,
            _ => self.instances.insert(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Screen Overlay Instances"),
                size: (bytes.len() as u64).next_power_of_two(),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
        };
        queue.write_buffer(buffer, 0, bytes);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Screen Overlay Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, buffer.slice(..));
        for (view, range) in batches.into_iter().filter(|(_, range)| !range.is_empty()) {
            pass.set_scissor_rect(view.x, view.y, view.width, view.height);
            pass.draw(0..6, range);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn instances_sit_in_target_pixels_with_premultiplied_colors() {
        let rect = OverlayRect {
            min: (10.0, 20.0),
            max: (30.0, 40.0),
            fill_color: RgbaColor { r: 0.2, g: 0.4, b: 1.0, a: 0.5 },
            stroke_color: RgbaColor::BLUE,
            stroke_width: 1.0,
        };
        let view = PixelRect { x: 100, y: 50, width: 400, height: 300 };
        let instance = RectInstance::new(&rect, view, (800, 600));

        assert_eq!((instance.min, instance.max), ([110.0, 70.0], [130.0, 90.0]));
        assert_eq!(instance.target_size, [800.0, 600.0]);
        assert_eq!(instance.stroke_color, [0.0, 0.0, 1.0, 1.0]);
        let expected = [0.1, 0.2, 0.5, 0.5];
        assert!(instance.fill_color.iter().zip(expected).all(|(a, b)| (a - b).abs() < EPSILON));
    }

    /// The shader through naga's parser and validator, the only device-free
    /// coverage it has.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn shader_is_valid_wgsl() {
        use wgpu::naga;

        let source = include_str!("overlay.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("WGSL parse failed: {}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation failed: {e:?}"));
    }
}
