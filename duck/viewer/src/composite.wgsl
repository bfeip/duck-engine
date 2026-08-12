// View compositor: blits one view's color texture onto the viewer target.
//
// Drawn as a fullscreen triangle with the render pass viewport set to the
// view's rect, so clip space spans exactly that rect and `uv` spans the source
// texture. Straight-alpha blending is configured on the pipeline.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    // Fullscreen triangle: (-1,-1), (3,-1), (-1,3) in NDC.
    let x = f32(i32(index & 1u) * 4 - 1);
    let y = f32(i32(index & 2u) * 2 - 1);
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // NDC y-up to texture y-down.
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@group(0) @binding(0) var view_texture: texture_2d<f32>;
@group(0) @binding(1) var view_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(view_texture, view_sampler, in.uv);
}
