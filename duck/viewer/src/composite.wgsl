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

// View textures are sRGB, so sampling decodes to linear.

// Used when the final target is an sRGB format and encodes on write.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(view_texture, view_sampler, in.uv);
}

// sRGB transfer encode (piecewise, not pow(1/2.2)).
fn gamma_from_linear_rgb(rgb: vec3<f32>) -> vec3<f32> {
    let lower = rgb * 12.92;
    let higher = 1.055 * pow(rgb, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(higher, lower, rgb < vec3<f32>(0.0031308));
}

// Used when the final target is not an sRGB format, so the encode must be
// applied here. Alpha is straight and stays linear.
@fragment
fn fs_encode(in: VertexOutput) -> @location(0) vec4<f32> {
    let linear = textureSample(view_texture, view_sampler, in.uv);
    return vec4<f32>(gamma_from_linear_rgb(linear.rgb), linear.a);
}
