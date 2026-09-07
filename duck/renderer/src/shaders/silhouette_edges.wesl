// silhouette_edges.wesl
// Screen-space silhouette edge detection using depth buffer discontinuities.
//
// Feature flag "depth_multisampled":
//   true  — depth_texture is texture_depth_multisampled_2d; reads sample 0.
//   false — depth_texture is texture_depth_2d; reads mip level 0.

struct SilhouetteUniform {
    edge_color: vec4<f32>,
    threshold: f32,
    // 1.0 for a perspective depth curve, 0.0 for a linear (orthographic) one.
    perspective_depth: f32,
    // WGSL aligns to 32 bytes; Rust repr(C) side carries _pad: [f32; 2]
};

@if(depth_multisampled)
@group(0) @binding(0)
var depth_texture: texture_depth_multisampled_2d;

@if(!depth_multisampled)
@group(0) @binding(0)
var depth_texture: texture_depth_2d;

@group(0) @binding(1)
var<uniform> silhouette: SilhouetteUniform;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_silhouette(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let c = vec2<i32>(i32(frag_coord.x), i32(frag_coord.y));
    let d = textureLoad(depth_texture, c, 0);

    // Background pixels have no geometry to border against.
    if d >= 1.0 { discard; }

    let dx = abs(textureLoad(depth_texture, c + vec2(1, 0), 0) - textureLoad(depth_texture, c + vec2(-1, 0), 0));
    let dy = abs(textureLoad(depth_texture, c + vec2(0, 1), 0) - textureLoad(depth_texture, c + vec2(0, -1), 0));

    // Under perspective, normalize the gradient by (1 - depth): because
    // 1-d ≈ near/z when far >> near, dividing removes the depth-range
    // dependency, leaving a value proportional to the view-space angular depth
    // change that works for any near/far ratio. Orthographic depth is already
    // linear in view-space z, so its gradient needs no normalization.
    let scale = max(mix(1.0, 1.0 - d, silhouette.perspective_depth), 1e-6);
    if max(dx, dy) / scale < silhouette.threshold { discard; }

    return silhouette.edge_color;
}
