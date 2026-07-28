// outline_screenspace.wesl
// Turns the coverage mask — red for the primary selection, green for the
// secondary — into a signed distance to each silhouette, then shades a band of
// uniform width whose alpha falls off over the last texel. Fractional coverage,
// from the MSAA resolve, is what locates an edge to subpixel precision.

struct OutlineUniform {
    primary_color: vec4<f32>,
    secondary_color: vec4<f32>,
    // (primary_width, secondary_width, screen_width, screen_height) in pixels.
    // A width of zero disables that outline.
    params: vec4<f32>,
};

@group(0) @binding(0)
var mask_texture: texture_2d<f32>;

// Binding 1 intentionally absent — no sampler needed; all reads use textureLoad.

@group(0) @binding(2)
var<uniform> outline: OutlineUniform;

// Hard ceiling on the search radius, in texels.
override max_search_radius: i32;

// Stands in for "no edge found yet", so min() always prefers a real distance.
const UNRESOLVED: f32 = 1.0e9;
// Unorm quantization can land a hair under 1.0.
const FULLY_COVERED: f32 = 0.999;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Coverage of both channels at a texel, clamping to the edge so a selection
// running off screen does not read the out-of-bounds zeros as a silhouette.
fn coverage(coord: vec2<i32>) -> vec2<f32> {
    let dims = vec2<i32>(textureDimensions(mask_texture));
    return textureLoad(mask_texture, clamp(coord, vec2<i32>(0, 0), dims - vec2<i32>(1, 1)), 0).rg;
}

// Distance from a texel's center to the edge that left it with coverage `c`,
// positive when the center is outside: a straight edge at distance t covers
// 0.5 - t of the texel.
fn edge_offset(c: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.5) - c;
}

// Distance from `center` to the silhouette as witnessed by a single tap, or
// UNRESOLVED for a channel the tap holds no coverage of.
fn tap_distance(center: vec2<i32>, offset: vec2<i32>) -> vec2<f32> {
    let c = coverage(center + offset);
    let along_offset = vec2<f32>(length(vec2<f32>(offset))) + edge_offset(c);
    return select(vec2<f32>(UNRESOLVED), along_offset, c > vec2<f32>(0.0));
}

// Whether each channel covers this texel and all four of its neighbors.
fn is_deep_interior(center: vec2<i32>, c: vec2<f32>) -> vec2<bool> {
    let neighbors = min(
        min(coverage(center + vec2<i32>(1, 0)), coverage(center + vec2<i32>(-1, 0))),
        min(coverage(center + vec2<i32>(0, 1)), coverage(center + vec2<i32>(0, -1))),
    );
    let full = vec2<f32>(FULLY_COVERED);
    return (c >= full) & (neighbors >= full);
}

@fragment
fn fs_outline(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let center = vec2<i32>(i32(frag_coord.x), i32(frag_coord.y));
    let radius = vec2<f32>(outline.params.x, outline.params.y);

    let c = coverage(center);
    let deep_interior = is_deep_interior(center, c);

    // A texel with coverage of its own already knows its distance; the rest have
    // to go looking.
    var signed_distance = select(vec2<f32>(UNRESOLVED), edge_offset(c), c > vec2<f32>(0.0));

    // Walk the perimeter of expanding squares. Nothing at radius r or beyond can
    // be nearer than r - 0.5, so once both channels are within that the rest of
    // the search cannot improve on what we have.
    let max_radius = min(i32(ceil(max(radius.x, radius.y))) + 1, max_search_radius);
    for (var r = 1i; r <= max_radius; r++) {
        if all(signed_distance <= vec2<f32>(f32(r) - 0.5)) {
            break;
        }
        for (var i = -r; i <= r; i++) {
            signed_distance = min(signed_distance, tap_distance(center, vec2<i32>(i, -r)));
            signed_distance = min(signed_distance, tap_distance(center, vec2<i32>(i, r)));
        }
        for (var i = 1 - r; i <= r - 1; i++) {
            signed_distance = min(signed_distance, tap_distance(center, vec2<i32>(-r, i)));
            signed_distance = min(signed_distance, tap_distance(center, vec2<i32>(r, i)));
        }
    }

    // Full alpha out to radius, then one texel of falloff. Boundary texels carry a
    // negative distance, so the band also covers the object's own antialiased edge
    // and leaves no seam.
    var alpha = clamp(radius + vec2<f32>(0.5) - signed_distance, vec2<f32>(0.0), vec2<f32>(1.0));
    alpha *= vec2<f32>(outline.primary_color.a, outline.secondary_color.a);
    alpha = select(alpha, vec2<f32>(0.0), deep_interior | (radius <= vec2<f32>(0.0)));

    // Primary over secondary, then the result over the scene.
    let out_alpha = alpha.x + alpha.y * (1.0 - alpha.x);
    if out_alpha <= 0.0 {
        discard;
    }
    let rgb = (outline.primary_color.rgb * alpha.x
        + outline.secondary_color.rgb * alpha.y * (1.0 - alpha.x)) / out_alpha;
    return vec4<f32>(rgb, out_alpha);
}
