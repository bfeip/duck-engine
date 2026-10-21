// Compute shader to generate a pre-filtered environment cubemap for specular IBL.
//
// Uses GGX importance sampling to convolve the environment map.
// Each mip level represents a different roughness value.

const PI: f32 = 3.14159265359;
const SAMPLE_COUNT: u32 = 1024u;

struct PrefilterParams {
    roughness: f32,
    // Rust side pads to 16 bytes for uniform buffer requirements
}

@group(0) @binding(0) var env_cubemap: texture_cube<f32>;
@group(0) @binding(1) var env_sampler: sampler;
@group(0) @binding(2) var output_prefiltered: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: PrefilterParams;

// Convert cube face UV to direction
fn cube_uv_to_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x * 2.0 - 1.0;
    let v = uv.y * 2.0 - 1.0;

    switch face {
        case 0u: { return normalize(vec3<f32>( 1.0,   -v,   -u)); }
        case 1u: { return normalize(vec3<f32>(-1.0,   -v,    u)); }
        case 2u: { return normalize(vec3<f32>(   u,  1.0,    v)); }
        case 3u: { return normalize(vec3<f32>(   u, -1.0,   -v)); }
        case 4u: { return normalize(vec3<f32>(   u,   -v,  1.0)); }
        case 5u: { return normalize(vec3<f32>(  -u,   -v, -1.0)); }
        default: { return vec3<f32>(0.0, 0.0, 1.0); }
    }
}

// Radical inverse using Van der Corput sequence
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // 0x100000000
}

// Hammersley point set for quasi-Monte Carlo sampling
fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

// GGX importance sampling - generates a microfacet normal H
fn importance_sample_ggx(xi: vec2<f32>, normal: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;

    // Sample spherical coordinates
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Spherical to cartesian (tangent space)
    let h_tangent = vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );

    // Build tangent space basis
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(normal.y) > 0.999 {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    // Transform to world space
    return normalize(tangent * h_tangent.x + bitangent * h_tangent.y + normal * h_tangent.z);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_size = textureDimensions(output_prefiltered);
    let face = gid.z;

    if gid.x >= output_size.x || gid.y >= output_size.y || face >= 6u {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(output_size.xy);
    let normal = cube_uv_to_direction(face, uv);
    let view = normal; // Assume view direction equals normal for pre-filtering

    var prefiltered_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = importance_sample_ggx(xi, normal, params.roughness);
        let l = normalize(2.0 * dot(view, h) * h - view);

        let n_dot_l = max(dot(normal, l), 0.0);

        if n_dot_l > 0.0 {
            // Sample environment at light direction
            let env_color = textureSampleLevel(env_cubemap, env_sampler, l, 0.0).rgb;
            prefiltered_color += env_color * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    prefiltered_color = prefiltered_color / total_weight;

    textureStore(output_prefiltered, vec2<i32>(gid.xy), i32(face), vec4<f32>(prefiltered_color, 1.0));
}
