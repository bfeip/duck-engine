// Compute shader to generate an irradiance cubemap from an environment cubemap.
//
// The irradiance map stores the diffuse component of IBL by convolving the
// environment with a cosine-weighted hemisphere for each direction.

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const SAMPLE_DELTA: f32 = 0.025; // Sampling step size in radians

@group(0) @binding(0) var env_cubemap: texture_cube<f32>;
@group(0) @binding(1) var env_sampler: sampler;
@group(0) @binding(2) var output_irradiance: texture_storage_2d_array<rgba16float, write>;

// Convert a cube face UV coordinate to a 3D direction vector.
fn cube_uv_to_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x * 2.0 - 1.0;
    let v = uv.y * 2.0 - 1.0;

    switch face {
        case 0u: { return normalize(vec3<f32>( 1.0,   -v,   -u)); } // +X
        case 1u: { return normalize(vec3<f32>(-1.0,   -v,    u)); } // -X
        case 2u: { return normalize(vec3<f32>(   u,  1.0,    v)); } // +Y
        case 3u: { return normalize(vec3<f32>(   u, -1.0,   -v)); } // -Y
        case 4u: { return normalize(vec3<f32>(   u,   -v,  1.0)); } // +Z
        case 5u: { return normalize(vec3<f32>(  -u,   -v, -1.0)); } // -Z
        default: { return vec3<f32>(0.0, 0.0, 1.0); }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_size = textureDimensions(output_irradiance);
    let face = gid.z;

    if gid.x >= output_size.x || gid.y >= output_size.y || face >= 6u {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(output_size.xy);
    let normal = cube_uv_to_direction(face, uv);

    // Build tangent space basis from normal
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(normal.y) > 0.999 {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    // Convolve hemisphere with cosine weighting
    var irradiance = vec3<f32>(0.0);
    var sample_count = 0.0;

    // Sample hemisphere using spherical coordinates
    var phi = 0.0;
    while phi < TAU {
        var theta = 0.0;
        while theta < PI * 0.5 {
            // Spherical to cartesian (in tangent space)
            let tangent_sample = vec3<f32>(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
            );

            // Transform to world space
            let sample_dir = tangent_sample.x * tangent +
                           tangent_sample.y * bitangent +
                           tangent_sample.z * normal;

            // Sample environment and apply cosine weighting
            // cos(theta) is the normal dot sample_dir in tangent space
            // sin(theta) accounts for hemisphere area at this latitude
            let env_color = textureSampleLevel(env_cubemap, env_sampler, sample_dir, 0.0).rgb;
            irradiance += env_color * cos(theta) * sin(theta);
            sample_count += 1.0;

            theta += SAMPLE_DELTA;
        }
        phi += SAMPLE_DELTA;
    }

    // Normalize: integral over hemisphere is PI, so we multiply by PI and divide by sample count
    irradiance = PI * irradiance / sample_count;

    textureStore(output_irradiance, vec2<i32>(gid.xy), i32(face), vec4<f32>(irradiance, 1.0));
}
