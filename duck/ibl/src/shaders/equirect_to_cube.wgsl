// Compute shader to convert an equirectangular panorama to a cubemap.
//
// Each invocation writes one pixel to one face of the output cubemap.
// The shader samples the equirectangular input at the appropriate direction.

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

@group(0) @binding(0) var equirect: texture_2d<f32>;
@group(0) @binding(1) var equirect_sampler: sampler;
@group(0) @binding(2) var output_cube: texture_storage_2d_array<rgba16float, write>;

// Convert a cube face UV coordinate to a 3D direction vector.
// face: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
fn cube_uv_to_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    // Convert UV from [0,1] to [-1,1]
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

// Convert a 3D direction to equirectangular UV coordinates.
fn direction_to_equirect(dir: vec3<f32>) -> vec2<f32> {
    // Spherical coordinates: phi = atan2(z, x), theta = acos(y)
    // UV mapping: u = phi / TAU + 0.5, v = theta / PI
    let phi = atan2(dir.z, dir.x);
    let theta = acos(clamp(dir.y, -1.0, 1.0));

    let u = phi / TAU + 0.5;
    let v = theta / PI;

    return vec2<f32>(u, v);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_size = textureDimensions(output_cube);
    let face = gid.z;

    // Skip if outside texture bounds
    if gid.x >= output_size.x || gid.y >= output_size.y || face >= 6u {
        return;
    }

    // Convert pixel coordinates to UV [0,1]
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(output_size.xy);

    // Get 3D direction for this pixel on this cube face
    let dir = cube_uv_to_direction(face, uv);

    // Sample equirectangular map
    let equirect_uv = direction_to_equirect(dir);
    let color = textureSampleLevel(equirect, equirect_sampler, equirect_uv, 0.0);

    // Write to output cubemap
    textureStore(output_cube, vec2<i32>(gid.xy), i32(face), color);
}
