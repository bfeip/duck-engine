// common.wesl
// Common type definitions used by all shaders

struct CameraUniform {
    view_proj: mat4x4<f32>,
    eye_position: vec3<f32>,
    _padding: u32,
    // Camera forward in world space; the view vector for orthographic cameras,
    // where eye_position does not define one.
    view_direction: vec3<f32>,
    is_ortho: u32,
};

// Light type constants
const LIGHT_TYPE_POINT: u32 = 0u;
const LIGHT_TYPE_DIRECTIONAL: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;

const MAX_LIGHTS: u32 = 8u;

// Single light uniform (64 bytes, matches Rust LightUniform)
struct LightUniform {
    light_type: u32,
    range: f32,
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    position: vec3<f32>,
    intensity: f32,
    direction: vec3<f32>,
    _padding1: f32,
    color: vec3<f32>,
    _padding2: f32,
};

// Lights array uniform (528 bytes, matches Rust LightsArrayUniform)
// WGSL adds implicit padding after light_count to align the array to 16 bytes.
// The Rust struct has explicit [u32; 3] padding to match this layout.
struct LightsUniform {
    light_count: u32,
    // implicit 12 bytes padding here (WGSL aligns array to 16 bytes)
    lights: array<LightUniform, 8>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec3<f32>,
    @location(2) normal: vec3<f32>
};

struct InstanceInput {
    @location(3) model_matrix_0: vec4<f32>,
    @location(4) model_matrix_1: vec4<f32>,
    @location(5) model_matrix_2: vec4<f32>,
    @location(6) model_matrix_3: vec4<f32>,
    @location(7) normal_matrix_0: vec3<f32>,
    @location(8) normal_matrix_1: vec3<f32>,
    @location(9) normal_matrix_2: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>
};
