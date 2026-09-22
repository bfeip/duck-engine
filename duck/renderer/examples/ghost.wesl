// X-ray "ghost" shading for a custom pass: a rim-lit silhouette that a
// companion pass draws where the geometry is *occluded*, so a highlighted
// object stays visible through whatever is in front of it.
//
// Engine types come from package::common, so this shader stays in sync with the
// engine automatically.

import package::common::{VertexInput, InstanceInput};
import package::camera::camera;  // CameraUniform global at @group(0) @binding(0)

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal:   vec3<f32>,
    @location(1) world_position: vec3<f32>,
};

@vertex
fn vs_main(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(
        inst.model_matrix_0, inst.model_matrix_1,
        inst.model_matrix_2, inst.model_matrix_3,
    );
    let normal_mat = mat3x3<f32>(
        inst.normal_matrix_0, inst.normal_matrix_1, inst.normal_matrix_2,
    );
    let world_pos = model * vec4<f32>(vert.position, 1.0);

    var out: VertexOutput;
    out.clip_position  = camera.view_proj * world_pos;
    out.world_normal   = normalize(normal_mat * vert.normal);
    out.world_position = world_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ghost = vec3<f32>(0.25, 0.85, 1.0);

    // Fresnel-style rim: opaque at grazing angles, nearly clear head-on, so the
    // ghost reads as a shape rather than a flat wash.
    let view_dir = normalize(camera.eye_position - in.world_position);
    let n_dot_v  = max(dot(in.world_normal, view_dir), 0.0);
    let rim      = pow(1.0 - n_dot_v, 2.0);

    return vec4<f32>(ghost, 0.15 + 0.55 * rim);
}
