// lit_surface.wesl
// PBR shading (Cook-Torrance BRDF, multi-light, optional IBL) for the lit
// surface path. Imported by `surface.wesl` only when the `lit` feature is set.

import package::common::{
    VertexOutput, LightUniform,
    LIGHT_TYPE_POINT, LIGHT_TYPE_DIRECTIONAL, LIGHT_TYPE_SPOT, MAX_LIGHTS
};
import package::camera::camera;
import package::lighting::{lights, calculate_attenuation, calculate_spot_cone_falloff};
import package::material::get_metallic_roughness;
import package::normal_mapping::get_normal;
import package::pbr::calculate_pbr_direct_light;

// Conditionally import IBL or constant ambient
@if(has_ibl)
    import package::ibl::calculate_ibl;
@if(!has_ibl)
    import package::constants::AMBIENT_FACTOR;

// Calculate light direction and radiance for a point light
fn get_point_light_params(
    light: LightUniform,
    world_position: vec3<f32>
) -> array<vec3<f32>, 2> {
    let to_light = light.position - world_position;
    let distance = length(to_light);
    let light_dir = to_light / distance;
    let attenuation = calculate_attenuation(distance, light.range);
    let radiance = light.color * light.intensity * attenuation;
    return array<vec3<f32>, 2>(light_dir, radiance);
}

// Calculate light direction and radiance for a directional light
fn get_directional_light_params(light: LightUniform) -> array<vec3<f32>, 2> {
    let light_dir = -light.direction;
    let radiance = light.color * light.intensity;
    return array<vec3<f32>, 2>(light_dir, radiance);
}

// Calculate light direction and radiance for a spotlight
fn get_spot_light_params(
    light: LightUniform,
    world_position: vec3<f32>
) -> array<vec3<f32>, 2> {
    let to_light = light.position - world_position;
    let distance = length(to_light);
    let light_dir = to_light / distance;
    let attenuation = calculate_attenuation(distance, light.range);
    let cone_falloff = calculate_spot_cone_falloff(
        -light_dir, light.direction,
        light.inner_cone_cos, light.outer_cone_cos
    );
    let radiance = light.color * light.intensity * attenuation * cone_falloff;
    return array<vec3<f32>, 2>(light_dir, radiance);
}

// Shade a fragment with the full PBR pipeline. `base_color` is precomputed by
// the caller (texture * factor) and already alpha-tested.
fn shade_lit(in: VertexOutput, front_facing: bool, base_color: vec4<f32>) -> vec4<f32> {
    let albedo = base_color.rgb;
    let alpha = base_color.a;

    // Get metallic and roughness
    let mr = get_metallic_roughness(in.tex_coords);
    let metallic = mr.x;
    let roughness = mr.y;

    // Get normal (with normal mapping if available)
    var n = get_normal(in.world_position, in.world_normal, in.tex_coords);

    // For double-sided materials, flip normal for back-facing fragments
    @if(double_sided) {
        if (!front_facing) {
            n = -n;
        }
    }

    // View direction (from surface to camera). Orthographic cameras have no
    // eye convergence: every fragment is viewed down the same axis.
    var v = normalize(camera.eye_position - in.world_position);
    if (camera.is_ortho != 0u) {
        v = -camera.view_direction;
    }

    // Accumulate light contributions
    var lo = vec3<f32>(0.0);

    for (var i = 0u; i < lights.light_count && i < MAX_LIGHTS; i = i + 1u) {
        let light = lights.lights[i];
        var light_dir: vec3<f32>;
        var radiance: vec3<f32>;

        if (light.light_type == LIGHT_TYPE_POINT) {
            let params = get_point_light_params(light, in.world_position);
            light_dir = params[0];
            radiance = params[1];
        } else if (light.light_type == LIGHT_TYPE_DIRECTIONAL) {
            let params = get_directional_light_params(light);
            light_dir = params[0];
            radiance = params[1];
        } else if (light.light_type == LIGHT_TYPE_SPOT) {
            let params = get_spot_light_params(light, in.world_position);
            light_dir = params[0];
            radiance = params[1];
        } else {
            continue;
        }

        lo += calculate_pbr_direct_light(
            radiance,
            light_dir,
            v,
            n,
            albedo,
            metallic,
            roughness
        );
    }

    // Ambient term: IBL when available, constant fallback otherwise
    var ambient: vec3<f32>;
    @if(has_ibl) {
        ambient = calculate_ibl(n, v, albedo, metallic, roughness);
    }
    @if(!has_ibl) {
        ambient = vec3<f32>(AMBIENT_FACTOR) * albedo;
    }

    let color = ambient + lo;

    // HDR tonemapping (simple Reinhard)
    let mapped = color / (color + vec3<f32>(1.0));

    // Gamma correction
    let gamma_corrected = pow(mapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, alpha);
}
