// pbr.wesl
// Physically Based Rendering BRDF functions (Cook-Torrance with GGX)

import package::constants::PI;
import package::common::LightUniform;

// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Fresnel-Schlick with roughness term for IBL
// At grazing angles, rough surfaces have less Fresnel effect
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let one_minus_roughness = vec3<f32>(1.0 - roughness);
    return f0 + (max(one_minus_roughness, f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// GGX/Trowbridge-Reitz Normal Distribution Function
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith's Schlick-GGX Geometry Function (single direction)
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

// Smith's Geometry Function (combines view and light directions)
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx_v = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx_l = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx_v * ggx_l;
}

// Calculate PBR lighting contribution from a single point light
fn calculate_pbr_direct_light(
    light_color: vec3<f32>,
    light_dir: vec3<f32>,
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let h = normalize(view_dir + light_dir);

    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let n_dot_v = max(dot(normal, view_dir), 0.001);
    let n_dot_h = max(dot(normal, h), 0.0);
    let h_dot_v = max(dot(h, view_dir), 0.0);

    // F0 = reflectance at normal incidence
    // Dielectrics use 0.04, metals use albedo
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Cook-Torrance BRDF components
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);

    // Specular term
    let numerator = d * g * f;
    let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
    let specular = numerator / denominator;

    // Energy conservation: kS + kD = 1
    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

    // Lambertian diffuse
    let diffuse = k_d * albedo / PI;

    return (diffuse + specular) * light_color * n_dot_l;
}

// Analytic approximation of the split-sum environment BRDF (Lazarov), standing
// in for the prefiltered BRDF LUT so ambient needs no environment bindings.
fn env_brdf_approx(f0: vec3<f32>, roughness: f32, n_dot_v: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

// Ambient contribution of a hemisphere light: a sky-to-ground gradient sampled
// along the surface normal for diffuse and the reflection vector for specular.
// The gradient value is irradiance, so the diffuse term takes no 1/PI.
fn calculate_hemisphere_ambient(
    light: LightUniform,
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    // `direction` runs sky to ground, matching directional lights.
    let up = -light.direction;
    let sky = light.color;
    let ground = light.secondary_color;

    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let diffuse_env = mix(ground, sky, dot(normal, up) * 0.5 + 0.5);
    let k_d = (vec3<f32>(1.0) - fresnel_schlick_roughness(n_dot_v, f0, roughness))
        * (1.0 - metallic);
    let diffuse = k_d * albedo * diffuse_env;

    let reflect_dir = reflect(-view_dir, normal);
    let specular_env = mix(ground, sky, dot(reflect_dir, up) * 0.5 + 0.5);
    let specular = specular_env * env_brdf_approx(f0, roughness, n_dot_v);

    return (diffuse + specular) * light.intensity;
}
