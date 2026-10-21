//! GGX importance sampling over a Hammersley set, and the vector math it needs.
//!
//! Self-contained: `build.rs` compiles this file too.

use std::f32::consts::PI;

pub type Vec3 = [f32; 3];

pub fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale(a: Vec3, s: f32) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn normalize(a: Vec3) -> Vec3 {
    scale(a, 1.0 / dot(a, a).sqrt())
}

pub fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    (bits as f32) * 2.328_306_4e-10
}

pub fn hammersley(i: u32, n: u32) -> [f32; 2] {
    [(i as f32) / (n as f32), radical_inverse_vdc(i)]
}

/// A GGX microfacet normal in tangent space (normal = +Z), with α = roughness².
pub fn importance_sample_ggx(xi: [f32; 2], roughness: f32) -> Vec3 {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    [phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta]
}

/// The GGX normal distribution, with α = roughness².
pub fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness * roughness * roughness;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d)
}

/// Tangent and bitangent completing `normal` to an orthonormal basis.
pub fn tangent_basis(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal[1].abs() > 0.999 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
    let tangent = normalize(cross(up, normal));
    (tangent, cross(normal, tangent))
}

/// `v` from the tangent space of `(tangent, bitangent, normal)` to world space.
pub fn to_world(v: Vec3, (tangent, bitangent): (Vec3, Vec3), normal: Vec3) -> Vec3 {
    add(add(scale(tangent, v[0]), scale(bitangent, v[1])), scale(normal, v[2]))
}
