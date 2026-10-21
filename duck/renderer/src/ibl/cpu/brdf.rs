//! Split-sum BRDF integration, the content of the baked LUT.
//!
//! `build.rs` compiles this file to bake the LUT.

use super::sampling::{hammersley, importance_sample_ggx};

/// GGX samples per LUT texel.
pub const SAMPLE_COUNT: u32 = 1024;

/// The scale and bias the split sum applies to F0, for one N·V and roughness.
pub fn integrate_brdf(n_dot_v: f32, roughness: f32) -> (f32, f32) {
    // View direction in tangent space where N = (0, 0, 1)
    let v = [
        (1.0 - n_dot_v * n_dot_v).sqrt(),
        0.0f32,
        n_dot_v,
    ];

    let mut a = 0.0f32;
    let mut b = 0.0f32;

    for i in 0..SAMPLE_COUNT {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = importance_sample_ggx(xi, roughness);

        let v_dot_h = (v[0] * h[0] + v[1] * h[1] + v[2] * h[2]).max(0.0);
        // l = 2 * dot(v, h) * h - v
        let l = [
            2.0 * v_dot_h * h[0] - v[0],
            2.0 * v_dot_h * h[1] - v[1],
            2.0 * v_dot_h * h[2] - v[2],
        ];

        let n_dot_l = l[2].max(0.0); // N = (0, 0, 1)
        let n_dot_h = h[2].max(0.0);

        if n_dot_l > 0.0 {
            let g = geometry_smith(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v);
            let fc = (1.0 - v_dot_h).powf(5.0);

            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }

    (a / SAMPLE_COUNT as f32, b / SAMPLE_COUNT as f32)
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let a = roughness;
    let k = (a * a) / 2.0;
    n_dot_v / (n_dot_v * (1.0 - k) + k)
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-3;

    #[test]
    fn smooth_head_on_reflects_everything() {
        let (a, b) = integrate_brdf(0.999, 0.01);
        assert!((a + b - 1.0).abs() < EPSILON, "a + b = {}", a + b);
    }

    #[test]
    fn terms_stay_in_unit_range() {
        for i in 0..8 {
            for j in 0..8 {
                let n_dot_v = (i as f32 + 0.5) / 8.0;
                let roughness = (j as f32 + 0.5) / 8.0;
                let (a, b) = integrate_brdf(n_dot_v, roughness);
                assert!(a >= 0.0 && b >= 0.0, "({n_dot_v}, {roughness}) -> ({a}, {b})");
                assert!(a + b <= 1.0 + EPSILON, "({n_dot_v}, {roughness}) -> ({a}, {b})");
            }
        }
    }
}
