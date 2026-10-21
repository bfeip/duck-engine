//! Equirectangular panorama to cubemap.

use std::f32::consts::{PI, TAU};

use crate::HdrImage;

use super::cube::{lerp, CubeImage, Rgb};
use super::sampling::Vec3;

/// `hdr` resampled onto a cube with `size`² faces.
///
/// The panorama is box-averaged down to about four texels per face width
/// first, so small bright sources keep their energy.
pub(super) fn cube_from_equirect(hdr: &HdrImage, size: u32) -> CubeImage {
    let panorama = Panorama::reduced(hdr, (hdr.width / (4 * size)).max(1));
    CubeImage::from_fn(size, |dir| panorama.sample(dir))
}

struct Panorama {
    width: usize,
    height: usize,
    texels: Vec<Rgb>,
}

impl Panorama {
    /// `hdr` box-averaged by `factor` along each axis.
    fn reduced(hdr: &HdrImage, factor: u32) -> Self {
        let (source_width, source_height) = (hdr.width as usize, hdr.height as usize);
        let width = (source_width / factor as usize).max(1);
        let height = (source_height / factor as usize).max(1);
        let mut texels = Vec::with_capacity(width * height);
        for j in 0..height {
            let rows = j * source_height / height..(j + 1) * source_height / height;
            for i in 0..width {
                let columns = i * source_width / width..(i + 1) * source_width / width;
                let mut sum = [0.0f32; 3];
                for y in rows.clone() {
                    for x in columns.clone() {
                        let p = (y * source_width + x) * 3;
                        sum = [sum[0] + hdr.data[p], sum[1] + hdr.data[p + 1], sum[2] + hdr.data[p + 2]];
                    }
                }
                let count = (rows.len() * columns.len()) as f32;
                texels.push(sum.map(|c| c / count));
            }
        }
        Self { width, height, texels }
    }

    /// The bilinearly filtered radiance in direction `dir`, wrapping around
    /// the horizon and clamped at the poles.
    fn sample(&self, dir: Vec3) -> Rgb {
        let u = dir[2].atan2(dir[0]) / TAU + 0.5;
        let v = dir[1].clamp(-1.0, 1.0).acos() / PI;
        let x = u * self.width as f32 - 0.5;
        let y = (v * self.height as f32 - 0.5).clamp(0.0, (self.height - 1) as f32);
        let (fx, fy) = (x - x.floor(), y - y.floor());
        let x0 = (x.floor() as isize).rem_euclid(self.width as isize) as usize;
        let x1 = (x0 + 1) % self.width;
        let y0 = y as usize;
        let y1 = (y0 + 1).min(self.height - 1);
        let at = |x: usize, y: usize| self.texels[y * self.width + x];
        lerp(lerp(at(x0, y0), at(x1, y0), fx), lerp(at(x0, y1), at(x1, y1), fx), fy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube::{texel_direction, texels};

    const EPSILON: f32 = 2e-2;

    /// An equirect whose radiance is `f` of each texel centre's direction.
    fn panorama(width: u32, height: u32, f: impl Fn(Vec3) -> Rgb) -> HdrImage {
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for j in 0..height {
            let theta = (j as f32 + 0.5) / height as f32 * PI;
            for i in 0..width {
                let phi = ((i as f32 + 0.5) / width as f32 - 0.5) * TAU;
                let dir = [theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()];
                data.extend(f(dir));
            }
        }
        HdrImage { width, height, data }
    }

    #[test]
    fn cube_texels_see_the_panorama_in_their_direction() {
        let radiance = |d: Vec3| d.map(|c| (c + 1.0) * 0.5);
        let cube = cube_from_equirect(&panorama(256, 128, radiance), 16);
        for ((face, x, y), texel) in texels(16).zip(cube.level(0)) {
            let expected = radiance(texel_direction(16, face, x, y));
            assert!(
                (0..3).all(|c| (texel[c] - expected[c]).abs() < EPSILON),
                "texel ({face}, {x}, {y}): {texel:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn reduction_preserves_the_mean() {
        let hdr = panorama(64, 32, |d| [d[0].max(0.0) * 10.0, d[1] * d[1], 1.0]);
        let reduced = Panorama::reduced(&hdr, 4);
        assert_eq!((reduced.width, reduced.height), (16, 8));
        let mean = |texels: &[Rgb]| {
            texels.iter().fold([0.0; 3], |s, t| [s[0] + t[0], s[1] + t[1], s[2] + t[2]])
                .map(|c| c / texels.len() as f32)
        };
        let source: Vec<Rgb> = hdr.data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let (before, after) = (mean(&source), mean(&reduced.texels));
        assert!((0..3).all(|c| (before[c] - after[c]).abs() < 1e-4), "{before:?} vs {after:?}");
    }
}
