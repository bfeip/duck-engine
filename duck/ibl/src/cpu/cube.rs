//! Cubemap images on the CPU: texel layout, face mapping and filtering.

use super::sampling::{normalize, Vec3};

/// The linear RGB radiance of a texel.
pub type Rgb = [f32; 3];

/// A cubemap of linear RGB texels with a mip chain.
///
/// Each level holds its six faces in layer order (+X, −X, +Y, −Y, +Z, −Z),
/// each face row-major: the layout of a six-layer texture upload.
#[derive(Debug, Clone)]
pub struct CubeImage {
    size: u32,
    levels: Vec<Vec<Rgb>>,
}

impl CubeImage {
    /// A single-level cube with `size`² faces, `f` evaluated at each texel
    /// centre's direction.
    pub(super) fn from_fn(size: u32, f: impl Fn(Vec3) -> Rgb) -> Self {
        Self { size, levels: vec![level_from_fn(size, f)] }
    }

    /// A cube from whole levels, the first with `size`² faces, each next one
    /// half the size of the last.
    pub(super) fn from_levels(size: u32, levels: Vec<Vec<Rgb>>) -> Self {
        let cube = Self { size, levels };
        debug_assert!(
            (0..cube.level_count()).all(|l| {
                cube.level(l).len() == 6 * (cube.level_size(l) as usize).pow(2)
            })
        );
        cube
    }

    /// The same cube with box-filtered levels appended down to 1×1.
    pub(super) fn with_mips(mut self) -> Self {
        let mut size = self.level_size(self.level_count() - 1) as usize;
        while size > 1 {
            let half = size / 2;
            let parent = self.levels.last().expect("a cube has a level");
            let mut level = Vec::with_capacity(6 * half * half);
            for face in 0..6 {
                for y in 0..half {
                    for x in 0..half {
                        let texel = |dx, dy| parent[(face * size + 2 * y + dy) * size + 2 * x + dx];
                        let sum = [texel(0, 0), texel(1, 0), texel(0, 1), texel(1, 1)]
                            .into_iter()
                            .fold([0.0; 3], |s, t| [s[0] + t[0], s[1] + t[1], s[2] + t[2]]);
                        level.push(sum.map(|c| c * 0.25));
                    }
                }
            }
            self.levels.push(level);
            size = half;
        }
        self
    }

    /// Face size of the first level.
    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }

    /// Face size of `level`.
    pub fn level_size(&self, level: u32) -> u32 {
        (self.size >> level).max(1)
    }

    /// The texels of `level`, in layer order then row-major.
    pub fn level(&self, level: u32) -> &[Rgb] {
        &self.levels[level as usize]
    }

    /// The first level whose faces are at most `size` texels wide, or the last.
    pub(super) fn level_at_most(&self, size: u32) -> u32 {
        (0..self.level_count())
            .find(|&level| self.level_size(level) <= size)
            .unwrap_or(self.level_count() - 1)
    }

    /// The filtered texel in direction `dir` at fractional level `lod`:
    /// bilinear within a face, clamped to its edges, and linear between levels.
    pub(super) fn sample(&self, dir: Vec3, lod: f32) -> Rgb {
        let (face, a, b) = direction_to_face(dir);
        let lod = lod.clamp(0.0, (self.level_count() - 1) as f32);
        let level = lod.floor() as u32;
        let near = self.bilinear(level, face, a, b);
        let t = lod - level as f32;
        if t == 0.0 {
            return near;
        }
        lerp(near, self.bilinear(level + 1, face, a, b), t)
    }

    fn bilinear(&self, level: u32, face: usize, a: f32, b: f32) -> Rgb {
        let size = self.level_size(level) as usize;
        let max = (size - 1) as f32;
        let x = ((a + 1.0) * 0.5 * size as f32 - 0.5).clamp(0.0, max);
        let y = ((b + 1.0) * 0.5 * size as f32 - 0.5).clamp(0.0, max);
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(size - 1), (y0 + 1).min(size - 1));
        let texels = self.level(level);
        let at = |x, y| texels[(face * size + y) * size + x];
        let top = lerp(at(x0, y0), at(x1, y0), x - x0 as f32);
        let bottom = lerp(at(x0, y1), at(x1, y1), x - x0 as f32);
        lerp(top, bottom, y - y0 as f32)
    }
}

/// One level of `size`² faces, `f` evaluated at each texel centre's direction.
pub(super) fn level_from_fn(size: u32, f: impl Fn(Vec3) -> Rgb) -> Vec<Rgb> {
    texels(size).map(|(face, x, y)| f(texel_direction(size, face, x, y))).collect()
}

/// Every texel of a level with `size`² faces, as `(face, x, y)` in layout order.
pub(super) fn texels(size: u32) -> impl Iterator<Item = (usize, u32, u32)> {
    (0..6).flat_map(move |face| {
        (0..size).flat_map(move |y| (0..size).map(move |x| (face, x, y)))
    })
}

/// The unit direction through the centre of texel `(x, y)` of `face`.
pub(super) fn texel_direction(size: u32, face: usize, x: u32, y: u32) -> Vec3 {
    let to_face = |i: u32| (i as f32 + 0.5) / size as f32 * 2.0 - 1.0;
    face_to_direction(face, to_face(x), to_face(y))
}

/// The solid angle texel `(x, y)` subtends on any face of a `size`² cube.
pub(super) fn texel_solid_angle(size: u32, x: u32, y: u32) -> f32 {
    // Solid angle of the face region from the face centre to (a, b).
    let area = |a: f32, b: f32| (a * b).atan2((a * a + b * b + 1.0).sqrt());
    let to_face = |i: u32| i as f32 / size as f32 * 2.0 - 1.0;
    let (a0, a1) = (to_face(x), to_face(x + 1));
    let (b0, b1) = (to_face(y), to_face(y + 1));
    area(a0, b0) - area(a0, b1) - area(a1, b0) + area(a1, b1)
}

/// The unit direction through face coordinates `(a, b)` ∈ [-1, 1]² of `face`,
/// `a` running along a row and `b` down the rows.
fn face_to_direction(face: usize, a: f32, b: f32) -> Vec3 {
    normalize(match face {
        0 => [1.0, -b, -a],
        1 => [-1.0, -b, a],
        2 => [a, 1.0, b],
        3 => [a, -1.0, -b],
        4 => [a, -b, 1.0],
        _ => [-a, -b, -1.0],
    })
}

/// The face `dir` points into, and its face coordinates there.
fn direction_to_face(dir: Vec3) -> (usize, f32, f32) {
    let [x, y, z] = dir;
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    if ax >= ay && ax >= az {
        if x > 0.0 { (0, -z / ax, -y / ax) } else { (1, z / ax, -y / ax) }
    } else if ay >= az {
        if y > 0.0 { (2, x / ay, z / ay) } else { (3, x / ay, -z / ay) }
    } else if z > 0.0 {
        (4, x / az, -y / az)
    } else {
        (5, -x / az, -y / az)
    }
}

pub(super) fn lerp(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn face_centres_follow_layer_order() {
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (face, axis) in axes.into_iter().enumerate() {
            let dir = face_to_direction(face, 0.0, 0.0);
            assert!((0..3).all(|i| (dir[i] - axis[i]).abs() < EPSILON), "face {face}: {dir:?}");
        }
    }

    #[test]
    fn face_mapping_round_trips() {
        for (face, x, y) in texels(8) {
            let dir = texel_direction(8, face, x, y);
            let (back, a, b) = direction_to_face(dir);
            let again = face_to_direction(back, a, b);
            assert_eq!(back, face, "texel ({face}, {x}, {y})");
            assert!((0..3).all(|i| (dir[i] - again[i]).abs() < EPSILON), "texel ({face}, {x}, {y})");
        }
    }

    #[test]
    fn solid_angles_cover_the_sphere() {
        let total: f32 = texels(16).map(|(_, x, y)| texel_solid_angle(16, x, y)).sum();
        assert!((total - 4.0 * PI).abs() < 1e-4, "total {total}");
    }

    #[test]
    fn mips_preserve_the_mean() {
        let cube = CubeImage::from_fn(8, |d| [d[0] * d[0], d[1].max(0.0), 1.0 + d[2]]).with_mips();
        assert_eq!(cube.level_count(), 4);
        let mean = |level: u32| {
            let texels = cube.level(level);
            texels.iter().fold([0.0; 3], |s, t| [s[0] + t[0], s[1] + t[1], s[2] + t[2]])
                .map(|c| c / texels.len() as f32)
        };
        let top = mean(0);
        for level in 1..4 {
            let m = mean(level);
            assert!((0..3).all(|i| (m[i] - top[i]).abs() < EPSILON), "level {level}: {m:?} vs {top:?}");
        }
    }

    #[test]
    fn sampling_a_texel_centre_returns_the_texel() {
        let cube = CubeImage::from_fn(4, |d| d.map(|c| c.abs()));
        for (face, x, y) in texels(4) {
            let i = (face * 4 + y as usize) * 4 + x as usize;
            let sampled = cube.sample(texel_direction(4, face, x, y), 0.0);
            let texel = cube.level(0)[i];
            assert!((0..3).all(|c| (sampled[c] - texel[c]).abs() < EPSILON), "texel ({face}, {x}, {y})");
        }
    }
}
