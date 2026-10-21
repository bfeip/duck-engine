//! Loader for Radiance HDR (.hdr) images in RGBE format.
//!
//! The RGBE format stores HDR data as RGB + shared exponent, allowing high dynamic range
//! in a compact format. This module handles both raw and RLE-compressed scanlines.

use anyhow::{bail, Context, Result};
use std::io::Read;
use std::path::Path;

/// HDR image data in linear RGB float format.
#[derive(Debug, Clone)]
pub struct HdrImage {
    /// Width of the image in pixels.
    pub width: u32,
    /// Height of the image in pixels.
    pub height: u32,
    /// Pixel data as RGB f32 values (3 floats per pixel, row-major order).
    pub data: Vec<f32>,
}

/// Load an HDR image from a file path.
pub fn load_hdr_from_path(path: impl AsRef<Path>) -> Result<HdrImage> {
    let path = path.as_ref();
    let bytes =
        std::fs::read(path).with_context(|| format!("Failed to open HDR file: {:?}", path))?;
    load_hdr_from_bytes(&bytes)
}

/// Load an HDR image from a reader.
pub fn load_hdr_from_reader<R: Read>(mut reader: R) -> Result<HdrImage> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    load_hdr_from_bytes(&bytes)
}

/// Load an HDR image from raw .hdr file bytes.
pub fn load_hdr_from_bytes(bytes: &[u8]) -> Result<HdrImage> {
    let (width, height, mut pos) = parse_header(bytes)?;
    let row_len = (width as usize)
        .checked_mul(3)
        .filter(|_| width > 0 && height > 0)
        .with_context(|| format!("Invalid HDR dimensions {width}x{height}"))?;
    let mut data = vec![0.0f32; row_len * height as usize];

    // One scanline as four planes: R, G, B and E.
    let mut planes = vec![0u8; 4 * width as usize];
    for row in data.chunks_exact_mut(row_len) {
        pos = read_scanline(bytes, pos, &mut planes)?;
        let (r, rest) = planes.split_at(width as usize);
        let (g, rest) = rest.split_at(width as usize);
        let (b, e) = rest.split_at(width as usize);
        for ((((rgb, &r), &g), &b), &e) in row.chunks_exact_mut(3).zip(r).zip(g).zip(b).zip(e) {
            let scale = EXPONENT_SCALE[e as usize];
            rgb.copy_from_slice(&[r as f32 * scale, g as f32 * scale, b as f32 * scale]);
        }
    }

    Ok(HdrImage { width, height, data })
}

/// `2^(e - 136)` for each exponent byte `e`, the factor taking a mantissa byte
/// to its linear value. Exponent 0 encodes black.
const EXPONENT_SCALE: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut e = 1;
    while e < 256 {
        // Exact powers of two, down through the subnormals.
        let mut scale = 1.0f32;
        let mut n = e as i32 - 136;
        while n > 0 {
            scale *= 2.0;
            n -= 1;
        }
        while n < 0 {
            scale *= 0.5;
            n += 1;
        }
        table[e] = scale;
        e += 1;
    }
    table
};

/// Parse the HDR file header, returning (width, height) and where the pixel
/// data starts.
fn parse_header(bytes: &[u8]) -> Result<(u32, u32, usize)> {
    let mut pos = 0;
    let mut next_line = || -> Result<&[u8]> {
        let rest = &bytes[pos..];
        let end = rest
            .iter()
            .position(|&b| b == b'\n')
            .context("Invalid HDR format: header truncated")?;
        pos += end + 1;
        Ok(&rest[..end])
    };

    // Read and verify magic number
    let magic = next_line()?;
    if !magic.starts_with(b"#?RADIANCE") && !magic.starts_with(b"#?RGBE") {
        bail!("Invalid HDR format: missing RADIANCE or RGBE magic number");
    }

    // Header lines run until an empty one. We could parse FORMAT here, but we
    // assume 32-bit_rle_rgbe.
    while !next_line()?.trim_ascii().is_empty() {}

    // Resolution line: "-Y height +X width" or "+X width +Y height"
    let line = next_line()?;
    let line = std::str::from_utf8(line).context("Invalid HDR resolution line")?;
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.len() != 4 {
        bail!("Invalid HDR resolution line: {}", line.trim());
    }

    // Parse width and height based on format
    let (width, height) = if parts[0] == "-Y" && parts[2] == "+X" {
        let h: u32 = parts[1]
            .parse()
            .with_context(|| "Invalid height in HDR header")?;
        let w: u32 = parts[3]
            .parse()
            .with_context(|| "Invalid width in HDR header")?;
        (w, h)
    } else if parts[0] == "+X" && parts[2] == "+Y" {
        let w: u32 = parts[1]
            .parse()
            .with_context(|| "Invalid width in HDR header")?;
        let h: u32 = parts[3]
            .parse()
            .with_context(|| "Invalid height in HDR header")?;
        (w, h)
    } else {
        bail!("Unsupported HDR resolution format: {}", line.trim());
    };

    Ok((width, height, pos))
}

/// Decode the scanline at `pos` into `planes` (its R, G, B and E planes, one
/// after another), returning the position after it.
fn read_scanline(bytes: &[u8], mut pos: usize, planes: &mut [u8]) -> Result<usize> {
    const TRUNCATED: &str = "HDR pixel data truncated";
    let width = planes.len() / 4;
    let head = bytes.get(pos..pos + 4).context(TRUNCATED)?;

    // New-style RLE: starts with 2, 2, then the width. Each plane is stored
    // separately as runs and literals.
    if head[0] == 2 && head[1] == 2 && usize::from(head[2]) * 256 + usize::from(head[3]) == width {
        pos += 4;
        for plane in planes.chunks_exact_mut(width) {
            let mut x = 0;
            while x < width {
                let code = usize::from(*bytes.get(pos).context(TRUNCATED)?);
                pos += 1;
                if code > 128 {
                    // Run: the next byte repeated (code - 128) times
                    let count = code - 128;
                    let value = *bytes.get(pos).context(TRUNCATED)?;
                    pos += 1;
                    plane
                        .get_mut(x..x + count)
                        .context("HDR RLE run exceeds scanline width")?
                        .fill(value);
                    x += count;
                } else {
                    // Literal: the next `code` bytes
                    let literal = bytes.get(pos..pos + code).context(TRUNCATED)?;
                    pos += code;
                    plane
                        .get_mut(x..x + code)
                        .context("HDR literal run exceeds scanline width")?
                        .copy_from_slice(literal);
                    x += code;
                }
            }
        }
    } else {
        // Uncompressed: interleaved RGBE pixels.
        let pixels = bytes.get(pos..pos + 4 * width).context(TRUNCATED)?;
        pos += 4 * width;
        for (x, pixel) in pixels.chunks_exact(4).enumerate() {
            for (channel, &value) in pixel.iter().enumerate() {
                planes[channel * width + x] = value;
            }
        }
    }
    Ok(pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < EPSILON, "{actual:?} vs {expected:?}");
        }
    }

    /// A one-row HDR file `width` pixels wide around `pixels`.
    fn hdr_file(width: u32, pixels: &[u8]) -> Vec<u8> {
        let mut file = format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X {width}\n").into_bytes();
        file.extend_from_slice(pixels);
        file
    }

    #[test]
    fn exponent_scale() {
        assert_eq!(EXPONENT_SCALE[0].to_bits(), 0);
        // 128 * 2^(128 - 136) = 0.5
        assert!((128.0 * EXPONENT_SCALE[128] - 0.5).abs() < EPSILON);
        // 200 * 2^(140 - 136) = 3200
        assert!((200.0 * EXPONENT_SCALE[140] - 3200.0).abs() < EPSILON);
        // 2^-135, a subnormal: 2^14 times the smallest one.
        assert_eq!(EXPONENT_SCALE[1].to_bits(), 1 << 14);
    }

    #[test]
    fn decodes_flat_scanlines() {
        let image = load_hdr_from_bytes(&hdr_file(2, &[128, 128, 128, 128, 200, 100, 50, 140]))
            .unwrap();
        assert_eq!((image.width, image.height), (2, 1));
        assert_close(&image.data, &[0.5, 0.5, 0.5, 3200.0, 1600.0, 800.0]);
    }

    #[test]
    fn decodes_rle_scanlines() {
        #[rustfmt::skip]
        let pixels = [
            2, 2, 0, 8,
            128 + 8, 64,                  // R: a run of eight 64s
            8, 0, 1, 2, 3, 4, 5, 6, 7,    // G: a literal 0..8
            128 + 4, 10, 4, 1, 2, 3, 4,   // B: four 10s, then a literal 1..=4
            128 + 8, 129,                 // E: 2^-7 throughout
        ];
        let image = load_hdr_from_bytes(&hdr_file(8, &pixels)).unwrap();
        let blue = [10.0, 10.0, 10.0, 10.0, 1.0, 2.0, 3.0, 4.0];
        for x in 0..8 {
            let expected = [64.0, x as f32, blue[x]].map(|c| c / 128.0);
            assert_close(&image.data[3 * x..3 * x + 3], &expected);
        }
    }

    #[test]
    fn rejects_runs_past_the_scanline() {
        let pixels = [2, 2, 0, 8, 128 + 9, 64];
        assert!(load_hdr_from_bytes(&hdr_file(8, &pixels)).is_err());
    }

    #[test]
    fn rejects_truncated_data() {
        let file = hdr_file(2, &[128, 128, 128, 128, 200, 100, 50, 140]);
        assert!(load_hdr_from_bytes(&file[..file.len() - 1]).is_err());
        assert!(load_hdr_from_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n").is_err());
    }
}
