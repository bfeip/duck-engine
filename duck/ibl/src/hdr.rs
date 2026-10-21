//! Loader for Radiance HDR (.hdr) images in RGBE format.
//!
//! The RGBE format stores HDR data as RGB + shared exponent, allowing high dynamic range
//! in a compact format. This module handles both raw and RLE-compressed scanlines.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
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
    let file = File::open(path).with_context(|| format!("Failed to open HDR file: {:?}", path))?;
    let reader = BufReader::new(file);
    load_hdr_from_reader(reader)
}

/// Load an HDR image from raw .hdr file bytes.
pub fn load_hdr_from_bytes(bytes: &[u8]) -> Result<HdrImage> {
    load_hdr_from_reader(std::io::Cursor::new(bytes))
}

/// Load an HDR image from a reader.
pub fn load_hdr_from_reader<R: BufRead>(mut reader: R) -> Result<HdrImage> {
    // Parse header
    let (width, height) = parse_header(&mut reader)?;

    // Read pixel data
    let pixels = read_pixels(&mut reader, width, height)?;

    // Convert RGBE to linear RGB floats
    let data = rgbe_to_float(&pixels);

    Ok(HdrImage {
        width,
        height,
        data,
    })
}

/// Parse the HDR file header, returning (width, height).
fn parse_header<R: BufRead>(reader: &mut R) -> Result<(u32, u32)> {
    let mut line = String::new();

    // Read and verify magic number
    reader.read_line(&mut line)?;
    if !line.starts_with("#?RADIANCE") && !line.starts_with("#?RGBE") {
        bail!("Invalid HDR format: missing RADIANCE or RGBE magic number");
    }

    // Read header lines until we hit an empty line
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            break;
        }

        // We could parse FORMAT here, but we assume 32-bit_rle_rgbe
    }

    // Read resolution line: "-Y height +X width" or "+X width +Y height"
    line.clear();
    reader.read_line(&mut line)?;
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

    Ok((width, height))
}

/// Read all pixel data as RGBE values.
fn read_pixels<R: BufRead>(reader: &mut R, width: u32, height: u32) -> Result<Vec<[u8; 4]>> {
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for _ in 0..height {
        read_scanline(reader, width, &mut pixels)?;
    }

    Ok(pixels)
}

/// Read a single scanline of RGBE data.
fn read_scanline<R: BufRead>(
    reader: &mut R,
    width: u32,
    pixels: &mut Vec<[u8; 4]>,
) -> Result<()> {
    // Read first 4 bytes to determine encoding
    let mut header = [0u8; 4];
    reader.read_exact(&mut header)?;

    // Check for new-style RLE encoding: starts with 2, 2
    if header[0] == 2 && header[1] == 2 && (header[2] as u32) * 256 + (header[3] as u32) == width {
        read_rle_scanline(reader, width, pixels)
    } else {
        // Old format or uncompressed: first pixel is already read
        pixels.push(header);
        read_flat_scanline(reader, width - 1, pixels)
    }
}

/// Read an RLE-encoded scanline (new format).
fn read_rle_scanline<R: Read>(
    reader: &mut R,
    width: u32,
    pixels: &mut Vec<[u8; 4]>,
) -> Result<()> {
    // In RLE format, each channel is stored separately and RLE-encoded
    let mut scanline = vec![[0u8; 4]; width as usize];

    // Read each channel (R, G, B, E)
    for channel in 0..4 {
        let mut x = 0usize;
        while x < width as usize {
            let mut code = [0u8; 1];
            reader.read_exact(&mut code)?;
            let code = code[0];

            if code > 128 {
                // RLE run: repeat the next byte (code - 128) times
                let count = (code - 128) as usize;
                let mut value = [0u8; 1];
                reader.read_exact(&mut value)?;
                for i in 0..count {
                    if x + i >= width as usize {
                        bail!("HDR RLE run exceeds scanline width");
                    }
                    scanline[x + i][channel] = value[0];
                }
                x += count;
            } else {
                // Literal run: read `code` bytes
                let count = code as usize;
                for i in 0..count {
                    if x + i >= width as usize {
                        bail!("HDR literal run exceeds scanline width");
                    }
                    let mut value = [0u8; 1];
                    reader.read_exact(&mut value)?;
                    scanline[x + i][channel] = value[0];
                }
                x += count;
            }
        }
    }

    pixels.extend_from_slice(&scanline);
    Ok(())
}

/// Read an uncompressed scanline (old format).
fn read_flat_scanline<R: Read>(
    reader: &mut R,
    count: u32,
    pixels: &mut Vec<[u8; 4]>,
) -> Result<()> {
    for _ in 0..count {
        let mut pixel = [0u8; 4];
        reader.read_exact(&mut pixel)?;
        pixels.push(pixel);
    }
    Ok(())
}

/// Convert RGBE pixels to linear RGB float values.
fn rgbe_to_float(pixels: &[[u8; 4]]) -> Vec<f32> {
    let mut result = Vec::with_capacity(pixels.len() * 3);

    for rgbe in pixels {
        let (r, g, b) = rgbe_to_rgb(rgbe[0], rgbe[1], rgbe[2], rgbe[3]);
        result.push(r);
        result.push(g);
        result.push(b);
    }

    result
}

/// Convert a single RGBE pixel to RGB floats.
#[inline]
fn rgbe_to_rgb(r: u8, g: u8, b: u8, e: u8) -> (f32, f32, f32) {
    if e == 0 {
        (0.0, 0.0, 0.0)
    } else {
        // RGBE encoding: value = mantissa * 2^(exponent - 128 - 8)
        // The mantissa is stored in 0-255 range, representing 0.0-1.0
        let exp = (e as i32) - 128 - 8;
        let scale = (2.0f32).powi(exp);
        (r as f32 * scale, g as f32 * scale, b as f32 * scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgbe_to_rgb_zero() {
        let (r, g, b) = rgbe_to_rgb(0, 0, 0, 0);
        assert_eq!(r, 0.0);
        assert_eq!(g, 0.0);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn test_rgbe_to_rgb_midgray() {
        // Value of 0.5 with exponent 128 should give roughly 0.5
        // mantissa 128 * 2^(128-128-8) = 128 * 2^-8 = 128/256 = 0.5
        let (r, g, b) = rgbe_to_rgb(128, 128, 128, 128);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.5).abs() < 0.01);
        assert!((b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_rgbe_to_rgb_bright() {
        // High exponent should give bright values
        // 200 * 2^(140-128-8) = 200 * 2^4 = 3200
        let (r, _, _) = rgbe_to_rgb(200, 100, 50, 140);
        assert!(r > 1.0); // Should be HDR
    }
}
