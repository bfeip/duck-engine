use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};

/// Unique identifier for a texture in the scene.
pub type TextureId = super::Id<Texture>;

/// Texture image format.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureFormat {
    Png = 0,
    Jpeg = 1,
    Raw = 2,
}

impl TryFrom<image::ImageFormat> for TextureFormat {
    type Error = anyhow::Error;

    fn try_from(format: image::ImageFormat) -> Result<Self> {
        match format {
            image::ImageFormat::Png => Ok(TextureFormat::Png),
            image::ImageFormat::Jpeg => Ok(TextureFormat::Jpeg),
            _ => anyhow::bail!("unsupported image format: {:?}", format),
        }
    }
}

/// Describes how texture source data is stored.
///
/// Textures can be created from embedded image data or loaded lazily from a file path.
pub enum TextureSource {
    /// Image data embedded in memory.
    Embedded { image: DynamicImage },
    /// Image loaded lazily from a file path.
    /// The image can be released and reloaded by replacing the `OnceLock`.
    File {
        path: PathBuf,
        cache: OnceLock<DynamicImage>,
    },
}

impl Clone for TextureSource {
    fn clone(&self) -> Self {
        match self {
            TextureSource::Embedded { image } => TextureSource::Embedded {
                image: image.clone(),
            },
            TextureSource::File { path, cache } => TextureSource::File {
                path: path.clone(),
                cache: match cache.get() {
                    Some(img) => {
                        let lock = OnceLock::new();
                        let _ = lock.set(img.clone());
                        lock
                    }
                    None => OnceLock::new(),
                },
            },
        }
    }
}

/// An image texture.
///
/// Path-based textures load their image data lazily, when it is first
/// requested.
///
/// # Examples
///
/// ```
/// use duck_engine_scene::resource::Texture;
///
/// let texture = Texture::from_path("texture.png");
/// ```
pub struct Texture {
    /// Unique identifier
    pub id: TextureId,
    /// Source data for the texture
    source: TextureSource,
    /// Generation counter - increments on any mutation
    generation: u64,
}

impl Clone for Texture {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            source: self.source.clone(),
            generation: self.generation,
        }
    }
}

impl Texture {
    /// Create a texture from an embedded image.
    ///
    /// # Arguments
    /// * `image` - The image data to use for this texture
    pub fn from_image(image: DynamicImage) -> Self {
        Self {
            id: super::Id::new(),
            source: TextureSource::Embedded { image },
            generation: 1,
        }
    }

    /// Create a texture from raw RGBA8 pixel data.
    ///
    /// Pixels are row-major, 4 bytes per pixel (R, G, B, A). Convenient for
    /// procedurally generated textures, letting callers build a texture without
    /// depending on the `image` crate directly.
    ///
    /// # Panics
    /// Panics if `pixels.len()` is not exactly `width * height * 4`.
    pub fn from_rgba8(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        let img = image::RgbaImage::from_raw(width, height, pixels)
            .expect("pixel buffer length must equal width * height * 4");
        Self::from_image(image::DynamicImage::ImageRgba8(img))
    }

    /// Create a texture from a file path.
    ///
    /// The image is not loaded immediately - it will be loaded on demand when
    /// `get_image()` is called.
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            id: super::Id::new(),
            source: TextureSource::File {
                path: path.into(),
                cache: OnceLock::new(),
            },
            generation: 1,
        }
    }

    /// Load and return a reference to the image data.
    ///
    /// For path-based textures, this loads the image from disk on first access.
    /// The loaded image is cached for future access.
    ///
    /// # Errors
    /// Returns an error if the image cannot be loaded from the path.
    pub fn get_image(&self) -> Result<&DynamicImage> {
        match &self.source {
            TextureSource::Embedded { image, .. } => Ok(image),
            TextureSource::File { path, cache } => {
                if let Some(img) = cache.get() {
                    return Ok(img);
                }
                let img = image::open(path)
                    .with_context(|| format!("Failed to load texture from {:?}", path))?;
                // If another thread set it first, our image is dropped and
                // we use theirs. This is correct — just a rare double-load.
                let _ = cache.set(img);
                Ok(cache.get().expect("just set or another thread set it"))
            }
        }
    }


    /// Replace the texture's image data.
    pub fn set_image(&mut self, image: DynamicImage) {
        self.source = TextureSource::Embedded { image };
        self.generation += 1;
    }

    /// Get the texture's unique identifier.
    pub fn id(&self) -> TextureId {
        self.id
    }

    /// Get the texture dimensions, if known.
    ///
    /// Returns `None` if the texture was created from a path and hasn't been loaded yet.
    /// Call `get_image()` first to trigger loading if dimensions are needed.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        match &self.source {
            TextureSource::Embedded { image, .. } => Some(image.dimensions()),
            TextureSource::File { cache, .. } => cache.get().map(|img| img.dimensions()),
        }
    }

    /// Returns the current generation counter.
    ///
    /// This value increments on any mutation to the texture data.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Release cached image data to free memory.
    ///
    /// For path-based textures, this releases the cached image data while
    /// retaining the path for potential reloading. The image can be reloaded
    /// by calling `get_image()` again.
    ///
    /// For embedded textures, this has no effect (the image cannot be reloaded).
    pub fn release_image_cache(&mut self) {
        if let TextureSource::File { path, .. } = &self.source {
            let path = path.clone();
            self.source = TextureSource::File {
                path,
                cache: OnceLock::new(),
            };
        }
        // Embedded textures cannot release their image data
    }

    /// Get the source path if this texture was created from a path.
    pub fn source_path(&self) -> Option<&Path> {
        match &self.source {
            TextureSource::File { path, .. } => Some(path),
            TextureSource::Embedded { .. } => None,
        }
    }

}

#[cfg(feature = "serde")]
fn detect_texture_format(bytes: &[u8]) -> Option<TextureFormat> {
    image::guess_format(bytes).ok().and_then(|f| TextureFormat::try_from(f).ok())
}

#[cfg(feature = "serde")]
impl serde::Serialize for Texture {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{Error, SerializeStruct};
        use std::io::Cursor;

        let image = self.get_image().map_err(Error::custom)?;
        let (width, height) = image.dimensions();

        // Last-resort: PNG-encode the decoded image.
        let encode_png = || -> std::result::Result<Vec<u8>, S::Error> {
            let mut buf = Vec::new();
            image
                .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
                .map_err(Error::custom)?;
            Ok(buf)
        };

        let (data, format) = if let Some(path) = self.source_path() {
            // File texture: read the already-compressed file bytes directly.
            let bytes = std::fs::read(path).map_err(Error::custom)?;
            if let Some(fmt) = detect_texture_format(&bytes) {
                (bytes, fmt)
            } else {
                (encode_png()?, TextureFormat::Png)
            }
        } else {
            // Decoded-only texture: re-encode as PNG.
            (encode_png()?, TextureFormat::Png)
        };

        let mut s = serializer.serialize_struct("Texture", 5)?;
        s.serialize_field("id", &self.id)?;
        s.serialize_field("format", &format)?;
        s.serialize_field("width", &width)?;
        s.serialize_field("height", &height)?;
        s.serialize_field("data", &data)?;
        s.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Texture {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        #[derive(serde::Deserialize)]
        struct TextureData {
            id: TextureId,
            format: TextureFormat,
            width: u32,
            height: u32,
            data: Vec<u8>,
        }

        let td = TextureData::deserialize(deserializer)?;

        let image = match td.format {
            TextureFormat::Png | TextureFormat::Jpeg => {
                image::load_from_memory(&td.data).map_err(|e| Error::custom(e.to_string()))?
            }
            TextureFormat::Raw => {
                let rgba = image::RgbaImage::from_raw(td.width, td.height, td.data)
                    .ok_or_else(|| Error::custom("invalid raw image dimensions"))?;
                DynamicImage::ImageRgba8(rgba)
            }
        };

        Ok(Texture {
            id: td.id,
            source: TextureSource::Embedded { image },
            generation: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_rgba8_round_trips_pixels() {
        // 2x1 image: one red opaque pixel, one transparent pixel.
        let pixels = vec![255, 0, 0, 255, 0, 0, 0, 0];
        let texture = Texture::from_rgba8(2, 1, pixels.clone());

        assert_eq!(texture.dimensions(), Some((2, 1)));
        let img = texture.get_image().unwrap().to_rgba8();
        assert_eq!(img.as_raw(), &pixels);
    }

    #[test]
    #[should_panic]
    fn from_rgba8_rejects_wrong_length() {
        // 3 bytes is not 1*1*4.
        Texture::from_rgba8(1, 1, vec![0, 0, 0]);
    }
}
