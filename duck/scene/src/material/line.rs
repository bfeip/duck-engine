use crate::common::RgbaColor;
use crate::TextureId;

use super::{AlphaMode, MaterialProperties};

/// Unique identifier for a [`LineMaterial`].
pub type LineMaterialId = crate::Id<LineMaterial>;

/// Shading for line (edge / wireframe) primitives.
///
/// Lines are rendered unlit. The color tints an optional base-color texture
/// (multiplied), or stands alone when no texture is set.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineMaterial {
    /// Unique identifier for this material
    pub id: LineMaterialId,

    /// Line color (tints the base-color texture if present)
    color: RgbaColor,

    /// Optional base-color texture (multiplied by `color`)
    base_color_texture: Option<TextureId>,

    /// Generation counter (for change tracking)
    #[cfg_attr(feature = "serde", serde(skip, default = "crate::initial_generation"))]
    generation: u64,
}

impl LineMaterial {
    /// Create a new line material with the given color.
    pub fn new(color: RgbaColor) -> Self {
        Self {
            id: LineMaterialId::new(),
            color,
            base_color_texture: None,
            generation: crate::initial_generation(),
        }
    }

    /// Get the line color.
    pub fn color(&self) -> RgbaColor {
        self.color
    }

    /// Get the base-color texture id, if any.
    pub fn base_color_texture(&self) -> Option<TextureId> {
        self.base_color_texture
    }

    /// Get the material properties used to select shaders and pipeline state.
    ///
    /// Lines carry no explicit alpha mode: a color alpha below 1.0 blends.
    pub fn properties(&self) -> MaterialProperties {
        MaterialProperties {
            base_color_texture: self.base_color_texture.is_some(),
            alpha_mode: AlphaMode::Auto.resolve(self.color.a),
            ..MaterialProperties::UNLIT_OPAQUE
        }
    }

    /// Returns the generation counter. Increments on any mutation.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Return this material with a fresh, globally-unique id.
    ///
    /// Useful when a material is held as a template and instantiated once per
    /// object: scene insertion keys on the id, so each instance needs its own.
    pub fn with_fresh_id(mut self) -> Self {
        self.id = LineMaterialId::new();
        self
    }

    /// Set the line color (chainable).
    pub fn with_color(mut self, color: RgbaColor) -> Self {
        self.set_color(color);
        self
    }

    /// Set the line color, marking the material as dirty.
    pub fn set_color(&mut self, color: RgbaColor) {
        self.color = color;
        self.generation += 1;
    }

    /// Set the base-color texture (chainable).
    pub fn with_base_color_texture(mut self, texture_id: TextureId) -> Self {
        self.set_base_color_texture(texture_id);
        self
    }

    /// Set the base-color texture, marking the material as dirty.
    pub fn set_base_color_texture(&mut self, texture_id: TextureId) {
        self.base_color_texture = Some(texture_id);
        self.generation += 1;
    }
}
