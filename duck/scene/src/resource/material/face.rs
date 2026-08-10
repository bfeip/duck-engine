use crate::common::RgbaColor;
use crate::resource::TextureHandle;
use crate::resource::TextureId;

use super::{
    AlphaMode, MaterialFlags, MaterialProperties, DEFAULT_ALPHA_CUTOFF, DEFAULT_METALLIC,
    DEFAULT_NORMAL_SCALE, DEFAULT_ROUGHNESS,
};

/// Unique identifier for a [`FaceMaterial`].
pub type FaceMaterialId = crate::resource::Id<FaceMaterial>;

/// Physically-based shading for triangle (face) primitives.
///
/// Supports base color, normal maps, and metallic-roughness textures, with
/// scalar factors as fallbacks when textures are absent.
///
/// # Examples
///
/// ```
/// use duck_engine_scene::SceneData;
/// use duck_engine_scene::resource::FaceMaterial;
/// use duck_engine_scene::common::RgbaColor;
///
/// let face = FaceMaterial::new().with_base_color_factor(RgbaColor::RED);
///
/// let mut scene = SceneData::new();
/// let face_id = scene.add_face_material(face);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FaceMaterial {
    /// Unique identifier for this material
    pub id: FaceMaterialId,

    /// Base color texture (albedo).
    base_color_texture: Option<TextureHandle>,
    /// Normal map texture.
    normal_texture: Option<TextureHandle>,
    /// Metallic-roughness texture (G=roughness, B=metallic per glTF spec).
    metallic_roughness_texture: Option<TextureHandle>,

    /// Base color factor (multiplied with texture if present)
    base_color_factor: RgbaColor,
    /// Metallic factor (0.0 = dielectric, 1.0 = metal)
    metallic_factor: f32,
    /// Roughness factor (0.0 = smooth, 1.0 = rough)
    roughness_factor: f32,
    /// Normal map scale
    normal_scale: f32,

    /// Rendering flags
    flags: MaterialFlags,

    /// Alpha rendering mode
    alpha_mode: AlphaMode,
    /// Alpha cutoff threshold for Mask mode
    alpha_cutoff: f32,

    /// Generation counter (for change tracking)
    #[cfg_attr(feature = "serde", serde(skip, default = "crate::initial_generation"))]
    generation: u64,
}

impl FaceMaterial {
    /// Create a new face material with default values.
    ///
    /// Defaults: white base color, metallic=0.0, roughness=0.5, no textures.
    pub fn new() -> Self {
        Self {
            id: FaceMaterialId::new(),
            base_color_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            base_color_factor: RgbaColor::WHITE,
            metallic_factor: DEFAULT_METALLIC,
            roughness_factor: DEFAULT_ROUGHNESS,
            normal_scale: DEFAULT_NORMAL_SCALE,
            flags: MaterialFlags::NONE,
            alpha_mode: AlphaMode::Auto,
            alpha_cutoff: DEFAULT_ALPHA_CUTOFF,
            generation: crate::initial_generation(),
        }
    }

    // ========== Getter methods ==========

    /// Get the base color texture ID.
    pub fn base_color_texture(&self) -> Option<TextureId> {
        self.base_color_texture.as_ref().map(TextureHandle::id)
    }

    /// Get the normal map texture ID.
    pub fn normal_texture(&self) -> Option<TextureId> {
        self.normal_texture.as_ref().map(TextureHandle::id)
    }

    /// Get the metallic-roughness texture ID.
    pub fn metallic_roughness_texture(&self) -> Option<TextureId> {
        self.metallic_roughness_texture.as_ref().map(TextureHandle::id)
    }

    /// Get the base color factor.
    pub fn base_color_factor(&self) -> RgbaColor {
        self.base_color_factor
    }

    /// Get the metallic factor.
    pub fn metallic_factor(&self) -> f32 {
        self.metallic_factor
    }

    /// Get the roughness factor.
    pub fn roughness_factor(&self) -> f32 {
        self.roughness_factor
    }

    /// Get the normal map scale.
    pub fn normal_scale(&self) -> f32 {
        self.normal_scale
    }

    /// Get the flags.
    pub fn flags(&self) -> MaterialFlags {
        self.flags
    }

    /// Get the requested alpha rendering mode, which may be
    /// [`AlphaMode::Auto`]. For the mode this material renders with, see
    /// [`properties`](Self::properties).
    pub fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    /// Get the alpha cutoff threshold (used in Mask mode).
    pub fn alpha_cutoff(&self) -> f32 {
        self.alpha_cutoff
    }

    /// Returns the generation counter.
    ///
    /// Increments on any mutation; consumers compare it against the last value
    /// they saw to detect changes.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Get the material properties used to select shaders and pipeline state.
    ///
    /// The alpha mode is resolved against the base color factor, so a default
    /// (`Auto`) material with `a < 1.0` renders blended. Alpha carried by a
    /// base-color *texture* cannot be inferred — the factor stays 1.0 — so those
    /// materials need an explicit [`with_alpha_mode`](Self::with_alpha_mode).
    pub fn properties(&self) -> MaterialProperties {
        MaterialProperties {
            has_lighting: !self.flags.contains(MaterialFlags::DO_NOT_LIGHT),
            double_sided: self.flags.contains(MaterialFlags::DOUBLE_SIDED),
            alpha_mode: self.alpha_mode.resolve(self.base_color_factor.a),
            base_color_texture: self.base_color_texture.is_some(),
            normal_texture: self.normal_texture.is_some(),
            metallic_roughness_texture: self.metallic_roughness_texture.is_some(),
        }
    }

    // ========== Builder methods (chainable) ==========

    /// Return this material with a fresh, globally-unique id.
    ///
    /// Useful when a material is held as a template and instantiated once per
    /// object: scene insertion keys on the id, so each instance needs its own.
    pub fn with_fresh_id(mut self) -> Self {
        self.id = FaceMaterialId::new();
        self
    }

    /// Set the base color texture.
    pub fn with_base_color_texture(mut self, texture: TextureHandle) -> Self {
        self.set_base_color_texture(texture);
        self
    }

    /// Set the normal map texture.
    pub fn with_normal_texture(mut self, texture: TextureHandle) -> Self {
        self.set_normal_texture(texture);
        self
    }

    /// Set the metallic-roughness texture.
    pub fn with_metallic_roughness_texture(mut self, texture: TextureHandle) -> Self {
        self.set_metallic_roughness_texture(texture);
        self
    }

    /// Set the base color factor (multiplied with texture if present).
    pub fn with_base_color_factor(mut self, color: RgbaColor) -> Self {
        self.set_base_color_factor(color);
        self
    }

    /// Set the metallic factor (0.0 = dielectric, 1.0 = metal).
    pub fn with_metallic_factor(mut self, metallic: f32) -> Self {
        self.set_metallic_factor(metallic);
        self
    }

    /// Set the roughness factor (0.0 = smooth, 1.0 = rough).
    pub fn with_roughness_factor(mut self, roughness: f32) -> Self {
        self.set_roughness_factor(roughness);
        self
    }

    /// Set the normal map scale.
    pub fn with_normal_scale(mut self, scale: f32) -> Self {
        self.set_normal_scale(scale);
        self
    }

    /// Set the material flags.
    pub fn with_flags(mut self, flags: MaterialFlags) -> Self {
        self.set_flags(flags);
        self
    }

    /// Set the alpha rendering mode.
    pub fn with_alpha_mode(mut self, alpha_mode: AlphaMode) -> Self {
        self.set_alpha_mode(alpha_mode);
        self
    }

    /// Set the alpha cutoff threshold (used in Mask mode).
    pub fn with_alpha_cutoff(mut self, alpha_cutoff: f32) -> Self {
        self.set_alpha_cutoff(alpha_cutoff);
        self
    }

    // ========== Mutation methods (increment generation) ==========

    /// Set the base color texture, marking the material as dirty.
    pub fn set_base_color_texture(&mut self, texture: TextureHandle) {
        self.base_color_texture = Some(texture);
        self.generation += 1;
    }

    /// Set the base color factor, marking the material as dirty.
    pub fn set_base_color_factor(&mut self, color: RgbaColor) {
        self.base_color_factor = color;
        self.generation += 1;
    }

    /// Set the normal map texture, marking the material as dirty.
    pub fn set_normal_texture(&mut self, texture: TextureHandle) {
        self.normal_texture = Some(texture);
        self.generation += 1;
    }

    /// Set the metallic-roughness texture, marking the material as dirty.
    pub fn set_metallic_roughness_texture(&mut self, texture: TextureHandle) {
        self.metallic_roughness_texture = Some(texture);
        self.generation += 1;
    }

    /// Set the metallic factor, marking the material as dirty.
    pub fn set_metallic_factor(&mut self, metallic: f32) {
        self.metallic_factor = metallic;
        self.generation += 1;
    }

    /// Set the roughness factor, marking the material as dirty.
    pub fn set_roughness_factor(&mut self, roughness: f32) {
        self.roughness_factor = roughness;
        self.generation += 1;
    }

    /// Set the normal map scale, marking the material as dirty.
    pub fn set_normal_scale(&mut self, scale: f32) {
        self.normal_scale = scale;
        self.generation += 1;
    }

    /// Set the material flags, marking the material as dirty.
    pub fn set_flags(&mut self, flags: MaterialFlags) {
        self.flags = flags;
        self.generation += 1;
    }

    /// Set the alpha rendering mode, marking the material as dirty.
    pub fn set_alpha_mode(&mut self, alpha_mode: AlphaMode) {
        self.alpha_mode = alpha_mode;
        self.generation += 1;
    }

    /// Set the alpha cutoff threshold, marking the material as dirty.
    pub fn set_alpha_cutoff(&mut self, alpha_cutoff: f32) {
        self.alpha_cutoff = alpha_cutoff;
        self.generation += 1;
    }
}

impl Default for FaceMaterial {
    fn default() -> Self {
        Self::new()
    }
}
