//! The single value that fully describes one surface-shader variant.
//!
//! [`SurfaceConfig`] is a material's [`MaterialProperties`] *as seen by a
//! particular pass under particular scene lighting*: it wraps the device-free,
//! scene-layer [`MaterialProperties`] and adds the only two things that can't
//! live there — whether the scene has IBL, and whether this is a depth prepass.
//!
//! It is the one source of truth driving everything that must agree for a
//! variant: the WESL feature flags ([`features`](SurfaceConfig::features)), the
//! group-2 material bind-group layout (via [`textures`]), whether the
//! pipeline layout includes the IBL group, and which bind-group entries each
//! material builds. Deriving them all from one value is what lets us drop the
//! fallback textures, the `texture_flags` bitmask, and the fixed Color/PBR
//! layout split: a variant declares exactly the bindings it uses.
//!
//! Lighting-only inputs (normal map, metallic-roughness, IBL) are reported as
//! absent for unlit materials, so an unlit variant stays minimal regardless of
//! stray texture assignments.
//!
//! [`textures`]: SurfaceConfig::textures

use crate::scene::resource::{AlphaMode, MaterialProperties};

/// One optional texture channel a surface material can bind.
///
/// The single registry for each channel's GPU/shader identity: its fixed
/// texture+sampler binding slots (matching the `BINDING_MATERIAL_*` constants in
/// `material.wesl`) and its WESL feature flag. Adding a channel = add a variant
/// here and the matching WESL bindings; the layout, bind-group, and feature code
/// all iterate [`ALL`] and read this metadata, so nothing else hard-codes a
/// binding number. (Lit-only gating lives in [`SurfaceConfig::textures`].)
///
/// [`ALL`]: MaterialTextureSlot::ALL
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MaterialTextureSlot {
    BaseColor,
    Normal,
    MetallicRoughness,
}

impl MaterialTextureSlot {
    /// Every channel, in binding-slot order.
    pub const ALL: [MaterialTextureSlot; 3] =
        [Self::BaseColor, Self::Normal, Self::MetallicRoughness];

    /// Binding index of this channel's texture.
    pub fn texture_binding(self) -> u32 {
        match self {
            Self::BaseColor => 1,
            Self::Normal => 3,
            Self::MetallicRoughness => 5,
        }
    }

    /// Binding index of this channel's sampler (always `texture_binding + 1`).
    pub fn sampler_binding(self) -> u32 {
        self.texture_binding() + 1
    }

    /// The WESL feature flag that gates this channel's bindings and sampling.
    pub fn feature(self) -> &'static str {
        match self {
            Self::BaseColor => "has_base_color_texture",
            Self::Normal => "has_normal_texture",
            Self::MetallicRoughness => "has_metallic_roughness_texture",
        }
    }
}

/// Which texture channels a variant binds — one presence flag per
/// [`MaterialTextureSlot`]. Keys the material bind-group-layout cache; iterate
/// [`slots`](TexturePresence::slots) to walk the present channels in binding order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(crate) struct TexturePresence {
    pub base_color: bool,
    pub normal: bool,
    pub metallic_roughness: bool,
}

impl TexturePresence {
    /// Whether `slot` is present.
    pub fn contains(self, slot: MaterialTextureSlot) -> bool {
        match slot {
            MaterialTextureSlot::BaseColor => self.base_color,
            MaterialTextureSlot::Normal => self.normal,
            MaterialTextureSlot::MetallicRoughness => self.metallic_roughness,
        }
    }

    /// The present channels, in binding-slot order.
    pub fn slots(self) -> impl Iterator<Item = MaterialTextureSlot> {
        MaterialTextureSlot::ALL
            .into_iter()
            .filter(move |slot| self.contains(*slot))
    }
}

/// A material variant for one pass: scene-layer [`MaterialProperties`] plus the
/// IBL and depth-prepass context. Hashable so it can key the pipeline cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SurfaceConfig {
    /// The material's own GPU-relevant properties (device-free, scene-layer).
    pub props: MaterialProperties,
    /// Whether the scene has an active environment map. Only matters when lit.
    pub scene_has_ibl: bool,
    /// Depth-only prepass variant (color writes masked off).
    pub depth_prepass: bool,
}

impl SurfaceConfig {
    /// Wrap a material's properties with the scene/pass context.
    pub fn new(props: MaterialProperties, scene_has_ibl: bool, depth_prepass: bool) -> Self {
        Self { props, scene_has_ibl, depth_prepass }
    }

    /// Run PBR lighting (otherwise output the base color directly).
    pub fn lit(&self) -> bool {
        self.props.has_lighting
    }

    /// Which texture channels this variant binds. Lit-only channels (normal,
    /// metallic-roughness) are dropped for unlit materials so the variant — and
    /// its layout — stays minimal.
    pub fn textures(&self) -> TexturePresence {
        let lit = self.lit();
        TexturePresence {
            base_color: self.props.base_color_texture,
            normal: lit && self.props.normal_texture,
            metallic_roughness: lit && self.props.metallic_roughness_texture,
        }
    }

    pub fn double_sided(&self) -> bool {
        self.props.double_sided
    }

    /// Effective IBL: the scene has IBL *and* this material is lit.
    pub fn has_ibl(&self) -> bool {
        self.lit() && self.scene_has_ibl
    }

    /// Whether the alpha-mask discard path is compiled in: explicit `Mask` mode,
    /// or any depth prepass (where blend materials are alpha-tested into depth).
    pub fn alpha_mask(&self) -> bool {
        self.props.alpha_mode == AlphaMode::Mask || self.depth_prepass
    }

    /// The WESL feature flags for this variant. The per-channel `has_*_texture`
    /// flags are sourced from [`MaterialTextureSlot`] so they can't drift from the
    /// bindings they gate.
    pub fn features(&self) -> Vec<(&'static str, bool)> {
        let textures = self.textures();
        let mut features = vec![
            ("lit", self.lit()),
            ("has_ibl", self.has_ibl()),
            ("double_sided", self.double_sided()),
            ("alpha_mask", self.alpha_mask()),
            ("depth_prepass", self.depth_prepass),
        ];
        for slot in MaterialTextureSlot::ALL {
            features.push((slot.feature(), textures.contains(slot)));
        }
        features
    }
}
