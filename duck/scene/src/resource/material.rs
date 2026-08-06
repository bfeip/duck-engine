//! Per-primitive material types.
//!
//! Rather than one monolithic material, shading is split into three independent,
//! top-level types — [`FaceMaterial`], [`LineMaterial`], [`PointMaterial`] — each
//! with its own id, scene collection, and generation counter. An [`super::Instance`]
//! references up to three of them, one per primitive kind it draws.

use bitflags::bitflags;

mod face;
mod line;
mod point;

pub use face::{FaceMaterial, FaceMaterialId};
pub use line::{LineMaterial, LineMaterialId};
pub use point::{PointMaterial, PointMaterialId};

/// Default roughness factor when not specified
pub const DEFAULT_ROUGHNESS: f32 = 0.5;
/// Default metallic factor when not specified
pub const DEFAULT_METALLIC: f32 = 0.0;
/// Default normal scale when not specified
pub const DEFAULT_NORMAL_SCALE: f32 = 1.0;
/// Default alpha cutoff for mask mode (per glTF spec)
pub const DEFAULT_ALPHA_CUTOFF: f32 = 0.5;

/// Alpha rendering mode
///
/// [`Auto`](AlphaMode::Auto) is the default and infers the mode from the color
/// factor's alpha. The other three are explicit instructions, always honoured as
/// written — see [`resolve`](AlphaMode::resolve).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AlphaMode {
    /// Fully opaque, alpha channel ignored.
    Opaque,
    /// Binary alpha test: alpha >= cutoff is fully opaque, otherwise discarded.
    Mask,
    /// Standard alpha blending (source alpha, one minus source alpha).
    Blend,
    /// Infer from the color factor: blended below 1.0, opaque otherwise.
    #[default]
    Auto,
}

impl AlphaMode {
    /// The mode a material actually renders with, given the alpha of its color
    /// factor.
    ///
    /// Only [`Auto`](AlphaMode::Auto) resolves to anything — the explicit modes
    /// pass through untouched. Asking for [`Blend`](AlphaMode::Blend)
    /// blends even at full alpha, which is what a material carrying its
    /// transparency in a base-color *texture* needs: its factor stays 1.0.
    ///
    /// Never returns `Auto`.
    pub fn resolve(self, factor_alpha: f32) -> AlphaMode {
        match self {
            AlphaMode::Auto if factor_alpha < 1.0 => AlphaMode::Blend,
            AlphaMode::Auto => AlphaMode::Opaque,
            explicit => explicit,
        }
    }
}

bitflags! {
    /// Additional face-material options
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    pub struct MaterialFlags: u32 {
        /// No special flags
        const NONE = 0;
        /// Disable back-face culling and flip normals for back faces
        const DOUBLE_SIDED = 1 << 1;
        /// Disables face lighting. Faces will appear at a constant luminance
        const DO_NOT_LIGHT = 1 << 2;
    }
}

/// Material properties helpful to know during shader generation.
///
/// Used by `ShaderGenerator` and `PipelineManager` to determine which shader
/// variant and pipeline state to use.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MaterialProperties {
    /// Whether lighting calculations should be applied
    pub has_lighting: bool,
    /// Whether the material is double-sided (disables back-face culling, flips normals)
    pub double_sided: bool,
    /// Alpha rendering mode, already resolved — never [`AlphaMode::Auto`]
    pub alpha_mode: AlphaMode,
    /// Whether the material binds a base-color texture
    pub base_color_texture: bool,
    /// Whether the material binds a normal-map texture (lit materials only)
    pub normal_texture: bool,
    /// Whether the material binds a metallic-roughness texture (lit materials only)
    pub metallic_roughness_texture: bool,
}

impl MaterialProperties {
    /// Fixed properties used for untextured line and point primitives: unlit,
    /// opaque, no textures.
    pub const UNLIT_OPAQUE: MaterialProperties = MaterialProperties {
        has_lighting: false,
        double_sided: false,
        alpha_mode: AlphaMode::Opaque,
        base_color_texture: false,
        normal_texture: false,
        metallic_roughness_texture: false,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::RgbaColor;

    fn translucent() -> RgbaColor {
        RgbaColor { r: 0.4, g: 0.7, b: 0.9, a: 0.3 }
    }

    #[test]
    fn auto_infers_from_factor_alpha() {
        assert_eq!(AlphaMode::Auto.resolve(1.0), AlphaMode::Opaque);
        assert_eq!(AlphaMode::Auto.resolve(0.3), AlphaMode::Blend);
    }

    #[test]
    fn explicit_modes_are_never_overridden() {
        // Blending at full alpha is how a base-color texture carrying alpha is
        // expressed, and opaque at partial alpha is glTF's OPAQUE semantic.
        assert_eq!(AlphaMode::Blend.resolve(1.0), AlphaMode::Blend);
        assert_eq!(AlphaMode::Opaque.resolve(0.3), AlphaMode::Opaque);
        assert_eq!(AlphaMode::Mask.resolve(0.3), AlphaMode::Mask);
        assert_eq!(AlphaMode::Mask.resolve(1.0), AlphaMode::Mask);
    }

    #[test]
    fn face_material_resolves_against_its_base_color() {
        let opaque = FaceMaterial::new();
        assert_eq!(opaque.properties().alpha_mode, AlphaMode::Opaque);

        let blended = FaceMaterial::new().with_base_color_factor(translucent());
        assert_eq!(blended.properties().alpha_mode, AlphaMode::Blend);

        let masked =
            FaceMaterial::new().with_base_color_factor(translucent()).with_alpha_mode(AlphaMode::Mask);
        assert_eq!(masked.properties().alpha_mode, AlphaMode::Mask);
    }

    #[test]
    fn line_and_point_materials_blend_when_translucent() {
        assert_eq!(
            LineMaterial::new(RgbaColor::WHITE).properties().alpha_mode,
            AlphaMode::Opaque
        );
        assert_eq!(
            LineMaterial::new(translucent()).properties().alpha_mode,
            AlphaMode::Blend
        );
        assert_eq!(
            PointMaterial::new(translucent()).properties().alpha_mode,
            AlphaMode::Blend
        );
    }
}
