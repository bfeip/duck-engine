use crate::common::RgbaColor;
use duck_engine_common::{InnerSpace, Matrix4, Vector3};

/// Maximum number of lights supported in the scene.
pub const MAX_LIGHTS: usize = 8;

/// Light type identifiers.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LightType {
    /// Point light (radiates in all directions from a position).
    Point = 0,
    /// Directional light (parallel rays, like sunlight).
    Directional = 1,
    /// Spotlight (cone of light from a position in a direction).
    Spot = 2,
    /// Hemisphere light (ambient gradient between a sky and a ground color).
    Hemisphere = 3,
}

/// The photometric properties of a light source in the scene.
///
/// Position and direction are **not** stored here — they are derived from the node's
/// world transform during rendering:
/// - Position (Point, Spot): translation column of the world transform matrix.
/// - Direction (Directional, Spot, Hemisphere): negative Z-axis of the world rotation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Light {
    /// Point light that radiates in all directions.
    Point {
        /// Light color.
        color: RgbaColor,
        /// Intensity multiplier.
        intensity: f32,
        /// Maximum range of the light. 0.0 means infinite range.
        range: f32,
    },
    /// Directional light with parallel rays (like sunlight).
    Directional {
        /// Light color.
        color: RgbaColor,
        /// Intensity multiplier.
        intensity: f32,
    },
    /// Spotlight with a cone of light.
    Spot {
        /// Light color.
        color: RgbaColor,
        /// Intensity multiplier.
        intensity: f32,
        /// Maximum range of the light. 0.0 means infinite range.
        range: f32,
        /// Inner cone angle in radians (full intensity).
        inner_cone_angle: f32,
        /// Outer cone angle in radians (zero intensity).
        outer_cone_angle: f32,
    },
    /// Ambient light graded between a sky and a ground color along an axis.
    ///
    /// The direction is the axis pointing from sky toward ground, following the
    /// same convention as `Directional`.
    Hemisphere {
        /// Color arriving along the sky axis.
        sky_color: RgbaColor,
        /// Color arriving along the ground axis.
        ground_color: RgbaColor,
        /// Intensity multiplier.
        intensity: f32,
    },
}

impl Light {
    /// Extracts world-space position and direction from the node's world transform matrix.
    ///
    /// - Position: translation column (W) of the matrix. Relevant for `Point` and `Spot`.
    /// - Direction: negative Z-axis of the matrix. Relevant for `Directional` and `Spot`.
    ///
    /// The direction is normalized; falls back to `[0, 0, -1]` for a degenerate matrix.
    pub fn world_position_and_direction(
        world_transform: &Matrix4,
    ) -> (Vector3, Vector3) {
        let position = world_transform.w.truncate();
        let neg_z = -world_transform.z.truncate();
        let direction = if neg_z.magnitude2() > 0.0 {
            neg_z.normalize()
        } else {
            Vector3::new(0.0, 0.0, -1.0)
        };
        (position, direction)
    }

    /// Creates a new point light.
    pub fn point(color: RgbaColor, intensity: f32) -> Self {
        Self::Point { color, intensity, range: 0.0 }
    }

    /// Creates a new point light with explicit range.
    pub fn point_with_range(color: RgbaColor, intensity: f32, range: f32) -> Self {
        Self::Point { color, intensity, range }
    }

    /// Creates a new directional light.
    pub fn directional(color: RgbaColor, intensity: f32) -> Self {
        Self::Directional { color, intensity }
    }

    /// Creates a new spotlight.
    pub fn spot(color: RgbaColor, intensity: f32, inner_cone_angle: f32, outer_cone_angle: f32) -> Self {
        Self::Spot { color, intensity, range: 0.0, inner_cone_angle, outer_cone_angle }
    }

    /// Creates a new spotlight with explicit range.
    pub fn spot_with_range(
        color: RgbaColor,
        intensity: f32,
        range: f32,
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    ) -> Self {
        Self::Spot { color, intensity, range, inner_cone_angle, outer_cone_angle }
    }

    /// Creates a new hemisphere light.
    pub fn hemisphere(sky_color: RgbaColor, ground_color: RgbaColor, intensity: f32) -> Self {
        Self::Hemisphere { sky_color, ground_color, intensity }
    }

    /// The type identifier for this light.
    pub fn light_type(&self) -> LightType {
        match self {
            Self::Point { .. } => LightType::Point,
            Self::Directional { .. } => LightType::Directional,
            Self::Spot { .. } => LightType::Spot,
            Self::Hemisphere { .. } => LightType::Hemisphere,
        }
    }
}
