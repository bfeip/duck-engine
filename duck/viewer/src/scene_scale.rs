//! Utility functions for computing navigation scale factors from scene bounds.
//!
//! These functions help ensure consistent navigation behavior regardless of model size
//! by computing scale factors from scene bounding boxes.

use crate::common::Aabb;

/// Default model radius when scene is empty or has zero-sized bounds.
const DEFAULT_MODEL_RADIUS: f32 = 1.0;

/// Minimum model radius to prevent issues with very small models.
const MIN_MODEL_RADIUS: f32 = 1e-6;

/// Maximum model radius to prevent issues with very large models.
const MAX_MODEL_RADIUS: f32 = 1e9;

/// Computes the model radius from scene bounds.
///
/// Returns the bounding sphere radius of the scene, clamped to reasonable limits.
/// If bounds is None (empty scene), returns a default value.
pub fn model_radius_from_bounds(bounds: Option<&Aabb>) -> f32 {
    match bounds {
        Some(aabb) => {
            let radius = aabb.bounding_sphere_radius();
            if radius > 0.0 {
                radius.clamp(MIN_MODEL_RADIUS, MAX_MODEL_RADIUS)
            } else {
                DEFAULT_MODEL_RADIUS
            }
        }
        None => DEFAULT_MODEL_RADIUS,
    }
}

/// Returns the minimum camera radius (closest zoom distance) for a given model radius.
/// Set to 1% of model radius to allow close inspection.
pub fn min_camera_radius(model_radius: f32) -> f32 {
    model_radius * 0.01
}

/// Returns the maximum camera radius (farthest zoom distance) for a given model radius.
/// Set to 100x model radius for viewing entire scene with margin.
pub fn max_camera_radius(model_radius: f32) -> f32 {
    model_radius * 100.0
}

/// Returns the smallest orthographic view half-height (closest zoom) for a given
/// model radius.
/// 
/// Tighter than [`min_camera_radius`]: an orthographic zoom leaves
/// the eye where it is, so it can magnify past the point a dolly would collide.
pub fn min_ortho_half_height(model_radius: f32) -> f32 {
    model_radius * 0.001
}

/// Returns the largest orthographic view half-height (farthest zoom) for a given
/// model radius.
pub fn max_ortho_half_height(model_radius: f32) -> f32 {
    model_radius * 100.0
}

/// Returns the zoom factor for exponential zoom.
/// Each scroll step moves this fraction of the current distance.
pub fn zoom_factor() -> f32 {
    0.1 // 10% per scroll step
}

/// Returns the walk movement speed (units per second) for a given model radius.
/// Set to 10% of model radius per second for comfortable navigation.
pub fn walk_speed(model_radius: f32) -> f32 {
    model_radius * 0.1
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_common::Point3;

    #[test]
    fn test_model_radius_from_bounds_none() {
        assert_eq!(model_radius_from_bounds(None), DEFAULT_MODEL_RADIUS);
    }

    #[test]
    fn test_model_radius_from_bounds_some() {
        let bounds = Aabb::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 10.0),
        );
        let radius = model_radius_from_bounds(Some(&bounds));
        let expected = bounds.bounding_sphere_radius();
        assert!((radius - expected).abs() < 0.01);
    }

    #[test]
    fn test_model_radius_clamped_small() {
        let bounds = Aabb::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1e-10, 1e-10, 1e-10),
        );
        let radius = model_radius_from_bounds(Some(&bounds));
        assert!(radius >= MIN_MODEL_RADIUS);
    }

    #[test]
    fn test_model_radius_clamped_large() {
        let bounds = Aabb::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1e12, 1e12, 1e12),
        );
        let radius = model_radius_from_bounds(Some(&bounds));
        assert!(radius <= MAX_MODEL_RADIUS);
    }

    #[test]
    fn test_scaling_factors() {
        let model_radius = 10.0;

        assert!((min_camera_radius(model_radius) - 0.1).abs() < 0.001);
        assert!((max_camera_radius(model_radius) - 1000.0).abs() < 0.1);
        assert!((walk_speed(model_radius) - 1.0).abs() < 0.01);
        assert_eq!(zoom_factor(), 0.1);
    }
}
