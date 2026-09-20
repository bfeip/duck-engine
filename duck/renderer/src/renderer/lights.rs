//! Resolving [`PositionedLight`]s to world space for upload.

use crate::scene::{Light, LightSpace, PositionedCamera, PositionedLight, SceneData};

/// A light resolved to world space: photometric data paired with the position
/// and direction derived from its pose.
///
/// Internal to the renderer — callers supply [`PositionedLight`]s and the
/// render entry points resolve them.
pub(crate) struct ResolvedLight {
    pub light: Light,
    /// World-space position (relevant for Point and Spot lights).
    pub position: [f32; 3],
    /// World-space direction pointing away from lit surfaces (relevant for Directional and Spot lights).
    pub direction: [f32; 3],
}

/// Resolves `light` against the space it is posed in.
///
/// Returns `None` for a [`LightSpace::Node`] light whose node is missing or
/// detached — an unresolvable anchor drops the light rather than stranding it
/// at the origin.
pub(crate) fn resolve_light(
    light: &PositionedLight,
    scene: &SceneData,
    camera: &PositionedCamera,
) -> Option<ResolvedLight> {
    let anchor = match light.space {
        LightSpace::World => None,
        LightSpace::Camera => Some(camera.pose_transform().to_matrix()),
        LightSpace::Node(node_id) => {
            if !scene.is_node_attached(node_id) {
                return None;
            }
            // `nodes_transform` computes on demand rather than reading a cache
            // the render traversal may not have filled yet this frame.
            Some(scene.nodes_transform(node_id)?)
        }
    };

    let local = light.transform.to_matrix();
    let world = match anchor {
        Some(anchor) => anchor * local,
        None => local,
    };
    let (position, direction) = Light::world_position_and_direction(&world);
    Some(ResolvedLight {
        light: light.light.clone(),
        position: position.into(),
        direction: direction.into(),
    })
}

/// Resolves every light that can be resolved, preserving order.
pub(crate) fn resolve_lights(
    lights: &[PositionedLight],
    scene: &SceneData,
    camera: &PositionedCamera,
) -> Vec<ResolvedLight> {
    lights.iter().filter_map(|l| resolve_light(l, scene, camera)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Projection;
    use crate::scene::common::{Deg, Point3, Quaternion, RgbaColor, Rotation3, Transform, Vector3};
    use crate::scene::resource::NodeFlags;

    const EPSILON: f32 = 1e-5;

    fn white() -> RgbaColor {
        RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
    }

    fn test_camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            projection: Projection::Perspective { fovy: 45.0, znear: 0.1, zfar: 100.0 },
        }
    }

    fn assert_vec3(actual: [f32; 3], expected: [f32; 3]) {
        for i in 0..3 {
            assert!(
                (actual[i] - expected[i]).abs() < EPSILON,
                "component {i}: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn world_space_uses_the_transform_directly() {
        let scene = SceneData::new();
        let light = PositionedLight::world(
            Light::point(white(), 1.0),
            Transform::from_position(Point3::new(1.0, 2.0, 3.0)),
        );

        let resolved = resolve_light(&light, &scene, &test_camera()).unwrap();
        assert_vec3(resolved.position, [1.0, 2.0, 3.0]);
        assert_vec3(resolved.direction, [0.0, 0.0, -1.0]);
    }

    #[test]
    fn camera_space_composes_with_the_camera_pose() {
        let scene = SceneData::new();
        let camera = test_camera();

        // An identity camera-space light sits at the eye, aimed where the
        // camera looks.
        let light =
            PositionedLight::camera(Light::directional(white(), 1.0), Transform::IDENTITY);
        let resolved = resolve_light(&light, &scene, &camera).unwrap();
        assert_vec3(resolved.position, [0.0, 0.0, 5.0]);
        assert_vec3(resolved.direction, [0.0, 0.0, -1.0]);

        // Yawing the light 90 degrees in camera space aims it along -X, since
        // the camera itself is unrotated here.
        let turned = PositionedLight::camera(
            Light::directional(white(), 1.0),
            Transform::from_rotation(Quaternion::from_angle_y(Deg(90.0))),
        );
        let resolved = resolve_light(&turned, &scene, &camera).unwrap();
        assert_vec3(resolved.direction, [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn node_space_composes_through_a_hierarchy() {
        let mut scene = SceneData::new();
        let parent = scene
            .add_node(None, None, Transform::from_position(Point3::new(10.0, 0.0, 0.0)), NodeFlags::NONE)
            .unwrap()
            .id();
        let child = scene
            .add_node(
                Some(parent),
                None,
                Transform::from_position(Point3::new(0.0, 4.0, 0.0)),
                NodeFlags::NONE,
            )
            .unwrap()
            .id();

        let light = PositionedLight::node(
            child,
            Light::point(white(), 1.0),
            Transform::from_position(Point3::new(0.0, 0.0, 2.0)),
        );

        let resolved = resolve_light(&light, &scene, &test_camera()).unwrap();
        assert_vec3(resolved.position, [10.0, 4.0, 2.0]);
    }

    #[test]
    fn node_space_follows_the_node() {
        let mut scene = SceneData::new();
        let node = scene
            .add_node(None, None, Transform::IDENTITY, NodeFlags::NONE)
            .unwrap()
            .id();
        let light =
            PositionedLight::node(node, Light::point(white(), 1.0), Transform::IDENTITY);

        let resolved = resolve_light(&light, &scene, &test_camera()).unwrap();
        assert_vec3(resolved.position, [0.0, 0.0, 0.0]);

        scene.set_node_transform(node, Transform::from_position(Point3::new(0.0, 7.0, 0.0)));
        let resolved = resolve_light(&light, &scene, &test_camera()).unwrap();
        assert_vec3(resolved.position, [0.0, 7.0, 0.0]);
    }

    #[test]
    fn a_missing_anchor_drops_the_light() {
        let mut scene = SceneData::new();
        let node = scene
            .add_node(None, None, Transform::IDENTITY, NodeFlags::NONE)
            .unwrap()
            .id();
        let light =
            PositionedLight::node(node, Light::point(white(), 1.0), Transform::IDENTITY);
        assert!(resolve_light(&light, &scene, &test_camera()).is_some());

        scene.remove_node(node);
        assert!(
            resolve_light(&light, &scene, &test_camera()).is_none(),
            "a light anchored to a removed node must not fall back to the origin"
        );
    }

    #[test]
    fn resolve_lights_skips_only_the_unresolvable() {
        let mut scene = SceneData::new();
        let node = scene
            .add_node(None, None, Transform::IDENTITY, NodeFlags::NONE)
            .unwrap()
            .id();
        scene.remove_node(node);

        let lights = vec![
            PositionedLight::world(
                Light::point(white(), 1.0),
                Transform::from_position(Point3::new(1.0, 0.0, 0.0)),
            ),
            PositionedLight::node(node, Light::point(white(), 1.0), Transform::IDENTITY),
            PositionedLight::world(
                Light::point(white(), 1.0),
                Transform::from_position(Point3::new(2.0, 0.0, 0.0)),
            ),
        ];

        let resolved = resolve_lights(&lights, &scene, &test_camera());
        assert_eq!(resolved.len(), 2);
        assert_vec3(resolved[0].position, [1.0, 0.0, 0.0]);
        assert_vec3(resolved[1].position, [2.0, 0.0, 0.0]);
    }
}
