//! Per-node render-presentation behavior.
//!
//! [`DisplayBehavior`] groups the placement/presentation modifiers that affect
//! *how* a node's geometry is drawn — distinct from [`super::NodePayload`] (what
//! a node is) and [`super::NodeFlags`] (non-visual scene-system semantics like
//! selection, export, bounding).
//!
//! These modifiers inherit down the subtree: setting a layer or screen-space
//! flag on a group root applies to all descendants unless a descendant
//! overrides it. Inheritance is resolved by the renderer during its per-frame
//! traversal, so the scene crate stores only the per-node value and needs no
//! cache. Camera-dependent effects (`screen_sized`, `screen_facing`) are applied
//! at render time and never pollute the node's cached world transform.

use duck_engine_common::{
    EuclideanSpace, InnerSpace, Matrix4, Point3, Vector3,
};

use crate::PositionedCamera;

/// Which render layer / pass a node's geometry draws in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderLayer {
    /// Ordinary scene geometry, depth-tested against the rest of the scene.
    #[default]
    Scene,
    /// Drawn on top of the scene in a separate pass with its own depth buffer,
    /// so it depth-tests among itself but not against scene geometry. Used for
    /// gizmos, handles, and other always-visible annotation geometry.
    Overlay,
}

/// How a node's geometry is presented at render time.
///
/// Defaults to ordinary scene geometry, so a node with the default value is
/// rendered exactly as a node with no special behavior. Inherits down the
/// subtree (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DisplayBehavior {
    /// Keep a constant pixel size regardless of camera distance: `Some(px)`
    /// requests a constant on-screen size of `px` pixels for the geometry's
    /// unit extent (the renderer scales the geometry up as the camera recedes);
    /// `None` leaves the geometry at its authored world size. Applied at render
    /// time.
    pub screen_size: Option<f32>,
    /// Orient the geometry to face the camera (billboard): a `look_at` is
    /// applied at render time in addition to the node's own transform.
    pub screen_facing: bool,
    /// Which render layer / pass this node draws in.
    pub layer: RenderLayer,
}

impl DisplayBehavior {
    /// Whether this behavior needs a camera-dependent transform (screen-sizing
    /// and/or billboarding). Ordinary geometry returns `false`.
    pub fn is_screen_space(&self) -> bool {
        self.screen_size.is_some() || self.screen_facing
    }

    /// Resolves a node's own behavior against the inherited (parent) behavior: a
    /// non-`Scene` layer or a set screen-space flag overrides downward, otherwise
    /// the parent's value is carried.
    pub fn inherit(parent: DisplayBehavior, node: DisplayBehavior) -> DisplayBehavior {
        DisplayBehavior {
            // A node that opts into screen-sizing supplies its own size; otherwise
            // inherit the parent's.
            screen_size: node.screen_size.or(parent.screen_size),
            screen_facing: parent.screen_facing || node.screen_facing,
            layer: if node.layer != RenderLayer::Scene {
                node.layer
            } else {
                parent.layer
            },
        }
    }

    /// Camera-dependent model matrix for a node with this behavior.
    ///
    /// Billboarding (`screen_facing`) replaces the rotation basis with one that
    /// faces the camera; screen-sizing (`screen_size`) replaces the per-axis
    /// scale with a uniform constant-pixel-size scale. The node origin
    /// (translation) is always preserved. Ordinary (non-screen-space) behavior
    /// returns `world` unchanged.
    pub fn effective_transform(
        &self,
        world: Matrix4,
        camera: &PositionedCamera,
        viewport: (u32, u32),
    ) -> Matrix4 {
        // Ordinary geometry is drawn at its authored world transform.
        if !self.is_screen_space() {
            return world;
        }

        // Node origin in world space (translation column of the world transform).
        let p = Point3::from_vec(world.w.truncate());

        // Rotation basis (orthonormal columns: right, up, forward).
        let (right, up, fwd) = if self.screen_facing {
            // Billboard: face the camera, discarding the node's authored rotation.
            // Falls back to the camera forward if the node sits on the eye.
            let fwd = normalize_or(camera.eye - p, -camera.forward());
            let right = camera.up.cross(fwd).normalize();
            let up = fwd.cross(right);
            (right, up, fwd)
        } else {
            // Keep the authored orientation, orthonormalized so the basis carries
            // rotation only (scale is reintroduced below).
            (
                normalize_or(world.x.truncate(), Vector3::unit_x()),
                normalize_or(world.y.truncate(), Vector3::unit_y()),
                normalize_or(world.z.truncate(), Vector3::unit_z()),
            )
        };

        // Per-axis scale.
        let scale = match self.screen_size {
            // Constant pixel size: uniform scale on every axis. Parent scale is
            // intentionally discarded — that is the point of constant pixel size.
            Some(target_px) => {
                let s = screen_size_scale(p, target_px, camera, viewport);
                Vector3::new(s, s, s)
            }
            // Preserve the authored world scale (the world column lengths).
            None => Vector3::new(
                world.x.truncate().magnitude(),
                world.y.truncate().magnitude(),
                world.z.truncate().magnitude(),
            ),
        };

        Matrix4::from_cols(
            (right * scale.x).extend(0.0),
            (up * scale.y).extend(0.0),
            (fwd * scale.z).extend(0.0),
            p.to_vec().extend(1.0),
        )
    }
}

/// Normalizes `v`, returning `fallback` if `v` is (near) zero-length.
fn normalize_or(v: Vector3, fallback: Vector3) -> Vector3 {
    if v.magnitude2() > f32::EPSILON {
        v.normalize()
    } else {
        fallback
    }
}

/// Uniform world-space scale that makes a unit-extent geometry span `target_px`
/// pixels on screen, regardless of camera distance.
fn screen_size_scale(
    p: Point3,
    target_px: f32,
    camera: &PositionedCamera,
    viewport: (u32, u32),
) -> f32 {
    let depth = (p - camera.eye).dot(camera.forward()).max(camera.znear);
    let depth_size = camera.world_size_per_pixel(depth, viewport.1);
    depth_size * target_px
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        }
    }

    #[test]
    fn is_screen_space_gates_on_flags() {
        assert!(!DisplayBehavior::default().is_screen_space());
        assert!(DisplayBehavior { screen_size: Some(10.0), ..Default::default() }.is_screen_space());
        assert!(DisplayBehavior { screen_facing: true, ..Default::default() }.is_screen_space());
    }

    #[test]
    fn inherit_carries_screen_size_and_layer_down() {
        let parent = DisplayBehavior {
            screen_size: Some(90.0),
            layer: RenderLayer::Overlay,
            ..Default::default()
        };
        let child = DisplayBehavior::inherit(parent, DisplayBehavior::default());
        assert_eq!(child.screen_size, Some(90.0));
        assert_eq!(child.layer, RenderLayer::Overlay);
        assert!(!child.screen_facing);

        // A child's own screen_size overrides the parent's.
        let overriding = DisplayBehavior { screen_size: Some(20.0), ..Default::default() };
        assert_eq!(DisplayBehavior::inherit(parent, overriding).screen_size, Some(20.0));
    }

    #[test]
    fn screen_size_scale_grows_with_distance() {
        let display = DisplayBehavior { screen_size: Some(50.0), ..Default::default() };
        let cam = camera();

        let near = display.effective_transform(
            Matrix4::from_translation(Vector3::new(0.0, 0.0, 4.0)),
            &cam,
            (256, 256),
        );
        let far = display.effective_transform(
            Matrix4::from_translation(Vector3::new(0.0, 0.0, -20.0)),
            &cam,
            (256, 256),
        );

        // Uniform scale in the column magnitudes; farther node → larger scale to
        // keep the same on-screen size.
        let near_scale = near.x.truncate().magnitude();
        let far_scale = far.x.truncate().magnitude();
        assert!(far_scale > near_scale);

        // Translation (node origin) is preserved.
        assert!((near.w.truncate() - Vector3::new(0.0, 0.0, 4.0)).magnitude() < EPSILON);
    }

    #[test]
    fn effective_transform_passes_world_through_when_not_screen_space() {
        // Non-screen-space behavior returns the world matrix unchanged, exactly —
        // including any authored rotation and non-uniform scale.
        let display = DisplayBehavior::default();
        let world = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0))
            * Matrix4::from_nonuniform_scale(2.0, 0.5, 4.0);
        let m = display.effective_transform(world, &camera(), (256, 256));
        assert_eq!(m, world);
    }
}
