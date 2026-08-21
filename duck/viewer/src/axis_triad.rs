//! Axis-triad overlay: a corner view mirroring a target view's camera
//! orientation, whose axis handles snap that camera to axis-aligned views.

use std::sync::{Arc, Mutex};

use crate::common::{Aabb, Axis, InnerSpace, Matrix4, Point3, RgbaColor, Vector3};
use crate::event::{DeviceEvent, Event, EventContext};
use crate::geom_query::{RayPickQuery, pick_all_from_ray};
use crate::input::MouseButton;
use crate::operator::Operator;
use crate::scene::resource::{
    FaceMaterial, FaceMaterialHandle, Instance, MaterialFlags, Mesh, NodeFlags, NodeId,
    PrimitiveType,
};
use crate::scene::{PositionedCamera, Scene, SceneData};
use crate::view::{Corner, ViewId};

/// How far the negative-end balls are darkened from their axis color, so they
/// read as the dimmer siblings of the positive arrows.
const NEGATIVE_END_DARKEN: f32 = 0.45;

/// Configuration for an axis-triad overlay; see
/// [`Viewer::add_axis_triad`](crate::Viewer::add_axis_triad).
#[derive(Clone, Debug)]
pub struct AxisTriadConfig {
    /// Corner of the viewer target the triad view is anchored to.
    pub corner: Corner,
    /// Side length of the square triad view in physical pixels.
    pub size: u32,
    /// Distance from the anchored corner in physical pixels (x, y).
    pub margin: (u32, u32),
    /// Distance of the triad camera from the triad origin.
    pub camera_distance: f32,
    /// Orthographic triad camera.
    pub ortho: bool,
    /// Seconds a snap takes to animate the target camera; non-positive snaps
    /// instantly.
    pub snap_duration: f32,
    /// Refit the target camera to its scene bounds as part of a snap.
    pub fit_on_snap: bool,
}

impl Default for AxisTriadConfig {
    fn default() -> Self {
        Self {
            corner: Corner::TopRight,
            size: 140,
            margin: (16, 16),
            camera_distance: 3.2,
            ortho: true,
            snap_duration: 0.35,
            fit_on_snap: true,
        }
    }
}

impl AxisTriadConfig {
    pub fn with_corner(mut self, corner: Corner) -> Self {
        self.corner = corner;
        self
    }

    pub fn with_size(mut self, size: u32) -> Self {
        self.size = size;
        self
    }

    pub fn with_margin(mut self, margin: (u32, u32)) -> Self {
        self.margin = margin;
        self
    }

    pub fn with_camera_distance(mut self, distance: f32) -> Self {
        self.camera_distance = distance;
        self
    }

    pub fn with_ortho(mut self, ortho: bool) -> Self {
        self.ortho = ortho;
        self
    }

    pub fn with_snap_duration(mut self, seconds: f32) -> Self {
        self.snap_duration = seconds;
        self
    }

    pub fn with_fit_on_snap(mut self, fit: bool) -> Self {
        self.fit_on_snap = fit;
        self
    }
}

/// An axis view requested by clicking a triad handle.
pub(crate) struct SnapRequest {
    /// Unit offset of the eye from the target.
    pub(crate) direction: Vector3,
    pub(crate) up: Vector3,
}

/// One triad registered with a viewer: the overlay view, the view it mirrors
/// and snaps, and the channel its operator reports clicks through.
pub(crate) struct AxisTriad {
    pub(crate) view: ViewId,
    pub(crate) target: ViewId,
    pub(crate) config: AxisTriadConfig,
    pub(crate) pending_snap: Arc<Mutex<Option<SnapRequest>>>,
}

/// One clickable handle of the triad; maps its scene nodes to a camera pose.
struct TriadHandle {
    nodes: Vec<NodeId>,
    /// Unit offset of the eye from the target.
    direction: Vector3,
    up: Vector3,
}

/// Up vector for a view looking along `direction`.
fn view_up(direction: Vector3) -> Vector3 {
    if direction.y.abs() > 0.5 { Vector3::unit_z() } else { Vector3::unit_y() }
}

/// Build the triad's scene: an unlit arrow per axis, a ball at each negative
/// axis end, and a hub. Returns the clickable handles.
fn build_triad_scene(scene: &Scene) -> Vec<TriadHandle> {
    let mut guard = scene.lock();

    fn add_part(
        scene: &mut SceneData,
        mesh: Mesh,
        material: &FaceMaterialHandle,
        name: String,
    ) -> NodeId {
        let mesh = scene.add_mesh(mesh);
        scene
            .add_instance_node(
                None,
                Instance::new(mesh).with_face_material(material.clone()),
                Some(name),
                crate::common::Transform::IDENTITY,
                NodeFlags::NONE,
            )
            .unwrap()
            .id()
    }

    // Unlit so the triad reads the same from every angle.
    let unlit = |color: RgbaColor| {
        FaceMaterial::new().with_base_color_factor(color).with_flags(MaterialFlags::DO_NOT_LIGHT)
    };
    let mut handles = Vec::new();
    for axis in Axis::ALL {
        let dir = axis.direction();
        let name = format!("{axis:?}");

        // Arrow along the positive axis: shaft built along +Y, rotated onto
        // the axis, plus a cone head at the tip.
        let material = guard.add_face_material(unlit(axis.color()));
        let shaft = Mesh::cylinder(0.05, 0.7, 16, true, PrimitiveType::TriangleList).transformed(
            &(axis.rotation_from_y() * Matrix4::from_translation(Vector3::unit_y() * 0.35)),
        );
        let head = Mesh::cone_directed(
            Point3::new(0.0, 0.0, 0.0) + dir,
            -dir,
            0.12,
            0.3,
            16,
            true,
            PrimitiveType::TriangleList,
        );
        handles.push(TriadHandle {
            nodes: vec![
                add_part(&mut guard, shaft, &material, format!("{name} shaft")),
                add_part(&mut guard, head, &material, format!("{name} head")),
            ],
            direction: dir,
            up: view_up(dir),
        });

        // Ball at the negative end for the opposite view.
        let material = guard.add_face_material(unlit(axis.color().darkened(NEGATIVE_END_DARKEN)));
        let ball = Mesh::sphere(0.14, 16, 8, PrimitiveType::TriangleList)
            .transformed(&Matrix4::from_translation(-dir * 0.8));
        handles.push(TriadHandle {
            nodes: vec![add_part(&mut guard, ball, &material, format!("-{name} ball"))],
            direction: -dir,
            up: view_up(-dir),
        });
    }

    let hub_material = guard.add_face_material(unlit(RgbaColor {
        r: 0.7,
        g: 0.7,
        b: 0.7,
        a: 1.0,
    }));
    add_part(
        &mut guard,
        Mesh::sphere(0.12, 16, 8, PrimitiveType::TriangleList),
        &hub_material,
        "Hub".to_string(),
    );

    handles
}

/// Operator on the triad view: clicking a handle records a snap request for
/// the viewer to apply to the target view's camera.
pub(crate) struct AxisTriadOperator {
    handles: Vec<TriadHandle>,
    pending: Arc<Mutex<Option<SnapRequest>>>,
}

impl Operator for AxisTriadOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(DeviceEvent::MouseClick { button: MouseButton::Left, position, .. }) =
            event
        else {
            return false;
        };

        let ray =
            ctx.camera.ray_from_screen_point(position.0, position.1, ctx.size.0, ctx.size.1);
        let query = RayPickQuery::for_kinds(ray, 0.0, true, false, false);
        let results = pick_all_from_ray(&query, &ctx.scene);
        let Some(hit) = results.first() else { return false };
        let Some(handle) = self.handles.iter().find(|h| h.nodes.contains(&hit.node_id)) else {
            return false;
        };

        *self.pending.lock().unwrap() = Some(SnapRequest {
            direction: handle.direction,
            up: handle.up,
        });
        true
    }

    fn name(&self) -> &str {
        "AxisTriad"
    }
}

/// Build the triad scene and its operator, sharing `pending` with the viewer.
pub(crate) fn build_triad(
    scene: &Scene,
    pending: Arc<Mutex<Option<SnapRequest>>>,
) -> Arc<Mutex<AxisTriadOperator>> {
    let handles = build_triad_scene(scene);
    Arc::new(Mutex::new(AxisTriadOperator { handles, pending }))
}

/// The camera pose a snap request moves `current` to: looking along the
/// requested axis from the preserved eye distance. Requesting the view the
/// camera is already in flips to the opposite side.
pub(crate) fn snap_camera(
    current: &PositionedCamera,
    request: &SnapRequest,
    bounds: Option<&Aabb>,
) -> PositionedCamera {
    let mut camera = current.clone();
    let mut direction = request.direction;
    if current.forward().dot(-direction) > 1.0 - 1e-4 {
        direction = -direction;
    }
    camera.eye = camera.target + direction * current.length();
    camera.up = request.up;
    if let Some(bounds) = bounds {
        camera.fit_to_bounds(bounds);
    }
    camera
}

/// Aim `camera` at the triad origin from `distance`, matching the orientation
/// of `reference`.
pub(crate) fn aim_triad_camera(
    camera: &mut PositionedCamera,
    reference: &PositionedCamera,
    distance: f32,
) {
    let direction = (reference.eye - reference.target).normalize();
    camera.target = Point3::new(0.0, 0.0, 0.0);
    camera.eye = camera.target + direction * distance;
    camera.up = reference.up;
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    fn camera(eye: Point3, up: Vector3) -> PositionedCamera {
        PositionedCamera {
            eye,
            target: Point3::new(0.0, 0.0, 0.0),
            up,
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        }
    }

    #[test]
    fn triad_scene_covers_all_six_directions() {
        let scene = Scene::default();
        let handles = build_triad_scene(&scene);
        assert_eq!(handles.len(), 6);

        for axis in Axis::ALL {
            for sign in [1.0, -1.0] {
                let dir = axis.direction() * sign;
                let handle = handles
                    .iter()
                    .find(|h| (h.direction - dir).magnitude() < EPSILON)
                    .expect("handle for direction");
                assert!(!handle.nodes.is_empty());
                assert!((handle.up.dot(handle.direction)).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn snap_preserves_distance_and_sets_up() {
        let current = camera(Point3::new(3.0, 4.0, 0.0), Vector3::unit_y());
        let request = SnapRequest { direction: Vector3::unit_z(), up: Vector3::unit_y() };

        let snapped = snap_camera(&current, &request, None);
        assert!((snapped.length() - 5.0).abs() < EPSILON);
        assert!((snapped.eye - Point3::new(0.0, 0.0, 5.0)).magnitude() < EPSILON);
        assert!((snapped.up - Vector3::unit_y()).magnitude() < EPSILON);
    }

    #[test]
    fn snap_uses_z_up_for_vertical_views() {
        let current = camera(Point3::new(5.0, 0.0, 0.0), Vector3::unit_y());
        let request = SnapRequest { direction: Vector3::unit_y(), up: view_up(Vector3::unit_y()) };

        let snapped = snap_camera(&current, &request, None);
        assert!((snapped.eye - Point3::new(0.0, 5.0, 0.0)).magnitude() < EPSILON);
        assert!((snapped.up - Vector3::unit_z()).magnitude() < EPSILON);
    }

    #[test]
    fn snap_to_current_view_flips_to_opposite() {
        let current = camera(Point3::new(0.0, 0.0, 5.0), Vector3::unit_y());
        let request = SnapRequest { direction: Vector3::unit_z(), up: Vector3::unit_y() };

        let snapped = snap_camera(&current, &request, None);
        assert!((snapped.eye - Point3::new(0.0, 0.0, -5.0)).magnitude() < EPSILON);
    }

    #[test]
    fn aim_matches_reference_orientation() {
        let reference = camera(Point3::new(10.0, 10.0, 10.0), Vector3::unit_y());
        let mut triad_camera = camera(Point3::new(0.0, 0.0, 3.2), Vector3::unit_y());

        aim_triad_camera(&mut triad_camera, &reference, 3.2);
        assert!((triad_camera.length() - 3.2).abs() < EPSILON);
        let expected = Vector3::new(1.0, 1.0, 1.0).normalize() * 3.2;
        assert!((triad_camera.eye - Point3::new(expected.x, expected.y, expected.z)).magnitude() < EPSILON);
    }
}
