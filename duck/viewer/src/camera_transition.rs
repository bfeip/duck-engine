//! Time-driven camera transitions, advanced by the viewer each update.

use crate::scene::PositionedCamera;

const EPSILON: f32 = 1e-6;

/// An in-flight animated camera move for one view. The viewer advances it with
/// the frame delta time and drops it as soon as the view's camera is written
/// by anyone else.
pub(crate) struct CameraTransition {
    from: PositionedCamera,
    to: PositionedCamera,
    /// Seconds; always > 0.
    duration: f32,
    elapsed: f32,
    /// The pose this transition last wrote to the view, for detecting
    /// external writes.
    last_written: PositionedCamera,
}

impl CameraTransition {
    pub(crate) fn new(from: PositionedCamera, to: PositionedCamera, duration: f32) -> Self {
        Self { last_written: from.clone(), from, to, duration, elapsed: 0.0 }
    }

    /// Whether the view's camera no longer matches what this transition last
    /// wrote. Aspect is ignored: the viewer re-stamps it on resize.
    pub(crate) fn externally_modified(&self, current: &PositionedCamera) -> bool {
        let a = &self.last_written;
        (a.eye.x - current.eye.x).abs() > EPSILON
            || (a.eye.y - current.eye.y).abs() > EPSILON
            || (a.eye.z - current.eye.z).abs() > EPSILON
            || (a.target.x - current.target.x).abs() > EPSILON
            || (a.target.y - current.target.y).abs() > EPSILON
            || (a.target.z - current.target.z).abs() > EPSILON
            || (a.up.x - current.up.x).abs() > EPSILON
            || (a.up.y - current.up.y).abs() > EPSILON
            || (a.up.z - current.up.z).abs() > EPSILON
            || (a.fovy - current.fovy).abs() > EPSILON
            || a.ortho != current.ortho
    }

    /// Advance by `dt` seconds and return the pose to write to the view.
    /// Returns exactly the destination once the duration has elapsed.
    pub(crate) fn advance(&mut self, dt: f32) -> PositionedCamera {
        self.elapsed += dt;
        let pose = if self.finished() {
            self.to.clone()
        } else {
            let t = self.elapsed / self.duration;
            self.from.interpolated(&self.to, smoothstep(t))
        };
        self.last_written = pose.clone();
        pose
    }

    pub(crate) fn finished(&self) -> bool {
        self.elapsed >= self.duration
    }
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{Point3, Vector3};

    fn camera(eye: Point3) -> PositionedCamera {
        PositionedCamera {
            eye,
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::unit_y(),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        }
    }

    #[test]
    fn lands_exactly_on_destination() {
        let from = camera(Point3::new(5.0, 0.0, 0.0));
        let to = camera(Point3::new(0.0, 0.0, 5.0));
        let mut transition = CameraTransition::new(from, to.clone(), 0.3);

        transition.advance(0.1);
        assert!(!transition.finished());
        transition.advance(0.1);
        assert!(!transition.finished());
        let pose = transition.advance(0.15);
        assert!(transition.finished());
        assert_eq!(pose.eye, to.eye);
        assert_eq!(pose.up, to.up);
    }

    #[test]
    fn detects_external_writes() {
        let from = camera(Point3::new(5.0, 0.0, 0.0));
        let to = camera(Point3::new(0.0, 0.0, 5.0));
        let mut transition = CameraTransition::new(from, to, 0.3);

        let mut pose = transition.advance(0.1);
        assert!(!transition.externally_modified(&pose));

        // Aspect re-stamping must not read as an external write.
        pose.aspect = 2.0;
        assert!(!transition.externally_modified(&pose));

        pose.eye.x += 0.5;
        assert!(transition.externally_modified(&pose));
    }

    #[test]
    fn smoothstep_endpoints() {
        assert_eq!(smoothstep(0.0), 0.0);
        assert_eq!(smoothstep(1.0), 1.0);
        assert_eq!(smoothstep(-1.0), 0.0);
        assert_eq!(smoothstep(2.0), 1.0);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-6);
    }
}
