use crate::PositionedCamera;
use duck_engine_common::InnerSpace;

/// Unique identifier for a named view in the scene.
pub type ViewId = crate::resource::Id<View>;

/// A named camera state that can be saved and restored.
///
/// A view is a complete [`PositionedCamera`] snapshot under a human-readable
/// name — a vantage point like "Front" or "Isometric". Restore one by using
/// its [`camera`](Self::camera) directly, or via
/// [`apply_to`](Self::apply_to) to keep an existing camera's calibration.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct View {
    /// Unique identifier for this view.
    pub id: ViewId,
    /// Human-readable name (e.g. "Front", "Isometric").
    pub name: String,
    /// The saved camera configuration.
    pub camera: PositionedCamera,
}

impl View {
    /// Creates a view from an id, a name, and a camera snapshot.
    pub fn new(id: ViewId, name: impl Into<String>, camera: PositionedCamera) -> Self {
        Self { id, name: name.into(), camera }
    }

    /// The view's human-readable name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The saved camera snapshot.
    pub fn camera(&self) -> &PositionedCamera {
        &self.camera
    }

    /// Produce a camera for rendering by blending this view's pose into `reference`.
    ///
    /// Takes `eye`, the view direction, `up`, and `ortho` from the stored
    /// camera; the target is placed along that direction at `reference`'s orbit
    /// distance. All other fields — `znear`, `zfar`, `fovy`, `aspect` — come
    /// from `reference`, which is expected to be already calibrated for the
    /// scene (e.g. via [`PositionedCamera::fit_to_bounds`]).
    pub fn apply_to(&self, reference: &PositionedCamera) -> PositionedCamera {
        let orbit_dist = reference.length().max(0.001);
        let dir = (self.camera.target - self.camera.eye).normalize();
        let mut camera = reference.clone();
        camera.eye = self.camera.eye;
        camera.target = self.camera.eye + dir * orbit_dist;
        camera.up = self.camera.up;
        camera.ortho = self.camera.ortho;
        camera
    }
}
