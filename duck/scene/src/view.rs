use crate::PositionedCamera;
use duck_engine_common::InnerSpace;

/// Unique identifier for a named view in the scene.
pub type ViewId = crate::resource::Id<View>;

/// A named camera state that can be saved and restored.
///
/// Views are typically imported from CAD files (STEP/IGES) where the author
/// has defined specific vantage points (e.g. "Front", "Isometric", "Section A").
/// They can also be created programmatically.
///
/// A view is a complete [`PositionedCamera`] snapshot. To apply a view, clone its
/// camera and pass it to the renderer:
///
/// ```rust,ignore
/// if let Some(view) = scene.get_view(view_id) {
///     camera = view.camera().clone();
/// }
/// ```
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
    pub fn new(id: ViewId, name: impl Into<String>, camera: PositionedCamera) -> Self {
        Self { id, name: name.into(), camera }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn camera(&self) -> &PositionedCamera {
        &self.camera
    }

    /// Produce a camera for rendering by blending this view's orientation into `reference`.
    ///
    /// Copies `eye`, `target` (preserving `reference.length()` as the orbit distance),
    /// `up`, and `ortho` from the stored camera. All other fields — `znear`, `zfar`,
    /// `fovy`, `aspect` — are taken from `reference`, which is expected to be already
    /// calibrated for the scene (e.g. via `PositionedCamera::fit_to_bounds`).
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
