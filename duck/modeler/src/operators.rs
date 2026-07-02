mod boolean;
mod r#box;
mod circle;
mod curve;
mod cylinder;
mod extrude;
mod line;
mod loft;
mod rectangle;
mod sphere;
mod transform;

pub use boolean::BooleanOperator;
pub use r#box::BoxOperator;
pub use circle::CircleOperator;
pub use curve::CurveOperator;
pub use cylinder::CylinderOperator;
pub use extrude::ExtrudeOperator;
pub use line::LineOperator;
pub use loft::LoftOperator;
pub use rectangle::RectangleOperator;
pub use sphere::SphereOperator;
pub use transform::TransformTool;

use duck_engine_common::{Plane, RgbaColor};
use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::{FaceMaterial, LineMaterial, MaterialFlags, NodeId};
use duck_engine_viewer::event::EventContext;
use duck_engine_viewer::scene::PositionedCamera;

use crate::grid::GridConfig;
use crate::snap::{Snap, SnapEngine, SnapFlags, SnapInput, SnapProvider};

pub struct ConstructionOptions {
    /// Canonical (fine) tessellation options for committed geometry. Previews
    /// reuse these via [`preview_options`](ConstructionOptions::preview_options)
    /// with a coarser tolerance.
    pub geometry_options: CadTessellationOptions,
    /// Coarser deflection used for transient previews, which are re-tessellated
    /// on every cursor move. Larger than `geometry_options.tessellation_tolerance`
    /// to keep dragging cheap on complex shapes.
    pub preview_tolerance: f64,
    pub construction_plane: Plane,
    pub grid: GridConfig,
    /// Shared snap engine (providers + user settings) consulted by every operator.
    pub snap: SnapEngine,
}

impl ConstructionOptions {
    pub fn new() -> Self {
        let geometry_options = CadTessellationOptions {
            tessellation_tolerance: 0.1,
            scale_factor: 1.0,
            face_material: FaceMaterial::new()
                .with_base_color_factor(RgbaColor { r: 0.55, g: 0.65, b: 0.9, a: 1.0 })
                // Double sided for now since regions are created with arbitrary orientation
                .with_flags(MaterialFlags::DOUBLE_SIDED),
            line_material: LineMaterial::new(RgbaColor { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
            include_edges: true,
            include_points: true,
        };
        let construction_plane = Plane::xz();
        let grid = GridConfig::default();
        let snap = SnapEngine::with_defaults();
        Self {
            geometry_options,
            preview_tolerance: 1.0,
            construction_plane,
            grid,
            snap
        }
    }

    /// Coarser clone of the canonical options for transient previews.
    pub fn preview_options(&self) -> CadTessellationOptions {
        let mut o = self.geometry_options.clone();
        o.tessellation_tolerance = self.preview_tolerance;
        o
    }

    /// Resolves a cursor position to a snapped world location via the shared snap
    /// engine, using this context's construction plane, grid, and snap settings.
    /// `exclude` lists nodes the snap should ignore — e.g. an operator's own
    /// in-progress preview geometry. The caller supplies `camera` so we never
    /// rebuild it while holding a scene lock. `additional_providers` lets the
    /// caller inject per-call snaps (e.g. an in-progress wire's start point)
    /// that compete in the normal ranking alongside the registered providers.
    pub fn resolve_snap(
        &self,
        cursor: (f32, f32),
        exclude: &[NodeId],
        camera: &PositionedCamera,
        ctx: &EventContext,
        additional_providers: &[&dyn SnapProvider],
    ) -> Option<Snap> {
        let input = SnapInput {
            ray: camera.ray_from_screen_point(cursor.0, cursor.1, ctx.size.0, ctx.size.1),
            cursor,
            viewport: ctx.size,
            camera,
            plane: &self.construction_plane,
            grid: &self.grid,
            requested: SnapFlags::all(),
            exclude_nodes: exclude,
        };
        let scene = ctx.scene.lock().unwrap();
        self.snap.snap(&input, &scene, additional_providers)
    }
}