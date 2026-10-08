//! The settings every construction tool builds with: tessellation, the
//! construction plane and grid, and snapping.

use duck_engine_common::{
    EuclideanSpace, InnerSpace, Plane, Point3, Real, RgbaColor, Vector3, EPSILON,
};
use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::resource::{FaceMaterial, LineMaterial, MaterialFlags};

use crate::grid::GridConfig;
use crate::snap::SnapEngine;

/// How far a view may turn off the axis it snapped to, as `1 - cos θ` (about
/// 0.8°), and still count as that view.
const VIEW_TOLERANCE: Real = 1e-4;

/// A construction plane a view snap put in place, and the plane it replaced.
#[derive(Clone, Copy)]
struct ViewPlane {
    /// Unit offset of the eye from its target in the snapped view.
    toward_eye: Vector3,
    placed: Plane,
    replaced: Plane,
}

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
    /// Whether snapping the view to an axis turns the construction plane to
    /// face it until the view is left; see
    /// [`face_view`](ConstructionOptions::face_view).
    pub follow_view: bool,
    /// Set while the construction plane faces a snapped view.
    view_plane: Option<ViewPlane>,
    pub grid: GridConfig,
    /// Shared snap engine (providers + user settings) consulted by every tool.
    pub snap: SnapEngine,
}

impl ConstructionOptions {
    pub fn new() -> Self {
        let geometry_options = CadTessellationOptions {
            tessellation_tolerance: 0.1,
            scale_factor: 1.0,
            face_material: FaceMaterial::new()
                .with_base_color_factor(RgbaColor { r: 0.34, g: 0.40, b: 0.52, a: 1.0 })
                .with_roughness_factor(0.32)
                // Double sided for now since regions are created with arbitrary orientation
                .with_flags(MaterialFlags::DOUBLE_SIDED),
            line_material: LineMaterial::new(RgbaColor { r: 0.02, g: 0.025, b: 0.035, a: 1.0 }),
            // Sketch appearance for geometry that bounds no volume: free edges,
            // wires, and lone faces.
            free_face_material: Some(
                FaceMaterial::new()
                    .with_base_color_factor(RgbaColor { r: 0.42, g: 0.68, b: 0.92, a: 0.3 })
                    .with_flags(MaterialFlags::DOUBLE_SIDED | MaterialFlags::DO_NOT_LIGHT),
            ),
            free_line_material: Some(LineMaterial::new(RgbaColor {
                r: 0.16,
                g: 0.40,
                b: 0.78,
                a: 1.0,
            })),
            include_edges: true,
            include_points: true,
            show_seam_edges: false,
        };
        let construction_plane = Plane::xz();
        let grid = GridConfig::default();
        let snap = SnapEngine::with_defaults();
        Self {
            geometry_options,
            preview_tolerance: 1.0,
            construction_plane,
            follow_view: true,
            view_plane: None,
            grid,
            snap
        }
    }

    /// Turns the construction plane to face a view snapped to look from
    /// `toward_eye`, the unit offset of the eye from its target, until
    /// [`leave_view`](Self::leave_view) sees the view left.
    /// 
    /// Does nothing unless [`follow_view`](Self::follow_view) is set. The plane passes
    /// through the origin, unless it only flips to face the other way, which
    /// keeps it in place. Returns whether the plane changed.
    pub fn face_view(&mut self, toward_eye: Vector3) -> bool {
        if !self.follow_view {
            return false;
        }
        let current = self.construction_plane;
        let anchor = if current.normal.dot(toward_eye).abs() > 1.0 - EPSILON {
            current.project_point(Point3::origin())
        } else {
            Point3::origin()
        };
        let plane = Plane::from_point(toward_eye, anchor);
        // Snap after snap still leaves for the plane from before the first,
        // unless the user has since set their own.
        let replaced = self
            .view_plane
            .filter(|view_plane| same_plane(&view_plane.placed, &current))
            .map_or(current, |view_plane| view_plane.replaced);
        self.view_plane = Some(ViewPlane { toward_eye, placed: plane, replaced });
        self.set_plane(plane)
    }

    /// Puts back the plane a view snap replaced once the view looks from
    /// `toward_eye` instead of the snapped direction. A plane set since the
    /// snap is the user's own, and stays. Returns whether the plane changed.
    pub fn leave_view(&mut self, toward_eye: Vector3) -> bool {
        let Some(view_plane) = self.view_plane else { return false };
        if view_plane.toward_eye.dot(toward_eye) > 1.0 - VIEW_TOLERANCE {
            return false;
        }
        self.view_plane = None;
        if !same_plane(&view_plane.placed, &self.construction_plane) {
            return false;
        }
        self.set_plane(view_plane.replaced)
    }

    /// Sets the construction plane, returning whether it changed.
    fn set_plane(&mut self, plane: Plane) -> bool {
        let changed = !same_plane(&plane, &self.construction_plane);
        self.construction_plane = plane;
        changed
    }

    /// Coarser clone of the canonical options for transient previews.
    pub fn preview_options(&self) -> CadTessellationOptions {
        let mut o = self.geometry_options.clone();
        o.tessellation_tolerance = self.preview_tolerance;
        o
    }
}

fn same_plane(a: &Plane, b: &Plane) -> bool {
    (a.normal - b.normal).magnitude() < EPSILON && (a.d - b.d).abs() < EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether `plane` has `normal` and passes through `point`.
    fn is_plane(plane: &Plane, normal: Vector3, point: Point3) -> bool {
        (plane.normal - normal).magnitude() < EPSILON && plane.signed_distance(point).abs() < EPSILON
    }

    #[test]
    fn a_side_view_turns_the_plane_toward_the_eye_through_the_origin() {
        let mut options = ConstructionOptions::new();
        options.construction_plane = Plane::from_point(Vector3::unit_y(), Point3::new(0.0, 20.0, 0.0));

        assert!(options.face_view(Vector3::unit_x()));
        assert!(is_plane(&options.construction_plane, Vector3::unit_x(), Point3::origin()));

        assert!(options.face_view(-Vector3::unit_z()));
        assert!(is_plane(&options.construction_plane, -Vector3::unit_z(), Point3::origin()));
    }

    #[test]
    fn flipping_to_the_opposite_view_keeps_the_plane_in_place() {
        let mut options = ConstructionOptions::new();
        options.construction_plane = Plane::from_point(Vector3::unit_z(), Point3::new(0.0, 0.0, 50.0));

        assert!(options.face_view(-Vector3::unit_z()));
        assert!(is_plane(&options.construction_plane, -Vector3::unit_z(), Point3::new(0.0, 0.0, 50.0)));
    }

    #[test]
    fn the_view_already_faced_changes_nothing() {
        let mut options = ConstructionOptions::new();
        assert!(!options.face_view(Vector3::unit_y()));
        assert!(is_plane(&options.construction_plane, Vector3::unit_y(), Point3::origin()));
    }

    #[test]
    fn a_plane_not_following_the_view_stays() {
        let mut options = ConstructionOptions::new();
        options.follow_view = false;
        assert!(!options.face_view(Vector3::unit_x()));
        assert!(!options.leave_view(Vector3::unit_z()));
        assert!(is_plane(&options.construction_plane, Vector3::unit_y(), Point3::origin()));
    }

    #[test]
    fn leaving_the_view_puts_the_plane_back() {
        let mut options = ConstructionOptions::new();
        options.face_view(Vector3::unit_x());

        // Still the snapped view: a pan or zoom keeps its direction.
        assert!(!options.leave_view(Vector3::unit_x()));
        assert!(is_plane(&options.construction_plane, Vector3::unit_x(), Point3::origin()));

        assert!(options.leave_view(Vector3::new(1.0, 0.3, 0.0).normalize()));
        assert!(is_plane(&options.construction_plane, Vector3::unit_y(), Point3::origin()));

        // Left once; orbiting further changes nothing.
        assert!(!options.leave_view(Vector3::unit_z()));
    }

    #[test]
    fn snap_after_snap_puts_back_the_plane_from_before_the_first() {
        let mut options = ConstructionOptions::new();
        let raised = Point3::new(0.0, 0.0, 50.0);
        options.construction_plane = Plane::from_point(Vector3::unit_z(), raised);

        options.face_view(Vector3::unit_x());
        options.face_view(-Vector3::unit_y());
        assert!(is_plane(&options.construction_plane, -Vector3::unit_y(), Point3::origin()));

        assert!(options.leave_view(Vector3::new(0.0, -1.0, 1.0).normalize()));
        assert!(is_plane(&options.construction_plane, Vector3::unit_z(), raised));
    }

    #[test]
    fn a_plane_set_in_the_snapped_view_stays_on_leaving() {
        let mut options = ConstructionOptions::new();
        options.face_view(Vector3::unit_x());
        let own = Plane::from_point(Vector3::unit_x(), Point3::new(10.0, 0.0, 0.0));
        options.construction_plane = own;

        assert!(!options.leave_view(Vector3::unit_z()));
        assert!(is_plane(&options.construction_plane, Vector3::unit_x(), Point3::new(10.0, 0.0, 0.0)));

        // And a snap from it leaves for it.
        options.face_view(Vector3::unit_z());
        assert!(options.leave_view(Vector3::unit_y()));
        assert!(is_plane(&options.construction_plane, Vector3::unit_x(), Point3::new(10.0, 0.0, 0.0)));
    }
}