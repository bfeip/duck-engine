use anyhow::Result;
use duck_engine_common::{Plane, Point3, Real, Transform};
use opencascade::primitives::Shape;

use crate::ops::primitives::{closed_polyline, rectangle_corners, region};
use crate::snap::Snap;
use crate::tools::ToolInfo;
use crate::ui::icons;
use super::edit::{dimension_field, Params};
use super::primitive::{
    flat, footprint, seat, unit_square, Next, Pointer, Primitive, PrimitiveTool, Track,
};

/// The size of a placed rectangle, about its centre.
#[derive(Clone, Copy)]
pub struct RectangleParams {
    center: Point3,
    plane: Plane,
    width: Real,
    depth: Real,
}

/// Placement of a rectangle: the centre picked, and the pointer sizing it
/// about that.
#[derive(Clone, Copy, Debug)]
pub struct RectangleStage {
    center: Point3,
    plane: Plane,
}

impl Params for RectangleParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = dimension_field(ui, "Width", &mut self.width);
        changed |= dimension_field(ui, "Length", &mut self.depth);
        changed
    }
}

/// Places a flat rectangle: its centre, then a corner.
pub type RectangleTool = PrimitiveTool<RectangleParams>;

impl Primitive for RectangleParams {
    type Stage = RectangleStage;

    const TOOL: ToolInfo = ToolInfo { id: "rectangle", icon: icons::RECTANGLE, shortcut: None };
    const NAME: &'static str = "Rectangle";

    fn start(first: &Snap, construction: &Plane) -> RectangleStage {
        RectangleStage { center: first.position, plane: seat(first, construction) }
    }

    fn track(stage: RectangleStage, pointer: &Pointer) -> Track<Self> {
        let RectangleStage { center, plane } = stage;
        let corner = pointer.snap.map(|snap| snap.position);
        let placed = corner
            .and_then(|corner| footprint(center, corner, &plane))
            .map(|(width, depth)| RectangleParams { center, plane, width, depth });
        Track {
            cursor: corner,
            preview: placed.map(|params| params.preview_transform()),
            click: placed.map(Next::Placed),
        }
    }

    fn reference(_stage: RectangleStage) -> Result<Shape> {
        unit_square()
    }

    fn preview_transform(&self) -> Transform {
        flat(self.center, &self.plane, self.width, self.depth)
    }

    /// World-space face with an analytic planar surface.
    fn build(&self) -> Result<Shape> {
        let corners = rectangle_corners(self.center, &self.plane, self.width, self.depth);
        Ok(region(closed_polyline(&corners)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::primitive::pointer_at;

    const EPSILON: Real = 1e-6;

    #[test]
    fn a_corner_places_the_rectangle_about_its_centre() {
        let stage = RectangleStage { center: Point3::new(0.0, 0.0, 0.0), plane: Plane::xz() };
        let track = RectangleParams::track(stage, &pointer_at(Point3::new(1.5, 0.0, 2.0)));
        let Some(Next::Placed(params)) = track.click else {
            panic!("the corner places the rectangle");
        };
        assert!((params.width - 3.0).abs() < EPSILON);
        assert!((params.depth - 4.0).abs() < EPSILON);
    }

    #[test]
    fn a_corner_on_an_axis_cannot_be_clicked() {
        let stage = RectangleStage { center: Point3::new(0.0, 0.0, 0.0), plane: Plane::xz() };
        let track = RectangleParams::track(stage, &pointer_at(Point3::new(1.0, 0.0, 0.0)));
        assert!(track.click.is_none());
        assert!(track.preview.is_none());
    }
}
