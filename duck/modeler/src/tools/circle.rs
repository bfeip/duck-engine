use anyhow::Result;
use duck_engine_common::{Plane, Point3, Real, Transform};
use opencascade::primitives::Shape;

use crate::ops::primitives::{circle, region};
use crate::snap::Snap;
use crate::tools::ToolInfo;
use crate::ui::icons;
use super::edit::{dimension_field, Params};
use super::primitive::{
    flat, radius_to, seat, unit_disk, Next, Pointer, Primitive, PrimitiveTool, Track,
};

/// The size of a placed circle, about its centre.
#[derive(Clone, Copy)]
pub struct CircleParams {
    center: Point3,
    plane: Plane,
    radius: Real,
}

/// Placement of a circle: the centre picked, and the pointer sizing the radius
/// about it.
#[derive(Clone, Copy, Debug)]
pub struct CircleStage {
    center: Point3,
    plane: Plane,
}

impl Params for CircleParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        dimension_field(ui, "Radius", &mut self.radius)
    }
}

/// Places a flat disk: its centre, then a point on its rim.
pub type CircleTool = PrimitiveTool<CircleParams>;

impl Primitive for CircleParams {
    type Stage = CircleStage;

    const TOOL: ToolInfo = ToolInfo { id: "circle", icon: icons::CIRCLE, shortcut: None };
    const NAME: &'static str = "Circle";

    fn start(first: &Snap, construction: &Plane) -> CircleStage {
        CircleStage { center: first.position, plane: seat(first, construction) }
    }

    fn track(stage: CircleStage, pointer: &Pointer) -> Track<Self> {
        let CircleStage { center, plane } = stage;
        let rim = pointer.snap.map(|snap| snap.position);
        let placed = rim
            .and_then(|rim| radius_to(center, rim))
            .map(|radius| CircleParams { center, plane, radius });
        Track {
            cursor: rim,
            preview: placed.map(|params| params.preview_transform()),
            click: placed.map(Next::Placed),
        }
    }

    fn reference(_stage: CircleStage) -> Result<Shape> {
        unit_disk()
    }

    fn preview_transform(&self) -> Transform {
        flat(self.center, &self.plane, self.radius, self.radius)
    }

    fn build(&self) -> Result<Shape> {
        Ok(region(circle(self.center, self.plane.normal, self.radius)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::primitive::pointer_at;

    const EPSILON: Real = 1e-6;

    #[test]
    fn a_point_on_the_rim_places_the_circle() {
        let center = Point3::new(1.0, 0.0, 2.0);
        let stage = CircleStage { center, plane: Plane::xz() };
        let track = CircleParams::track(stage, &pointer_at(Point3::new(4.0, 0.0, 6.0)));
        let Some(Next::Placed(params)) = track.click else { panic!("the rim places the circle") };
        assert!((params.radius - 5.0).abs() < EPSILON);
    }

    #[test]
    fn a_point_on_the_centre_cannot_be_clicked() {
        let center = Point3::new(1.0, 0.0, 2.0);
        let stage = CircleStage { center, plane: Plane::xz() };
        let track = CircleParams::track(stage, &pointer_at(center));
        assert!(track.click.is_none());
        assert!(track.preview.is_none());
    }
}
