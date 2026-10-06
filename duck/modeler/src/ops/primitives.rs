//! Shapes built from picked points: primitive solids, and the outlines and
//! regions sketched with lines, curves and circles.

use anyhow::{ensure, Context, Result};
use duck_engine_scene::common::{Plane, Point3, Real, Vector3};
use opencascade::primitives::{Edge, Face, Shape, Wire};

use crate::document::{point3_to_dvec3, vec3_to_dvec3};

/// The corners of a `width` × `depth` rectangle centred on `centre` in
/// `plane`, in outline order.
pub fn rectangle_corners(centre: Point3, plane: &Plane, width: Real, depth: Real) -> [Point3; 4] {
    let (u, v) = plane.basis();
    let half_width = u * (0.5 * width);
    let half_depth = v * (0.5 * depth);
    [
        centre - half_width - half_depth,
        centre + half_width - half_depth,
        centre + half_width + half_depth,
        centre - half_width + half_depth,
    ]
}

/// The solid swept by the flat polygon through `outline` along `sweep`.
pub fn prism(outline: &[Point3], sweep: Vector3) -> Result<Shape> {
    let face = Face::from_wire(&closed_polyline(outline)?).context("The outline doesn't bound a flat face")?;
    Ok(face.extrude(vec3_to_dvec3(sweep)).into())
}

/// A cylinder standing on `base` along `axis`.
pub fn cylinder(base: Point3, axis: Vector3, radius: Real, height: Real) -> Shape {
    Shape::cylinder(point3_to_dvec3(base), f64::from(radius), vec3_to_dvec3(axis), f64::from(height))
}

/// A sphere about `centre`, with its poles on `axis`.
pub fn sphere(centre: Point3, axis: Vector3, radius: Real) -> Shape {
    Shape::sphere(f64::from(radius)).at(point3_to_dvec3(centre)).axis(vec3_to_dvec3(axis)).build()
}

/// The circle of `radius` about `centre`, in the plane facing `normal`.
pub fn circle(centre: Point3, normal: Vector3, radius: Real) -> Result<Wire> {
    let edge = Edge::circle(point3_to_dvec3(centre), vec3_to_dvec3(normal), f64::from(radius))
        .context("Failed to build the circle")?;
    Wire::from_edges(&[edge]).context("Failed to build the circle")
}

/// The open polyline through `points`, one segment per consecutive pair.
pub fn polyline(points: &[Point3]) -> Result<Wire> {
    ensure!(points.len() >= 2, "A line needs at least two points");
    let edges = points
        .windows(2)
        .map(|pair| Edge::segment(point3_to_dvec3(pair[0]), point3_to_dvec3(pair[1])))
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to build a line segment")?;
    Wire::from_edges(&edges).context("Failed to build the line")
}

/// The closed polygon through `points`, back to the first.
pub fn closed_polyline(points: &[Point3]) -> Result<Wire> {
    ensure!(points.len() >= 3, "A closed outline needs at least three points");
    Wire::from_ordered_points(points.iter().copied().map(point3_to_dvec3)).context("Failed to build the outline")
}

/// The open curve interpolating `points`.
pub fn spline(points: &[Point3]) -> Result<Wire> {
    ensure!(points.len() >= 2, "A curve needs at least two points");
    let edge = Edge::spline_from_points(points.iter().copied().map(point3_to_dvec3), None, false)
        .context("Failed to build the curve")?;
    Wire::from_edges(&[edge]).context("Failed to build the curve")
}

/// The closed, smooth curve interpolating `points`, back through the first.
pub fn closed_spline(points: &[Point3]) -> Result<Wire> {
    ensure!(points.len() >= 3, "A closed curve needs at least three points");
    let edge = Edge::spline_from_points(points.iter().copied().map(point3_to_dvec3), None, true)
        .context("Failed to build the closed curve")?;
    Wire::from_edges(&[edge]).context("Failed to build the closed curve")
}

/// What `boundary` encloses: the face it bounds, or the bare loop when it
/// isn't flat.
pub fn region(boundary: Wire) -> Shape {
    // `to_face` consumes the wire, so the loop is kept first.
    let outline = Shape::from(&boundary);
    match boundary.to_face() {
        Ok(face) => Shape::from(&face),
        Err(_) => outline,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::common::InnerSpace;
    use opencascade::primitives::ShapeType;

    fn p(x: Real, y: Real, z: Real) -> Point3 {
        Point3::new(x, y, z)
    }

    /// A plane aligned with no world axis, so a mistaken basis shows up.
    fn skewed_plane(origin: Point3) -> Plane {
        Plane::from_point(Vector3::new(1.0, 2.0, 3.0).normalize(), origin)
    }

    #[test]
    fn rectangle_corners_are_centred_and_span_the_extents() {
        let centre = p(-1.0, 0.5, 2.0);
        let plane = skewed_plane(centre);
        let corners = rectangle_corners(centre, &plane, 4.0, 6.0);

        let mean = corners.iter().fold(Vector3::new(0.0, 0.0, 0.0), |sum, c| sum + (c - centre)) / 4.0;
        assert!(mean.magnitude() < 1e-5, "the corners are centred on the centre");
        assert!(((corners[1] - corners[0]).magnitude() - 4.0).abs() < 1e-5);
        assert!(((corners[2] - corners[1]).magnitude() - 6.0).abs() < 1e-5);
    }

    #[test]
    fn a_prism_sweeps_its_outline() {
        let outline = rectangle_corners(p(0.0, 0.0, 0.0), &Plane::xz(), 2.0, 3.0);
        let solid = prism(&outline, Vector3::new(0.0, 4.0, 0.0)).expect("the prism builds");
        assert_eq!(solid.shape_type(), ShapeType::Solid);
        assert!((solid.volume() - 24.0).abs() < 1e-6, "got {}", solid.volume());
    }

    #[test]
    fn a_circle_bounds_a_disk() {
        let disk = region(circle(p(0.0, 0.0, 0.0), Vector3::unit_y(), 1.0).expect("the circle builds"));
        assert_eq!(disk.shape_type(), ShapeType::Face);
    }

    #[test]
    fn a_circle_builds_off_the_origin() {
        assert!(circle(p(3.0, 0.0, -2.0), Vector3::unit_y(), 2.5).is_ok());
    }

    #[test]
    fn a_line_needs_two_points() {
        assert!(polyline(&[p(0.0, 0.0, 0.0)]).is_err());
        assert!(polyline(&[p(0.0, 0.0, 0.0), p(1.0, 0.0, 1.0)]).is_ok());
    }

    #[test]
    fn a_closed_outline_needs_three_points() {
        assert!(closed_polyline(&[p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0)]).is_err());
        let square = [p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0), p(1.0, 0.0, 1.0), p(0.0, 0.0, 1.0)];
        let face = region(closed_polyline(&square).expect("the outline builds"));
        assert_eq!(face.shape_type(), ShapeType::Face);
    }

    #[test]
    fn an_open_curve_needs_two_points() {
        assert!(spline(&[p(0.0, 0.0, 0.0)]).is_err());
        assert!(spline(&[p(0.0, 0.0, 0.0), p(1.0, 0.0, 1.0)]).is_ok());
    }

    #[test]
    fn an_open_curve_passes_through_three_points() {
        assert!(spline(&[p(0.0, 0.0, 0.0), p(1.0, 0.0, 2.0), p(3.0, 0.0, 1.0)]).is_ok());
    }

    #[test]
    fn a_closed_curve_needs_three_points() {
        assert!(closed_spline(&[p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0)]).is_err());
        assert!(closed_spline(&[p(0.0, 0.0, 0.0), p(2.0, 0.0, 0.0), p(1.0, 0.0, 2.0)]).is_ok());
    }

    #[test]
    fn a_planar_closed_curve_bounds_a_face() {
        let points = [p(0.0, 0.0, 0.0), p(2.0, 0.0, 0.0), p(2.0, 0.0, 2.0), p(0.0, 0.0, 2.0)];
        let face = region(closed_spline(&points).expect("the curve builds"));
        assert_eq!(face.shape_type(), ShapeType::Face);
    }
}
