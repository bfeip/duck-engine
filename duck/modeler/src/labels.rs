//! Text over the 3D viewport, such as a measurement or what the pointer snaps
//! to. Tools report [`Label`]s at world points, through
//! [`ModelingTool::labels`](crate::tools::ModelingTool::labels); [`paint`]
//! draws them where those points land on the viewport image.

use std::f32::consts::{FRAC_PI_2, PI};

use duck_engine_viewer::common::{InnerSpace, LengthDisplay, MetricSpace, Point3};
use duck_engine_viewer::scene::PositionedCamera;
use egui::emath::Rot2;
use egui::epaint::TextShape;
use egui::{Align2, Painter, Pos2, Rect, Shape, Stroke, StrokeKind, Vec2};

use crate::document::DOCUMENT_UNITS;

/// Space between a dimension's text and the segment it runs along, in points.
const DIMENSION_GAP: f32 = 4.0;

/// Space between a dimension's text and the edge of the plate behind it, in
/// points.
const DIMENSION_PADDING: Vec2 = Vec2::new(3.0, 1.0);

/// How opaque the plate behind a dimension is, which keeps it legible over
/// bright faces and the background alike.
const DIMENSION_PLATE_OPACITY: f32 = 0.75;

/// Where a tag's badge sits from the point it marks: below right, clear of
/// the 3D cursor. In points.
const TAG_OFFSET: Vec2 = Vec2::new(12.0, 10.0);

/// Space between a tag's text and its badge's edge, in points.
const TAG_PADDING: Vec2 = Vec2::new(5.0, 2.0);

/// Text over the viewport, placed where world points land on screen.
#[derive(Clone, Debug, PartialEq)]
pub enum Label {
    /// The distance between two points, in the display units, written along
    /// the segment between them: a measurement.
    Dimension { from: Point3, to: Point3 },
    /// A short note beside a point, such as what the snap there locks onto.
    Tag { at: Point3, text: String },
}

/// Paints `labels` over `rect`, the image of `camera`'s view, with lengths
/// in `units`. A label at a point behind the eye is left out.
pub fn paint(
    painter: &Painter,
    rect: Rect,
    camera: &PositionedCamera,
    units: LengthDisplay,
    labels: &[Label],
) {
    let style = painter.ctx().style();
    let visuals = &style.visuals;
    let font = egui::TextStyle::Body.resolve(&style);
    for label in labels {
        match label {
            Label::Dimension { from, to } => {
                let (Some(a), Some(b)) =
                    (on_screen(camera, rect, *from), on_screen(camera, rect, *to))
                else {
                    continue;
                };
                let (anchor, angle) = along(a, b);
                let text = units.format(f64::from(from.distance(*to)), DOCUMENT_UNITS);
                let color = visuals.strong_text_color();
                let galley = painter.layout_no_wrap(text, font.clone(), color);
                // Laid out level with its bottom centre on the anchor, then
                // turned about it, plate and all.
                let text_rect = Align2::CENTER_BOTTOM.anchor_size(anchor, galley.size());
                let plate = text_rect.expand2(DIMENSION_PADDING);
                let turn = Rot2::from_angle(angle);
                let corners =
                    [plate.left_top(), plate.right_top(), plate.right_bottom(), plate.left_bottom()]
                        .map(|corner| anchor + turn * (corner - anchor));
                let fill = visuals.window_fill.gamma_multiply(DIMENSION_PLATE_OPACITY);
                painter.add(Shape::convex_polygon(corners.to_vec(), fill, Stroke::NONE));
                painter.add(
                    TextShape::new(text_rect.min, galley, color)
                        .with_angle_and_anchor(angle, Align2::CENTER_BOTTOM),
                );
            }
            Label::Tag { at, text } => {
                let Some(point) = on_screen(camera, rect, *at) else { continue };
                let galley = painter.layout_no_wrap(text.clone(), font.clone(), visuals.text_color());
                let text_rect =
                    Align2::LEFT_TOP.anchor_size(point + TAG_OFFSET + TAG_PADDING, galley.size());
                painter.rect(
                    text_rect.expand2(TAG_PADDING),
                    visuals.menu_corner_radius,
                    visuals.window_fill,
                    visuals.window_stroke,
                    StrokeKind::Inside,
                );
                painter.galley(text_rect.min, galley, visuals.text_color());
            }
        }
    }
}

/// Where `point` lands in `rect`, the image of `camera`'s view, or `None`
/// when it is behind a perspective eye.
fn on_screen(camera: &PositionedCamera, rect: Rect, point: Point3) -> Option<Pos2> {
    // The orthographic slab straddles the eye, so only perspective has a
    // behind.
    if !camera.projection.is_ortho() && (point - camera.eye).dot(camera.forward()) <= 0.0 {
        return None;
    }
    let ndc = camera.project_point_ndc(point);
    // NDC is Y-up; the rect is Y-down.
    let (x, y) = (ndc.x as f32, ndc.y as f32);
    Some(rect.lerp_inside(Vec2::new((x + 1.0) / 2.0, (1.0 - y) / 2.0)))
}

/// Where text running along the on-screen segment from `a` to `b` goes: the
/// bottom centre of the text, just off the segment's middle, and the
/// clockwise angle that turns the text to run with the segment.
///
/// The text never reads upside down: whichever way the segment runs, it reads
/// left to right, or bottom to top when upright, as drawings do.
fn along(a: Pos2, b: Pos2) -> (Pos2, f32) {
    let run = b - a;
    // Clockwise, as the screen is Y-down.
    let mut angle = run.y.atan2(run.x);
    if angle >= FRAC_PI_2 {
        angle -= PI;
    } else if angle < -FRAC_PI_2 {
        angle += PI;
    }
    // The text's up, turned with it.
    let above = Vec2::new(angle.sin(), -angle.cos());
    (a.lerp(b, 0.5) + above * DIMENSION_GAP, angle)
}

#[cfg(test)]
mod tests {
    use duck_engine_viewer::common::{LengthUnit, Real, Vector3};
    use duck_engine_viewer::scene::Projection;

    use super::*;

    const EPSILON: f32 = 1e-4;

    /// The image the views below fill: 800 × 600 points, off the origin.
    fn image() -> Rect {
        Rect::from_min_size(Pos2::new(200.0, 50.0), Vec2::new(800.0, 600.0))
    }

    /// Looking down -Z from 10 units away at the origin, Y up.
    fn camera(projection: Projection) -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 10.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::unit_y(),
            aspect: 800.0 / 600.0,
            projection,
        }
    }

    fn perspective() -> PositionedCamera {
        camera(Projection::Perspective { fovy: 45.0, znear: 0.1, zfar: 100.0 })
    }

    fn orthographic() -> PositionedCamera {
        camera(Projection::Orthographic { half_height: 5.0, half_depth: 50.0 })
    }

    fn near(a: Pos2, b: Pos2) -> bool {
        (a - b).length() < EPSILON
    }

    #[test]
    fn the_target_lands_in_the_middle_of_the_image() {
        for camera in [perspective(), orthographic()] {
            let at = on_screen(&camera, image(), camera.target).expect("the target is in view");
            assert!(near(at, image().center()), "{at:?}");
        }
    }

    /// Up in the world is up on screen, which is Y-down.
    #[test]
    fn a_point_above_the_target_lands_above_the_middle() {
        let up = Point3::new(0.0, 2.5, 0.0);
        let at = on_screen(&orthographic(), image(), up).unwrap();
        assert!(near(at, Pos2::new(600.0, 200.0)), "{at:?}");
    }

    #[test]
    fn a_point_behind_a_perspective_eye_is_nowhere() {
        assert_eq!(on_screen(&perspective(), image(), Point3::new(0.0, 0.0, 12.0)), None);
    }

    /// The slab straddles the eye, so what is behind it is still in view.
    #[test]
    fn a_point_behind_an_orthographic_eye_is_in_view() {
        let at = on_screen(&orthographic(), image(), Point3::new(0.0, 0.0, 12.0));
        assert!(at.is_some_and(|at| near(at, image().center())), "{at:?}");
    }

    #[test]
    fn text_along_a_segment_reads_the_same_whichever_way_it_runs() {
        let (a, b) = (Pos2::new(10.0, 40.0), Pos2::new(70.0, 0.0));
        let forward = along(a, b);
        let backward = along(b, a);
        assert!(near(forward.0, backward.0), "{forward:?} {backward:?}");
        assert!((forward.1 - backward.1).abs() < EPSILON, "{forward:?} {backward:?}");
        assert!(forward.1.abs() < FRAC_PI_2, "upside down at {}", forward.1);
    }

    /// Upright text reads bottom to top, to the left of the segment.
    #[test]
    fn text_along_an_upright_segment_reads_upward() {
        let (top, bottom) = (Pos2::new(50.0, 0.0), Pos2::new(50.0, 100.0));
        for (anchor, angle) in [along(top, bottom), along(bottom, top)] {
            assert!((angle + FRAC_PI_2).abs() < EPSILON, "turned {angle}");
            assert!(near(anchor, Pos2::new(50.0 - DIMENSION_GAP, 50.0)), "{anchor:?}");
        }
    }

    #[test]
    fn text_along_a_level_segment_sits_above_its_middle() {
        let (anchor, angle) = along(Pos2::new(100.0, 20.0), Pos2::new(0.0, 20.0));
        assert!(angle.abs() < EPSILON, "turned {angle}");
        assert!(near(anchor, Pos2::new(50.0, 20.0 - DIMENSION_GAP)), "{anchor:?}");
    }

    /// The shapes one frame of `labels` paints over the image, as seen through
    /// `camera`.
    fn painted(camera: &PositionedCamera, units: LengthDisplay, labels: &[Label]) -> Vec<Shape> {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 700.0))),
            ..Default::default()
        };
        let output = ctx.run(input, |ctx| {
            let painter = ctx.layer_painter(egui::LayerId::background());
            paint(&painter, image(), camera, units, labels);
        });
        output.shapes.into_iter().map(|clipped| clipped.shape).collect()
    }

    fn texts(shapes: &[Shape]) -> Vec<&TextShape> {
        shapes
            .iter()
            .filter_map(|shape| match shape {
                Shape::Text(text) => Some(text),
                _ => None,
            })
            .collect()
    }

    /// A diagonal from the origin up and right, 12.5 mm long.
    fn diagonal() -> Label {
        let length: Real = 12.5;
        let half = length / (2.0 as Real).sqrt();
        Label::Dimension { from: Point3::new(0.0, 0.0, 0.0), to: Point3::new(half, half, 0.0) }
    }

    #[test]
    fn a_dimension_writes_its_length_in_the_display_unit_along_itself() {
        for (unit, decimals, expected) in
            [(LengthUnit::Millimeter, 2, "12.50 mm"), (LengthUnit::Centimeter, 3, "1.250 cm")]
        {
            let shapes = painted(&orthographic(), LengthDisplay::new(unit, decimals), &[diagonal()]);
            let texts = texts(&shapes);
            assert_eq!(texts.len(), 1);
            let text = texts[0];
            assert_eq!(text.galley.text(), expected);
            // Up and right on screen is an eighth of a turn anticlockwise.
            assert!((text.angle + PI / 4.0).abs() < EPSILON, "turned {}", text.angle);

            // The plate behind it, turned with it.
            let Some(Shape::Path(plate)) = shapes.first() else { panic!("no plate in {shapes:?}") };
            assert_eq!(plate.points.len(), 4);
            let top = plate.points[1] - plate.points[0];
            assert!((top.y.atan2(top.x) - text.angle).abs() < EPSILON, "plate runs along {top:?}");
        }
    }

    #[test]
    fn a_tag_writes_upright_on_a_badge_below_right_of_its_point() {
        let tag = Label::Tag { at: Point3::new(0.0, 0.0, 0.0), text: "Midpoint".to_owned() };
        let units = LengthDisplay::new(LengthUnit::Millimeter, 2);
        let shapes = painted(&orthographic(), units, &[tag]);

        let Some(Shape::Rect(badge)) = shapes.first() else { panic!("no badge in {shapes:?}") };
        let texts = texts(&shapes);
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0].galley.text(), "Midpoint");
        assert_eq!(texts[0].angle, 0.0);
        assert!(badge.rect.contains(texts[0].pos), "{:?} outside {:?}", texts[0].pos, badge.rect);
        assert!(near(badge.rect.min, image().center() + TAG_OFFSET), "{:?}", badge.rect);
    }

    #[test]
    fn a_label_behind_the_eye_paints_nothing() {
        let behind = Point3::new(0.0, 0.0, 12.0);
        let labels = [
            Label::Dimension { from: Point3::new(0.0, 0.0, 0.0), to: behind },
            Label::Tag { at: behind, text: "Corner".to_owned() },
        ];
        let units = LengthDisplay::new(LengthUnit::Millimeter, 2);
        assert!(painted(&perspective(), units, &labels).is_empty());
    }
}
