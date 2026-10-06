use anyhow::{ensure, Context, Result};
use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::common::{EuclideanSpace, InnerSpace, Point3, Real, Vector3};
use duck_engine_scene::resource::NodeId;
use opencascade::history::ShapeHistory;
use opencascade::primitives::{Face, JoinType, Shape};

use crate::document::{
    dvec3_to_point3, dvec3_to_vec3, has_solid, vec3_to_dvec3, Document, SourceFate,
};

/// A distance at or below this is degenerate: there is nothing to extrude.
const MIN_DISTANCE: Real = 1e-6;

/// A draft at or below this many radians leaves the walls straight; OCCT
/// ignores anything smaller.
const MIN_DRAFT: Real = 1e-4;

/// A wall thickness at or below this is no wall at all.
const MIN_THICKNESS: Real = 1e-6;

/// The sub-geometry being extruded, identified the way the selection system reports
/// it: by tessellation order within a part's mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtrudeTarget {
    /// Extrude a face into a solid.
    Face { node: NodeId, face_index: u32 },
    /// Extrude an edge into a face.
    Edge { node: NodeId, edge_index: u32 },
}

impl ExtrudeTarget {
    pub fn node(&self) -> NodeId {
        match *self {
            ExtrudeTarget::Face { node, .. } | ExtrudeTarget::Edge { node, .. } => node,
        }
    }
}

/// Where an extrusion grows from, resolved once from its target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtrudeFrame {
    pub target: ExtrudeTarget,
    /// The face's centre of mass, or the edge's midpoint.
    pub origin: Point3,
    /// Unit direction out of the profile: the outward normal of a face, or the
    /// sketch plane normal for an edge.
    pub normal: Vector3,
    pub fate: SourceFate,
}

impl ExtrudeFrame {
    /// Resolves `target` against its part.
    ///
    /// `sketch_normal` is the modeler's construction-plane normal; it sets the
    /// edge extrusion direction so a sketch edge grows out of its plane into a wall.
    pub fn new(doc: &Document, target: ExtrudeTarget, sketch_normal: Vector3) -> Result<Self> {
        let source = doc
            .part_for_node(target.node())
            .and_then(|part| doc.get_part(part))
            .context("Extrude target is not a known CAD part")?;
        let source_is_solid = has_solid(&source.shape);

        let (origin, normal) = match target {
            ExtrudeTarget::Face { node, face_index } => {
                let face = doc
                    .face_subshape(node, face_index)
                    .context("Selected face is not part of a known CAD part")?;
                // `normal_at_center` goes through `BRepGProp_Face::Normal`, which already
                // applies the face's orientation — the normal points out of the material
                // for Reversed and Forward faces alike, so it needs no sign correction.
                let normal = face
                    .normal_at_center()
                    .context("Selected face has no well-defined extrusion direction")?;
                (dvec3_to_point3(face.center_of_mass()), dvec3_to_vec3(normal).normalize())
            }
            ExtrudeTarget::Edge { node, edge_index } => {
                let edge = doc
                    .edge_subshape(node, edge_index)
                    .context("Selected edge is not part of a known CAD part")?;
                let midpoint = (edge.start_point() + edge.end_point()) * 0.5;
                (dvec3_to_point3(midpoint), sketch_normal.normalize())
            }
        };

        // A face of a solid pads it. A face or edge of a bare sketch region grows
        // into what replaces the sketch. An edge of a solid grows a wall beside it.
        let fate = match (target, source_is_solid) {
            (_, false) => SourceFate::Replace,
            (ExtrudeTarget::Face { .. }, true) => SourceFate::Fuse,
            (ExtrudeTarget::Edge { .. }, true) => SourceFate::Keep,
        };

        Ok(Self { target, origin, normal, fate })
    }
}

/// An extrusion as configured: the profile it grows from, and how far and
/// which way it grows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtrudeParams {
    pub frame: ExtrudeFrame,
    /// Unit direction the profile sweeps along. Starts as the frame normal.
    pub direction: Vector3,
    /// Signed length along `direction`, never below
    /// [`min_distance`](Self::min_distance).
    pub distance: Real,
    /// Taper of the side walls away from `direction`, in radians. Positive
    /// narrows the extrusion as it grows; negative flares it.
    pub draft: Real,
    /// Wall thickness, zero for none. A face's walls grow inside its outline
    /// when positive and outside it when negative; an edge's wall grows to one
    /// side of its sheet or the other.
    pub thickness: Real,
}

impl ExtrudeParams {
    /// A zero-length extrusion straight out of `frame`.
    pub fn new(frame: ExtrudeFrame) -> Self {
        Self { frame, direction: frame.normal, distance: 0.0, draft: 0.0, thickness: 0.0 }
    }

    /// The end of the extrusion's axis.
    pub fn tip(&self) -> Point3 {
        self.frame.origin + self.direction * self.distance
    }

    /// The shortest distance the target allows. A face of a solid only adds
    /// material, so its extrusion never runs back into the body.
    pub fn min_distance(&self) -> Real {
        match self.frame.fate {
            SourceFate::Fuse => 0.0,
            SourceFate::Replace | SourceFate::Keep => Real::MIN,
        }
    }

    /// How far `direction` leans from the profile normal, in radians.
    pub fn tilt(&self) -> Real {
        self.direction.dot(self.frame.normal).clamp(-1.0, 1.0).acos()
    }

    /// Whether the extrusion is too short to build.
    pub fn is_degenerate(&self) -> bool {
        self.distance.abs() <= MIN_DISTANCE
    }

    /// Whether the side walls taper.
    pub fn has_draft(&self) -> bool {
        self.draft.abs() > MIN_DRAFT
    }

    /// Whether the extrusion is walls rather than a solid (a face) or a sheet
    /// (an edge).
    pub fn has_thickness(&self) -> bool {
        self.thickness.abs() > MIN_THICKNESS
    }
}

/// The extruded geometry on its own, before it is combined with its source:
/// a prism solid for a face, a swept face for an edge. Either is tapered by the
/// draft, then made into walls by the thickness.
pub fn build_extrusion(doc: &Document, params: &ExtrudeParams) -> Result<Shape> {
    ensure!(!params.is_degenerate(), "Nothing to extrude at zero distance");
    let sweep = vec3_to_dvec3(params.direction * params.distance);

    match params.frame.target {
        ExtrudeTarget::Face { node, face_index } => {
            let face = doc
                .face_subshape(node, face_index)
                .context("Selected face is not part of a known CAD part")?;
            let (prism, first, last) = face.extrude_with_caps(sweep);
            let mut prism = Shape::from(prism);
            let mut caps = vec![first, last];

            if params.has_draft() {
                let sides: Vec<Face> = prism
                    .faces()
                    .filter(|face| !caps.iter().any(|cap| cap.is_same(face)))
                    .collect();
                let (drafted, history) = draft(&prism, &sides, params)?;
                // A cap the draft left untouched is its own image.
                caps = caps
                    .into_iter()
                    .flat_map(|cap| match history.modified_faces(&cap) {
                        images if images.is_empty() => vec![cap],
                        images => images,
                    })
                    .collect();
                prism = drafted;
            }
            if params.has_thickness() {
                // Open at both caps, leaving only the walls. Intersection joins
                // keep their corners sharp where a concave outline pulls them apart.
                prism = prism
                    .hollow(-f64::from(params.thickness), &caps, JoinType::Intersection)
                    .context("Wall thickness failed")?;
            }
            Ok(prism)
        }
        ExtrudeTarget::Edge { node, edge_index } => {
            let edge = doc
                .edge_subshape(node, edge_index)
                .context("Selected edge is not part of a known CAD part")?;
            let mut wall: Shape = edge.extrude(sweep).into();

            if params.has_draft() {
                let sides: Vec<Face> = wall.faces().collect();
                wall = draft(&wall, &sides, params)?.0;
            }
            if params.has_thickness() {
                wall = wall
                    .thicken(f64::from(params.thickness), JoinType::Intersection)
                    .context("Wall thickness failed")?;
            }
            Ok(wall)
        }
    }
}

/// Tapers the `sides` of `shape` by the draft angle, pivoting where they cross
/// the profile's plane.
fn draft(shape: &Shape, sides: &[Face], params: &ExtrudeParams) -> Result<(Shape, ShapeHistory)> {
    // Pulled the way the extrusion grows, so that a positive draft narrows it.
    let pull = params.direction * params.distance.signum();
    shape
        .draft_faces_with_history(
            sides,
            vec3_to_dvec3(pull),
            f64::from(params.draft),
            vec3_to_dvec3(params.frame.origin.to_vec()),
            vec3_to_dvec3(params.frame.normal),
        )
        .context("Draft angle not supported for this profile")
}

/// Apply the extrusion: build it and commit it with its source as the frame's
/// [`SourceFate`] says, as one undo step.
pub fn execute_extrude(
    doc: &mut Document,
    params: &ExtrudeParams,
    options: &CadTessellationOptions,
) -> Result<()> {
    let extrusion = build_extrusion(doc, params)?;
    let source = doc
        .part_for_node(params.frame.target.node())
        .context("Extrude target is not a known CAD part")?;
    doc.commit_result(source, extrusion, params.frame.fate, "Extrude", "Extrusion", options)
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::common::consts;
    use duck_engine_scene::Scene;
    use opencascade::primitives::{Edge, ShapeType, Wire};

    use crate::document::PartKind;

    const SKETCH_NORMAL: Vector3 = Vector3::new(0.0, 1.0, 0.0);

    fn doc_with_box() -> (Document, NodeId) {
        let shape = opencascade::primitives::Shape::box_centered(2.0, 2.0, 2.0);
        doc_with_shape(shape)
    }

    fn doc_with_shape(shape: Shape) -> (Document, NodeId) {
        let scene = Scene::default();
        let mut doc = Document::new(scene);
        let part = doc
            .add_part("part", shape, &CadTessellationOptions::default())
            .expect("shape tessellates");
        let node = doc.node_for_part(part).expect("part has a node");
        (doc, node)
    }

    /// A closed unit-square planar region on the XZ plane, exactly what the line
    /// tool produces.
    fn region_shape() -> Shape {
        let wire = Wire::from_ordered_points([
            glam::dvec3(0.0, 0.0, 0.0),
            glam::dvec3(1.0, 0.0, 0.0),
            glam::dvec3(1.0, 0.0, 1.0),
            glam::dvec3(0.0, 0.0, 1.0),
        ])
        .expect("wire builds");
        Face::from_wire(&wire).expect("face builds").into()
    }

    /// A closed planar region on the XZ plane through `(x, z)` corners.
    fn polygon_region(corners: &[(f64, f64)]) -> Shape {
        let wire = Wire::from_ordered_points(corners.iter().map(|&(x, z)| glam::dvec3(x, 0.0, z)))
            .expect("wire builds");
        Face::from_wire(&wire).expect("face builds").into()
    }

    /// A region bounded by a single closed edge.
    fn edge_region(edge: Edge) -> Shape {
        let wire = Wire::from_edges([&edge]).expect("wire builds");
        Face::from_wire(&wire).expect("face builds").into()
    }

    /// Volume of a frustum between squares of side `a` and `b`, `height` apart.
    fn square_frustum(a: f64, b: f64, height: f64) -> f64 {
        height / 3.0 * (a * a + b * b + a * b)
    }

    /// Extrudes `params` and returns the resulting part's volume.
    fn extruded_volume(doc: &mut Document, params: &ExtrudeParams) -> f64 {
        execute_extrude(doc, params, &CadTessellationOptions::default()).expect("extrude succeeds");
        let part = doc.parts().last().expect("a part remains");
        assert_eq!(part.shape.shape_type(), ShapeType::Solid, "the extrusion must be a solid");
        part.shape.volume()
    }

    /// An extrusion of `target` by `distance` straight out of its profile.
    fn params(doc: &Document, target: ExtrudeTarget, distance: Real) -> ExtrudeParams {
        let frame = ExtrudeFrame::new(doc, target, SKETCH_NORMAL).expect("target resolves");
        ExtrudeParams { distance, ..ExtrudeParams::new(frame) }
    }

    #[test]
    fn box_face_extrude_fuses_and_replaces_source() {
        let (mut doc, node) = doc_with_box();
        assert_eq!(doc.parts().count(), 1);

        let params = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, 1.0);
        execute_extrude(&mut doc, &params, &CadTessellationOptions::default())
            .expect("face extrude succeeds");

        assert_eq!(doc.parts().count(), 1, "source box should be replaced by the pad");
        let part = doc.parts().next().expect("one part remains");
        assert_eq!(part.name, "part", "a result that supersedes its source keeps its name");
    }

    #[test]
    fn region_extrude_produces_a_solid() {
        // Repro for the reported bug: extruding a closed sketch region must yield a
        // solid, not a degenerate union of a solid with the 2D region it grew from.
        let (mut doc, node) = doc_with_shape(region_shape());
        let params = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, 2.0);
        execute_extrude(&mut doc, &params, &CadTessellationOptions::default())
            .expect("region extrude succeeds");

        assert_eq!(doc.parts().count(), 1, "region is replaced by its extrusion");
        let part = doc.parts().next().expect("one part remains");
        assert_eq!(part.shape.shape_type(), ShapeType::Solid, "extruded region must be a solid");
    }

    /// A pad fused into its source body must come back as a plain solid — an OCCT
    /// boolean always wraps its result in a compound — and must add material on
    /// every face, including the Reversed ones.
    #[test]
    fn box_face_extrude_produces_a_solid() {
        for face_index in 0..6 {
            let (mut doc, node) = doc_with_box();
            let params = params(&doc, ExtrudeTarget::Face { node, face_index }, 1.0);
            execute_extrude(&mut doc, &params, &CadTessellationOptions::default())
                .expect("face extrude succeeds");

            let part = doc.parts().next().expect("one part remains");
            assert_eq!(
                part.shape.shape_type(),
                ShapeType::Solid,
                "a fused pad must be a solid, face {face_index}"
            );
            assert_eq!(part.kind(), PartKind::Solid);
            assert!(
                (part.shape.volume() - (8.0 + 4.0)).abs() < 1e-6,
                "face {face_index}: expected a 2×2×2 box plus a 2×2×1 pad, got {}",
                part.shape.volume()
            );
        }
    }

    #[test]
    fn box_face_frame_normal_is_unit_and_points_outward() {
        let (doc, node) = doc_with_box();
        for face_index in 0..6 {
            let frame = ExtrudeFrame::new(&doc, ExtrudeTarget::Face { node, face_index }, SKETCH_NORMAL)
                .expect("box face resolves");
            assert!((frame.normal.magnitude() - 1.0).abs() < 1e-4, "normal should be unit");
            // A box face normal points along exactly one world axis.
            let aligned = [frame.normal.x.abs(), frame.normal.y.abs(), frame.normal.z.abs()]
                .iter()
                .filter(|c| (**c - 1.0).abs() < 1e-3)
                .count();
            assert_eq!(aligned, 1, "box face normal should be axis-aligned");
            // The box is centered on the origin, so its outward normals point away
            // from it. A Reversed face whose normal was wrongly flipped would aim
            // back into the body and pad nothing.
            let outward = frame.normal.x * frame.origin.x
                + frame.normal.y * frame.origin.y
                + frame.normal.z * frame.origin.z;
            assert!(outward > 0.0, "face {face_index} normal points into the body");
        }
    }

    #[test]
    fn box_edge_extrude_keeps_solid_and_adds_face() {
        let (mut doc, node) = doc_with_box();
        let params = params(&doc, ExtrudeTarget::Edge { node, edge_index: 0 }, 1.0);
        execute_extrude(&mut doc, &params, &CadTessellationOptions::default())
            .expect("edge extrude succeeds");
        // Extruding an edge of a solid must not delete the solid.
        assert_eq!(doc.parts().count(), 2, "solid kept, extruded face added");
        let names: Vec<_> = doc.parts().map(|p| p.name.as_str()).collect();
        assert!(
            names.contains(&"part") && names.contains(&"Extrusion-001"),
            "the untouched source keeps its name and the new face is numbered, got {names:?}"
        );
    }

    #[test]
    fn the_fate_follows_the_target_and_its_source() {
        let fate = |doc: &Document, target| {
            ExtrudeFrame::new(doc, target, SKETCH_NORMAL).expect("target resolves").fate
        };

        let (doc, node) = doc_with_box();
        assert_eq!(fate(&doc, ExtrudeTarget::Face { node, face_index: 0 }), SourceFate::Fuse);
        assert_eq!(fate(&doc, ExtrudeTarget::Edge { node, edge_index: 0 }), SourceFate::Keep);

        let (doc, node) = doc_with_shape(region_shape());
        assert_eq!(fate(&doc, ExtrudeTarget::Face { node, face_index: 0 }), SourceFate::Replace);
        assert_eq!(fate(&doc, ExtrudeTarget::Edge { node, edge_index: 0 }), SourceFate::Replace);
    }

    /// A pad only adds material; everything else may grow to either side.
    #[test]
    fn only_a_pad_is_held_out_of_its_body() {
        let (doc, node) = doc_with_box();
        let pad = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, 1.0);
        assert_eq!(pad.min_distance(), 0.0);
        let wall = params(&doc, ExtrudeTarget::Edge { node, edge_index: 0 }, 1.0);
        assert!(wall.min_distance() < 0.0);

        let (doc, node) = doc_with_shape(region_shape());
        let region = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, 1.0);
        assert!(region.min_distance() < 0.0);
    }

    #[test]
    fn a_region_extrudes_to_either_side() {
        let (mut doc, node) = doc_with_shape(region_shape());
        let params = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, -2.0);
        execute_extrude(&mut doc, &params, &CadTessellationOptions::default())
            .expect("a backwards region extrude succeeds");

        let part = doc.parts().next().expect("one part remains");
        assert_eq!(part.shape.shape_type(), ShapeType::Solid);
        assert!((part.shape.volume() - 2.0).abs() < 1e-6, "got {}", part.shape.volume());
    }

    /// A tilted sweep shears the prism: its volume is the base area times the
    /// height the tilted distance gains along the normal.
    #[test]
    fn a_tilted_extrusion_keeps_its_base_and_height() {
        let (mut doc, node) = doc_with_shape(region_shape());
        let mut params = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, 2.0);
        params.direction = (params.frame.normal + Vector3::unit_x()).normalize();
        let height = params.distance * params.direction.dot(params.frame.normal);
        assert!((params.tilt() - consts::FRAC_PI_4).abs() < 1e-5);

        execute_extrude(&mut doc, &params, &CadTessellationOptions::default())
            .expect("a tilted extrude succeeds");

        let part = doc.parts().next().expect("one part remains");
        let expected = 1.0 * f64::from(height);
        assert!(
            (part.shape.volume() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            part.shape.volume()
        );
    }

    /// Every side leans in by the draft, so a square region grows into a
    /// frustum; a negative draft flares it instead.
    #[test]
    fn a_draft_tapers_a_region_into_a_frustum() {
        let (height, angle) = (1.0, Real::to_radians(10.0));
        for sign in [1.0, -1.0] {
            let (mut doc, node) = doc_with_shape(polygon_region(&[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]));
            let mut params = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, height);
            params.draft = sign * angle;

            let top = 2.0 - 2.0 * f64::from(sign * height) * f64::from(angle).tan();
            let expected = square_frustum(2.0, top, f64::from(height));
            let volume = extruded_volume(&mut doc, &params);
            assert!((volume - expected).abs() < 1e-4, "sign {sign}: expected {expected}, got {volume}");
        }
    }

    /// Walls on a face of a solid leave the face as the floor of a cup.
    #[test]
    fn a_thick_pad_fuses_into_a_cup() {
        let (mut doc, node) = doc_with_box();
        let mut params = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, 1.0);
        params.thickness = 0.25;

        let expected = 8.0 + (4.0 - 1.5 * 1.5) * 1.0;
        let volume = extruded_volume(&mut doc, &params);
        assert!((volume - expected).abs() < 1e-5, "expected {expected}, got {volume}");
        assert_eq!(doc.parts().count(), 1, "the cup replaces the box");
    }

    /// Hollowed after the draft, the walls keep one thickness all the way up:
    /// the cavity is the outer frustum inset by the thickness across each
    /// leaning wall.
    #[test]
    fn drafted_walls_stay_parallel() {
        let (height, angle, thickness) = (1.0, Real::to_radians(10.0), 0.2);
        let (mut doc, node) = doc_with_shape(polygon_region(&[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]));
        let mut params = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, height);
        params.draft = angle;
        params.thickness = thickness;

        let (h, angle, t) = (f64::from(height), f64::from(angle), f64::from(thickness));
        let top = 2.0 - 2.0 * h * angle.tan();
        let inset = 2.0 * t / angle.cos();
        let expected = square_frustum(2.0, top, h) - square_frustum(2.0 - inset, top - inset, h);
        let volume = extruded_volume(&mut doc, &params);
        assert!((volume - expected).abs() < 1e-4, "expected {expected}, got {volume}");
    }

    /// Where a concave outline pulls the walls apart, their inner corner stays
    /// sharp rather than rounding off.
    #[test]
    fn a_concave_region_keeps_sharp_inner_corners() {
        let l_shape = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)];
        let (mut doc, node) = doc_with_shape(polygon_region(&l_shape));
        let mut params = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, 1.0);
        params.thickness = 0.1;

        // The L's area less the L inset by the thickness on every side.
        let expected = 3.0 - (1.8 * 0.8 + 0.8 * 1.0);
        let volume = extruded_volume(&mut doc, &params);
        assert!((volume - expected).abs() < 1e-5, "expected {expected}, got {volume}");
    }

    #[test]
    fn a_circular_region_hollows_into_a_tube() {
        let circle = Edge::circle(glam::DVec3::ZERO, glam::DVec3::Y, 1.0).expect("circle builds");
        let (mut doc, node) = doc_with_shape(edge_region(circle));
        let mut params = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, 1.0);
        params.thickness = 0.1;

        let expected = std::f64::consts::PI * (1.0 - 0.9 * 0.9);
        let volume = extruded_volume(&mut doc, &params);
        assert!((volume - expected).abs() < 1e-4, "expected {expected}, got {volume}");
    }

    /// A thick edge extrusion is a solid wall rather than a sheet.
    #[test]
    fn a_thick_edge_grows_a_solid_wall() {
        let (doc, node) = doc_with_shape(region_shape());
        for thickness in [0.1, -0.1] {
            let mut params = params(&doc, ExtrudeTarget::Edge { node, edge_index: 0 }, 1.0);
            params.thickness = thickness;
            let wall = build_extrusion(&doc, &params).expect("a thick edge extrudes");
            assert_eq!(wall.shape_type(), ShapeType::Solid, "thickness {thickness}");
            assert!((wall.volume() - 0.1).abs() < 1e-6, "thickness {thickness}: got {}", wall.volume());
        }
    }

    /// A spline's swept sides can't be tilted by OCCT's draft; that is reported,
    /// not crashed on.
    #[test]
    fn a_spline_region_refuses_a_draft() {
        let spline = Edge::spline_from_points(
            [
                glam::dvec3(0.0, 0.0, 0.0),
                glam::dvec3(2.0, 0.0, 0.5),
                glam::dvec3(1.5, 0.0, 2.0),
                glam::dvec3(-0.5, 0.0, 1.0),
            ],
            None,
            true,
        )
        .expect("spline builds");
        let (doc, node) = doc_with_shape(edge_region(spline));
        let mut params = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, 1.0);
        params.draft = Real::to_radians(5.0);

        let Err(error) = build_extrusion(&doc, &params) else {
            panic!("a spline side cannot take a draft");
        };
        assert!(format!("{error:#}").contains("Draft angle not supported"), "got {error:#}");
    }

    /// Zero length has no prism: the build refuses rather than handing OCCT a
    /// null sweep vector.
    #[test]
    fn a_zero_length_extrusion_is_refused() {
        let (doc, node) = doc_with_box();
        let params = params(&doc, ExtrudeTarget::Face { node, face_index: 0 }, 0.0);
        assert!(params.is_degenerate());
        assert!(build_extrusion(&doc, &params).is_err());
    }
}
