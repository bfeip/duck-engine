use anyhow::{Context, Result};
use duck_engine_scene::cad::{tessellate_into, CadTessellationOptions};
use duck_engine_scene::common::{InnerSpace, Point3, Vector3};
use duck_engine_scene::NodeId;
use glam::DVec3;
use opencascade::primitives::{FaceOrientation, Shape, ShapeType};

use crate::document::{Document, PartId};

/// The sub-geometry being extruded, identified the way the selection system reports
/// it: by tessellation order within a part's mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtrudeTarget {
    /// Extrude a face into a solid, fusing it into the source body (a "pad").
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

/// World-space anchor and unit direction the extrusion grows along. The extrusion
/// itself will be some signed distance along the axis.
#[derive(Clone, Copy, Debug)]
pub struct ExtrudeFrame {
    pub origin: Point3,
    pub axis: Vector3,
}

impl ExtrudeFrame {
    /// Builds the fixed extrusion frame for `target`: the outward face normal for a
    /// face, or the sketch plane normal (perpendicular to a flat sketch) for an edge.
    ///
    /// `sketch_normal` is the modeler's construction-plane normal; it sets the edge
    /// extrusion direction so a sketch edge grows out of its plane into a wall.
    pub fn new(
        doc: &Document,
        target: ExtrudeTarget,
        sketch_normal: Vector3,
    ) -> Result<Self> {
        match target {
            ExtrudeTarget::Face { node, face_index } => {
                let face = doc
                    .face_subshape(node, face_index)
                    .context("Selected face is not part of a known CAD part")?;
                // `normal_at_center` returns the surface's parametric normal, which points
                // inward for Reversed faces — flip it so the pad grows outward.
                let sign = if face.orientation() == FaceOrientation::Forward { 1.0 } else { -1.0 };
                let n = face
                    .normal_at_center()
                    .context("Selected face has no well-defined extrusion direction")?;
                let axis = Vector3::new(n.x as f32, n.y as f32, n.z as f32) * sign;
                Ok(Self { origin: dvec3_to_point(face.center_of_mass()), axis: axis.normalize() })
            }
            ExtrudeTarget::Edge { node, edge_index } => {
                let edge = doc
                    .edge_subshape(node, edge_index)
                    .context("Selected edge is not part of a known CAD part")?;
                let (start, end) = (edge.start_point(), edge.end_point());
                let origin = dvec3_to_point((start + end) * 0.5);
                Ok(Self { origin, axis: sketch_normal.normalize() })
            }
        }
    }
}

/// The transient geometry produced by [`preview_extrude`].
pub struct ExtrudePreview {
    /// The temporary scene node holding the previewed geometry. The caller owns it
    /// and must remove it when the preview is rebuilt or the operation ends.
    pub node: NodeId,
    /// Whether the source node should be hidden while this preview stands in for it.
    pub hide_source: bool,
}

/// Non-destructive preview: tessellate just the raw extruded geometry as a temporary
/// scene node.
///
/// The preview deliberately skips the boolean fuse used at execution — recomputing it
/// every cursor move is far too expensive. A retained solid body simply stays visible
/// with the prism overlapping it (visually identical to the finished pad); only a bare
/// sketch region is hidden, since the prism fully stands in for it.
pub fn preview_extrude(
    doc: &Document,
    target: ExtrudeTarget,
    frame: &ExtrudeFrame,
    length: f64,
    options: &CadTessellationOptions,
) -> Result<ExtrudePreview> {
    let raw = raw_extrude(doc, target, frame, length)?;
    let hide_source = !raw.source_is_solid;
    let mut scene = doc.scene().lock().unwrap();
    let node = tessellate_into(&raw.prism, &mut *scene, options, None, Some("Extrude preview"))
        .context("Failed to tessellate extrude preview")?;
    Ok(ExtrudePreview { node, hide_source })
}

/// Apply the extrusion: build the final shape — fusing the pad into the source body
/// here, once — tessellate it into a new part, then remove the superseded source.
pub fn execute_extrude(
    doc: &mut Document,
    target: ExtrudeTarget,
    frame: &ExtrudeFrame,
    length: f64,
    options: &CadTessellationOptions,
) -> Result<()> {
    let raw = raw_extrude(doc, target, frame, length)?;

    let (shape, remove_source) = if raw.fuse_into_source {
        // The pad fuses into the source body (the expensive boolean, done just once),
        // and the fused result supersedes it. Fuse a deep copy: the OCCT boolean
        // may mutate its inputs, and the source must survive unchanged if the
        // fuse or the tessellation below fails.
        let part = doc.get_part(raw.source_part).context("Source part not found")?;
        (part.shape.deep_copy().union(&raw.prism).context("Failed to fuse pad into source")?.shape, true)
    } else {
        // Region→solid or edge→face: the raw geometry is the result. A bare sketch
        // region is superseded; a solid whose edge was extruded is kept alongside.
        (raw.prism, !raw.source_is_solid)
    };

    // One undo step covers the added extrusion and the superseded source.
    let mut doc = doc.undo_scope("Extrude");

    // Tessellates atomically — if this fails, nothing is changed.
    doc.add_part("Extrusion".to_owned(), shape, options)
        .context("Failed to tessellate extrusion")?;
    if remove_source {
        doc.remove_part(raw.source_part);
    }
    Ok(())
}

/// The standalone extruded geometry plus the classification needed to decide how it
/// combines with its source. Cheap to build — no boolean operations.
struct RawExtrude {
    /// The extruded geometry on its own: a prism solid (face target) or swept face
    /// (edge target).
    prism: Shape,
    source_part: PartId,
    /// Whether the source part is a solid body (vs a bare sketch region/face).
    source_is_solid: bool,
    /// Whether this is a face-pad that should fuse into its solid source on execute.
    fuse_into_source: bool,
}

/// Resolve the sub-shape and extrude it along `frame.axis`, without any boolean fuse.
fn raw_extrude(
    doc: &Document,
    target: ExtrudeTarget,
    frame: &ExtrudeFrame,
    length: f64,
) -> Result<RawExtrude> {
    let dir = DVec3::new(
        frame.axis.x as f64 * length,
        frame.axis.y as f64 * length,
        frame.axis.z as f64 * length,
    );

    let source_part = doc
        .part_for_node(target.node())
        .context("Extrude target is not a known CAD part")?;
    let part = doc.get_part(source_part).context("Source part not found")?;
    let source_is_solid = source_has_solid(&part.shape);

    let (prism, fuse_into_source) = match target {
        ExtrudeTarget::Face { node, face_index } => {
            let face = doc
                .face_subshape(node, face_index)
                .context("Selected face is not part of a known CAD part")?;
            // Only a face on a solid body fuses (a "pad"). A face that *is* a bare sketch
            // region extrudes into the prism directly — fusing a solid with the 2D region
            // it sprang from is degenerate.
            (face.extrude(dir).into(), source_is_solid)
        }
        ExtrudeTarget::Edge { node, edge_index } => {
            let edge = doc
                .edge_subshape(node, edge_index)
                .context("Selected edge is not part of a known CAD part")?;
            (edge.extrude(dir).into(), false)
        }
    };

    Ok(RawExtrude { prism, source_part, source_is_solid, fuse_into_source })
}

/// Whether `shape` is (or contains) a solid body, as opposed to a bare
/// region/sketch (face, shell, wire, edge, vertex).
fn source_has_solid(shape: &Shape) -> bool {
    !matches!(
        shape.shape_type(),
        ShapeType::Face | ShapeType::Shell | ShapeType::Wire | ShapeType::Edge | ShapeType::Vertex
    )
}

fn dvec3_to_point(v: DVec3) -> Point3 {
    Point3::new(v.x as f32, v.y as f32, v.z as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use duck_engine_scene::Scene;
    use opencascade::primitives::{Face, Wire};

    const SKETCH_NORMAL: Vector3 = Vector3::new(0.0, 1.0, 0.0);

    fn doc_with_box() -> (Document, NodeId) {
        let shape = opencascade::primitives::Shape::box_centered(2.0, 2.0, 2.0);
        doc_with_shape(shape)
    }

    fn doc_with_shape(shape: Shape) -> (Document, NodeId) {
        let scene = Arc::new(Mutex::new(Scene::new()));
        let mut doc = Document::new(scene);
        let part = doc
            .add_part("part", shape, &CadTessellationOptions::default())
            .expect("shape tessellates");
        let node = doc.node_for_part(part).expect("part has a node");
        (doc, node)
    }

    /// A closed planar region on the XZ plane, exactly what the line tool produces.
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

    #[test]
    fn box_face_extrude_fuses_and_replaces_source() {
        let (mut doc, node) = doc_with_box();
        assert_eq!(doc.parts().count(), 1);

        let target = ExtrudeTarget::Face { node, face_index: 0 };
        let frame = ExtrudeFrame::new(&doc, target, SKETCH_NORMAL).expect("face resolves");
        execute_extrude(&mut doc, target, &frame, 1.0, &CadTessellationOptions::default())
            .expect("face extrude succeeds");

        assert_eq!(doc.parts().count(), 1, "source box should be replaced by the pad");
    }

    #[test]
    fn region_extrude_produces_a_solid() {
        // Repro for the reported bug: extruding a closed sketch region must yield a
        // solid, not a degenerate union of a solid with the 2D region it grew from.
        let (mut doc, node) = doc_with_shape(region_shape());
        let target = ExtrudeTarget::Face { node, face_index: 0 };
        let frame = ExtrudeFrame::new(&doc, target, SKETCH_NORMAL).expect("region face resolves");
        execute_extrude(&mut doc, target, &frame, 2.0, &CadTessellationOptions::default())
            .expect("region extrude succeeds");

        assert_eq!(doc.parts().count(), 1, "region is replaced by its extrusion");
        let part = doc.parts().next().expect("one part remains");
        assert_eq!(part.shape.shape_type(), ShapeType::Solid, "extruded region must be a solid");
    }

    #[test]
    fn box_face_frame_axis_is_unit_and_axis_aligned() {
        let (doc, node) = doc_with_box();
        for face_index in 0..6 {
            let frame = ExtrudeFrame::new(&doc, ExtrudeTarget::Face { node, face_index }, SKETCH_NORMAL)
                .expect("box face resolves");
            assert!((frame.axis.magnitude() - 1.0).abs() < 1e-4, "axis should be unit");
            // A box face normal points along exactly one world axis.
            let aligned = [frame.axis.x.abs(), frame.axis.y.abs(), frame.axis.z.abs()]
                .iter()
                .filter(|c| (**c - 1.0).abs() < 1e-3)
                .count();
            assert_eq!(aligned, 1, "box face normal should be axis-aligned");
        }
    }

    #[test]
    fn box_edge_extrude_keeps_solid_and_adds_face() {
        let (mut doc, node) = doc_with_box();
        let target = ExtrudeTarget::Edge { node, edge_index: 0 };
        let frame = ExtrudeFrame::new(&doc, target, SKETCH_NORMAL).expect("edge resolves");
        execute_extrude(&mut doc, target, &frame, 1.0, &CadTessellationOptions::default())
            .expect("edge extrude succeeds");
        // Extruding an edge of a solid must not delete the solid.
        assert_eq!(doc.parts().count(), 2, "solid kept, extruded face added");
    }
}
