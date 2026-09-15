use anyhow::{Context, Result};
use duck_engine_scene::common::{matrix4_to_row_major_f64, Matrix4};
use opencascade::primitives::Shape;

use crate::document::{Document, PartId};

/// Copy each of `sources` once per entry in `placements`, transforming each
/// copy's B-Rep by its placement. Results are source-major: every placement of
/// the first source, then of the second, and so on.
///
/// Copies are added under the next free name in their source's series and carry
/// the source's tessellation options.
///
/// Opens its own undo scope, which also covers the whole run when a caller
/// wants a different label: nested scopes merge and the outermost label wins.
pub fn duplicate_parts(
    doc: &mut Document,
    sources: &[PartId],
    placements: &[Matrix4],
) -> Result<Vec<PartId>> {
    let mut doc = doc.undo_scope("Duplicate");
    let mut copies = Vec::with_capacity(sources.len() * placements.len());
    for &source in sources {
        for placement in placements {
            // Read the source out before `add_part` takes the document mutably.
            let (shape, options) = {
                let part = doc.get_part(source).context("Duplicate: source part not found")?;
                (part.shape.deep_copy(), part.options().clone())
            };
            let name = doc.duplicate_name(source);
            let shape = place(shape, placement);
            let id = doc
                .add_part(name, shape, &options)
                .context("Failed to tessellate the duplicated part")?;
            copies.push(id);
        }
    }
    Ok(copies)
}

/// Move `shape` by `placement`. A similarity keeps surfaces analytic (planes
/// stay planes); only a non-uniform scale needs the B-spline-converting
/// general transform.
fn place(shape: Shape, placement: &Matrix4) -> Shape {
    let mat = matrix4_to_row_major_f64(placement);
    match shape.transformed(mat) {
        Ok(placed) => placed,
        Err(_) => shape.gtransform(mat),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::cad::CadTessellationOptions;
    use duck_engine_scene::common::{Point3, SquareMatrix, Vector3};
    use duck_engine_scene::Scene;

    const EPSILON: f32 = 1e-4;

    /// A document holding one unit box named `Box-001`.
    fn doc_with_box() -> (Document, PartId) {
        let mut doc = Document::new(Scene::default());
        let part = doc
            .add_part("Box-001", Shape::cube_centered(1.0), &CadTessellationOptions::default())
            .expect("tessellates");
        (doc, part)
    }

    /// Center of a part's tessellated bounds in the scene — what the user sees.
    fn center(doc: &Document, part: PartId) -> Point3 {
        let node = doc.node_for_part(part).expect("node");
        let scene = doc.scene().lock();
        scene.nodes_bounding(node).bounds.expect("bounds").center()
    }

    fn assert_close(actual: Point3, expected: Point3) {
        let close = (actual.x - expected.x).abs() < EPSILON
            && (actual.y - expected.y).abs() < EPSILON
            && (actual.z - expected.z).abs() < EPSILON;
        assert!(close, "expected {expected:?}, got {actual:?}");
    }

    #[test]
    fn copy_is_an_independent_part() {
        let (mut doc, source) = doc_with_box();
        let copies = duplicate_parts(&mut doc, &[source], &[Matrix4::identity()])
            .expect("duplicate succeeds");

        assert_eq!(copies.len(), 1);
        let copy = copies[0];
        assert_ne!(copy, source, "the copy is its own part");
        assert_ne!(
            doc.node_for_part(copy),
            doc.node_for_part(source),
            "the copy is its own scene node"
        );
        assert_eq!(doc.get_part(copy).unwrap().kind(), doc.get_part(source).unwrap().kind());
        // `deep_copy`, not the shallow `Clone`: a shared TShape would let a later
        // destructive operation on one part corrupt the other.
        assert!(
            !doc.get_part(copy).unwrap().shape.is_same(&doc.get_part(source).unwrap().shape),
            "the copy must share no B-Rep data with its source"
        );
    }

    #[test]
    fn copy_takes_the_next_name_in_the_series() {
        let (mut doc, source) = doc_with_box();
        let copies = duplicate_parts(&mut doc, &[source], &[Matrix4::identity()]).expect("copies");
        assert_eq!(doc.get_part(copies[0]).unwrap().name, "Box-002");
    }

    #[test]
    fn placement_moves_the_copy_and_leaves_the_source() {
        let (mut doc, source) = doc_with_box();
        let before = center(&doc, source);
        let offset = Vector3::new(5.0, 0.0, -2.0);

        let copies = duplicate_parts(&mut doc, &[source], &[Matrix4::from_translation(offset)])
            .expect("duplicate succeeds");

        assert_close(center(&doc, copies[0]), before + offset);
        assert_close(center(&doc, source), before);
    }

    #[test]
    fn every_source_is_copied_once_per_placement() {
        let (mut doc, first) = doc_with_box();
        let second = doc
            .add_part("Box-002", Shape::cube_centered(1.0), &CadTessellationOptions::default())
            .expect("tessellates");
        let placements = [
            Matrix4::from_translation(Vector3::new(2.0, 0.0, 0.0)),
            Matrix4::from_translation(Vector3::new(4.0, 0.0, 0.0)),
            Matrix4::from_translation(Vector3::new(6.0, 0.0, 0.0)),
        ];

        let copies = duplicate_parts(&mut doc, &[first, second], &placements).expect("copies");

        assert_eq!(copies.len(), 6);
        // Source-major: the first source's three placements come first.
        let first_center = center(&doc, first);
        for (i, &copy) in copies[..3].iter().enumerate() {
            assert_close(center(&doc, copy), first_center + Vector3::new(2.0 * (i as f32 + 1.0), 0.0, 0.0));
        }
    }

    #[test]
    fn the_whole_run_is_one_undo_step() {
        let (mut doc, source) = doc_with_box();
        let before = doc.parts().count();
        let placements = [
            Matrix4::from_translation(Vector3::new(2.0, 0.0, 0.0)),
            Matrix4::from_translation(Vector3::new(4.0, 0.0, 0.0)),
        ];
        duplicate_parts(&mut doc, &[source], &placements).expect("copies");
        assert_eq!(doc.parts().count(), before + 2);

        doc.undo().expect("undo succeeds");
        assert_eq!(doc.parts().count(), before, "one step removes every copy");
    }
}
