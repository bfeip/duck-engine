use anyhow::{bail, Context, Result};
use duck_engine_scene::cad::{tessellate_into, CadTessellationOptions};
use duck_engine_scene::NodeId;
use opencascade::primitives::{Shape, Shell, Solid, Wire};

use crate::document::Document;

/// What a loft produces from its profile wires.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LoftKind {
    /// An open shell skinned through the profiles (no end caps).
    #[default]
    Surface,
    /// A capped solid body through the profiles. Requires closed profile wires.
    Solid,
}

/// A single loft profile, identified the way the selection system reports it: by
/// the tessellation order of the edge the user clicked within a part's mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoftProfile {
    pub node: NodeId,
    pub edge_index: u32,
}

/// Resolve each picked edge to its containing wire, deduplicating profiles whose
/// edges resolve to the same wire (so two clicks on one rectangle count once) and
/// preserving click order. Errors if fewer than two distinct profiles remain.
fn resolve_profiles(doc: &Document, profiles: &[LoftProfile]) -> Result<Vec<Wire>> {
    let mut wires: Vec<Wire> = Vec::new();
    for profile in profiles {
        let wire = doc
            .wire_for_edge(profile.node, profile.edge_index)
            .context("Selected edge is not part of a known CAD wire")?;
        // Skip a wire we already have: it shares an edge with an accumulated profile.
        let duplicate = wires
            .iter()
            .any(|w| w.edges().any(|a| wire.edges().any(|b| a.is_same(&b))));
        if !duplicate {
            wires.push(wire);
        }
    }
    if wires.len() < 2 {
        bail!("A loft needs at least two distinct profiles");
    }
    Ok(wires)
}

/// Skin a surface or solid through the profile wires.
fn build_loft(wires: &[Wire], kind: LoftKind) -> Shape {
    match kind {
        LoftKind::Surface => Shell::loft(wires).into(),
        LoftKind::Solid => Solid::loft(wires).into(),
    }
}

/// Non-destructive preview: skin the loft and add a temporary scene node without
/// modifying the source parts. The caller owns the returned `NodeId` and must
/// remove it when the preview is rebuilt or the operation ends.
pub fn preview_loft(
    doc: &Document,
    profiles: &[LoftProfile],
    kind: LoftKind,
    options: &CadTessellationOptions,
) -> Result<NodeId> {
    let wires = resolve_profiles(doc, profiles)?;
    let shape = build_loft(&wires, kind);
    let mut scene = doc.scene().lock().unwrap();
    tessellate_into(&shape, &mut *scene, options, None, Some("Loft preview"))
        .context("Failed to tessellate loft preview")
}

/// Apply the loft: build the final shape and tessellate it into a new part. The
/// source profile bodies are kept — a loft is additive, and profiles are usually
/// reused as construction curves.
pub fn execute_loft(
    doc: &mut Document,
    profiles: &[LoftProfile],
    kind: LoftKind,
    options: &CadTessellationOptions,
) -> Result<()> {
    let wires = resolve_profiles(doc, profiles)?;
    let shape = build_loft(&wires, kind);
    // Tessellates atomically — if this fails (e.g. degenerate solid loft), nothing changes.
    doc.undo_scope("Loft")
        .add_part("Loft".to_owned(), shape, options)
        .context("Failed to tessellate loft")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use duck_engine_scene::Scene;
    use opencascade::primitives::{Face, ShapeType};

    /// A document holding two stacked square profiles (closed wires), returning the
    /// nodes and the index of one edge per profile.
    fn doc_with_two_squares() -> (Document, [LoftProfile; 2]) {
        let scene = Arc::new(Mutex::new(Scene::new()));
        let mut doc = Document::new(scene);

        let square = |y: f64| {
            let wire = Wire::from_ordered_points([
                glam::dvec3(0.0, y, 0.0),
                glam::dvec3(1.0, y, 0.0),
                glam::dvec3(1.0, y, 1.0),
                glam::dvec3(0.0, y, 1.0),
            ])
            .expect("wire builds");
            // Faces so the profile has a wire OCCT can iterate via Shape::wires().
            Shape::from(Face::from_wire(&wire).expect("face builds"))
        };

        let opts = CadTessellationOptions::default();
        let lower = doc.add_part("lower", square(0.0), &opts).expect("tessellates");
        let upper = doc.add_part("upper", square(2.0), &opts).expect("tessellates");
        let lower_node = doc.node_for_part(lower).expect("node");
        let upper_node = doc.node_for_part(upper).expect("node");

        let profiles = [
            LoftProfile { node: lower_node, edge_index: 0 },
            LoftProfile { node: upper_node, edge_index: 0 },
        ];
        (doc, profiles)
    }

    #[test]
    fn surface_loft_produces_a_shell() {
        let (mut doc, profiles) = doc_with_two_squares();
        let before = doc.parts().count();
        execute_loft(&mut doc, &profiles, LoftKind::Surface, &CadTessellationOptions::default())
            .expect("surface loft succeeds");
        assert_eq!(doc.parts().count(), before + 1, "profiles kept, loft added");
        let loft = doc.parts().last().expect("loft part");
        assert_eq!(loft.shape.shape_type(), ShapeType::Shell);
    }

    #[test]
    fn solid_loft_produces_a_solid() {
        let (mut doc, profiles) = doc_with_two_squares();
        execute_loft(&mut doc, &profiles, LoftKind::Solid, &CadTessellationOptions::default())
            .expect("solid loft succeeds");
        let loft = doc.parts().last().expect("loft part");
        assert_eq!(loft.shape.shape_type(), ShapeType::Solid);
    }

    #[test]
    fn single_profile_is_rejected() {
        let (mut doc, profiles) = doc_with_two_squares();
        let one = &profiles[..1];
        let err = execute_loft(&mut doc, one, LoftKind::Surface, &CadTessellationOptions::default());
        assert!(err.is_err(), "a loft needs at least two profiles");
    }
}
