use anyhow::{bail, Context, Result};
use duck_engine_scene::resource::NodeId;
use opencascade::primitives::{Shape, Shell, Wire};

use crate::document::Document;

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

/// The surface skinned through the profiles, to add as a part of its own: a
/// loft is additive, and profiles are usually reused as construction curves.
pub fn build_loft(doc: &Document, profiles: &[LoftProfile]) -> Result<Shape> {
    let wires = resolve_profiles(doc, profiles)?;
    Ok(Shell::loft(&wires).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    use opencascade::primitives::ShapeType;

    use crate::testing::doc_with_two_squares;

    #[test]
    fn surface_loft_produces_a_shell() {
        let (doc, profiles) = doc_with_two_squares();
        let loft = build_loft(&doc, &profiles).expect("surface loft succeeds");
        assert_eq!(loft.shape_type(), ShapeType::Shell);
    }

    #[test]
    fn single_profile_is_rejected() {
        let (doc, profiles) = doc_with_two_squares();
        assert!(build_loft(&doc, &profiles[..1]).is_err(), "a loft needs at least two profiles");
    }
}
