//! Modeling operations on the document's parts: the geometry behind each
//! tool. Nothing here depends on the viewer or the UI.

pub mod boolean;
pub mod delete;
pub mod draft;
pub mod duplicate;
pub mod extrude;
pub mod fillet;
pub mod hollow;
pub mod join;
pub mod loft;
pub mod primitives;
pub mod revolve;
pub mod thicken;

use duck_engine_scene::common::Real;

/// Coincidence tolerance of offsets (hollow, thicken, extrude walls), in
/// document units: 1 µm. An offset no larger than this is null.
pub const OFFSET_TOLERANCE: Real = 1e-3;
