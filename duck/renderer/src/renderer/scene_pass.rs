mod flat_color;
mod main_pass;
mod outline_pass;
mod silhouette;
mod sub_geom_highlight;

pub(crate) use flat_color::{FlatColorPass, FlatColorPassDesc};
pub(crate) use main_pass::{MainPass, OverlayPass};
pub(crate) use outline_pass::outline_passes;
pub(crate) use silhouette::SilhouetteEdgesPass;
pub(crate) use sub_geom_highlight::SubGeomHighlightPass;
