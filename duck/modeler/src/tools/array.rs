//! The array tools: copies of the selected parts laid out by a pattern,
//! previewed until applied.

mod radial;
mod rectangular;

pub use radial::RadialArrayTool;
pub use rectangular::RectangularArrayTool;

use std::sync::{Arc, Mutex};

use anyhow::Result;
use duck_engine_scene::resource::NodeId;
use duck_engine_viewer::{
    common::{decompose_matrix, Matrix4, Point3, Real, Transform},
    selection::{SelectionItem, SelectionManager},
};

use crate::document::{Document, PartId};
use crate::ops::array::ArrayMode;
use crate::preview::PreviewSession;
use super::targets::selected_parts;
use super::transform::parts_frame;
use super::Workspace;

/// Most members a panel field or arrow key gives one direction.
const MAX_COUNT: u32 = 100;

/// Where an array's sources sit.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Spot {
    /// The centre of the sources.
    pivot: Point3,
    /// Half the diagonal of their bounds.
    size: Real,
}

/// Copies of the sources previewed at an array's members. Each draws its
/// source's own mesh, so a change of layout tessellates nothing.
struct ArrayPreview {
    session: PreviewSession,
    document: Arc<Mutex<Document>>,
    /// Part-backed selected nodes, in selection order.
    sources: Vec<NodeId>,
    /// The members the copies stand at, as placements of their sources.
    placements: Vec<Matrix4>,
    /// Each copy, where its source rests, and the index of its member.
    copies: Vec<(NodeId, Transform, usize)>,
}

impl ArrayPreview {
    fn new(workspace: &Workspace) -> Self {
        Self {
            session: workspace.preview_session(),
            document: Arc::clone(&workspace.document),
            sources: Vec::new(),
            placements: Vec::new(),
            copies: Vec::new(),
        }
    }

    /// The copy nodes, to keep out of snapping.
    fn nodes(&self) -> &[NodeId] {
        self.session.preview_nodes()
    }

    /// Takes the parts `selection` holds as the sources. Returns whether they
    /// changed.
    fn follow(&mut self, selection: &SelectionManager) -> bool {
        let sources = selected_parts(selection, &self.document.lock().unwrap());
        if sources == self.sources {
            return false;
        }
        self.set_sources(sources);
        true
    }

    /// Where the sources sit, if there are any.
    fn spot(&self) -> Option<Spot> {
        let document = self.document.lock().unwrap();
        let pivot = parts_frame(&self.sources, None, &document)?.pivot;
        let scene = document.scene().lock();
        let bounds = self.sources.iter().filter_map(|&node| scene.nodes_bounding(node).bounds);
        let size = bounds.reduce(|a, b| a.merge(&b)).map_or(0.0, |bounds| bounds.diagonal() / 2.0);
        Some(Spot { pivot, size })
    }

    /// Copies `sources` from now on, dropping the copies shown so far.
    fn set_sources(&mut self, sources: Vec<NodeId>) {
        self.session.cancel();
        self.copies.clear();
        self.placements.clear();
        self.sources = sources;
    }

    /// Shows a copy of every source at each of `placements`: the copies shown
    /// already move there if there are as many, else they are made afresh.
    fn show(&mut self, placements: Vec<Matrix4>) {
        if placements == self.placements {
            return;
        }
        if placements.len() == self.placements.len() {
            let scene = self.document.lock().unwrap().scene().clone();
            let mut scene = scene.lock();
            for &(copy, rest, member) in &self.copies {
                scene.set_node_transform(copy, placed(rest, &placements[member]));
            }
        } else {
            self.respawn(&placements);
        }
        self.placements = placements;
    }

    fn respawn(&mut self, placements: &[Matrix4]) {
        self.session.cancel();
        self.copies.clear();
        let rests: Vec<(NodeId, Transform)> = {
            let scene = self.document.lock().unwrap().scene().clone();
            let scene = scene.lock();
            self.sources.iter().filter_map(|&source| Some((source, scene.get_node(source)?.transform()))).collect()
        };
        for (source, rest) in rests {
            for (member, placement) in placements.iter().enumerate() {
                match self.session.add_node_copy(source, placed(rest, placement)) {
                    Some(copy) => self.copies.push((copy, rest, member)),
                    None => log::warn!("Node {source:?} has no geometry to copy"),
                }
            }
        }
    }

    /// Copies the sources into the document by `array`, which returns the
    /// copies; then selects the sources and their copies and drops the
    /// preview.
    fn commit(
        &mut self,
        selection: &mut SelectionManager,
        array: impl FnOnce(&mut Document, &[PartId]) -> Result<Vec<PartId>>,
    ) -> Result<()> {
        let members = {
            let mut document = self.document.lock().unwrap();
            let parts: Vec<PartId> = self.sources.iter().filter_map(|&node| document.part_for_node(node)).collect();
            let copies = array(&mut document, &parts)?;
            let copies = copies.iter().filter_map(|&part| document.node_for_part(part));
            self.sources.iter().copied().chain(copies).collect::<Vec<_>>()
        };
        self.set_sources(Vec::new());
        selection.clear();
        selection.extend(members.into_iter().map(SelectionItem::Node));
        Ok(())
    }
}

/// `rest` carried by `placement`. Parts are top-level, so their nodes' own
/// transforms are in world space.
fn placed(rest: Transform, placement: &Matrix4) -> Transform {
    decompose_matrix(&(placement * rest.to_matrix()))
}

/// The Extent / Spacing row of an array's panel grid. Returns the mode picked,
/// when it is a change.
fn mode_field(ui: &mut egui::Ui, mode: ArrayMode) -> Option<ArrayMode> {
    let mut picked = None;
    ui.label("Mode");
    ui.horizontal(|ui| {
        for (option, label) in [(ArrayMode::Extent, "Extent"), (ArrayMode::Spacing, "Spacing")] {
            let selected = mode == option;
            if ui.selectable_label(selected, label).clicked() && !selected {
                picked = Some(option);
            }
        }
    });
    ui.end_row();
    picked
}

/// One labelled member-count row of an array's panel grid. Returns true when
/// the count changed.
fn count_field(ui: &mut egui::Ui, label: &str, count: &mut u32) -> bool {
    ui.label(label);
    let changed = ui.add(egui::DragValue::new(count).speed(0.1).range(1..=MAX_COUNT)).changed();
    ui.end_row();
    changed
}
