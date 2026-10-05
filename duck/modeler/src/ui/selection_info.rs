//! The selection section of the model tab: the primary item and the selection
//! totals, measured from the parts' B-Reps.

use std::collections::HashSet;

use duck_engine_viewer::common::{Aabb, Point3, Real, Vector3};
use duck_engine_viewer::scene::resource::{NodeId, SubGeometryKind};
use duck_engine_viewer::selection::{SelectionItem, SelectionManager};
use opencascade::primitives::{Edge, EdgeType, Face, FaceType, Shape, ShapeType};

use crate::document::{dvec3_to_point3, dvec3_to_vec3, Document, PartKind};
use crate::ui::icons;

/// The selection section, caching its measurements between frames.
#[derive(Default)]
pub struct SelectionInfo {
    cache: Option<Cached>,
}

/// A summary and the document and selection state it was measured from.
struct Cached {
    generation: u64,
    items: Vec<SelectionItem>,
    primary: Option<SelectionItem>,
    summary: SelectionSummary,
}

/// The selection, measured.
#[derive(Default)]
struct SelectionSummary {
    primary: Option<Primary>,
    counts: Counts,
    /// Nodes of the parts selected whole.
    parts: Vec<NodeId>,
    surface_area: Option<Real>,
    volume: Option<Real>,
    edge_length: Option<Real>,
}

/// Selected items by kind.
#[derive(Default, Debug, PartialEq)]
struct Counts {
    parts: usize,
    faces: usize,
    edges: usize,
    vertices: usize,
}

/// The primary selected item.
struct Primary {
    node: NodeId,
    part_name: String,
    part_kind: PartKind,
    element: Element,
}

/// A selected item and its measurements.
#[derive(Debug)]
enum Element {
    Part { faces: usize, edges: usize, vertices: usize, area: Real, volume: Option<Real> },
    Face {
        index: u32,
        ty: FaceType,
        dimensions: Option<Dimensions>,
        normal: Option<Vector3>,
        area: Real,
    },
    Edge { index: u32, ty: EdgeType, dimensions: Option<Dimensions>, length: Real },
    Vertex { index: u32, position: Point3 },
}

/// The defining dimensions of an analytic face or edge.
#[derive(Debug)]
enum Dimensions {
    Radius(Real),
    MajorMinor(Real, Real),
    /// `half_angle` in radians.
    Cone { radius: Real, half_angle: Real },
}

impl SelectionInfo {
    pub fn show(&mut self, ui: &mut egui::Ui, document: &Document, selection: &SelectionManager) {
        let stale = self.cache.as_ref().is_none_or(|cached| {
            cached.generation != document.generation()
                || cached.items != selection.as_slice()
                || cached.primary != selection.primary()
        });
        if stale {
            self.cache = Some(Cached {
                generation: document.generation(),
                items: selection.as_slice().to_vec(),
                primary: selection.primary(),
                summary: summarize(document, selection),
            });
        }
        let summary = &self.cache.as_ref().expect("cache filled above").summary;

        egui::CollapsingHeader::new("Selection")
            .default_open(true)
            .show(ui, |ui| summary_ui(ui, summary, document));
    }
}

fn summary_ui(ui: &mut egui::Ui, summary: &SelectionSummary, document: &Document) {
    if summary.counts.total() == 0 {
        ui.add_space(4.0);
        ui.weak("Nothing selected");
        return;
    }

    if let Some(primary) = &summary.primary {
        primary_ui(ui, primary, document);
    }

    if summary.counts.total() > 1 {
        ui.separator();
        egui::Grid::new("selection_totals").num_columns(2).show(ui, |ui| {
            row(ui, "Selected", format!("{} ({})", summary.counts.total(), summary.counts.describe()));
            if let Some(area) = summary.surface_area {
                row(ui, "Total area", number(area));
            }
            if let Some(volume) = summary.volume {
                row(ui, "Total volume", number(volume));
            }
            if let Some(length) = summary.edge_length {
                row(ui, "Total length", number(length));
            }
            let bounds = summary
                .parts
                .iter()
                .filter_map(|&node| document.scene().nodes_bounding(node).bounds)
                .reduce(|a, b| a.merge(&b));
            if let Some(bounds) = bounds {
                row(ui, "Size", size(&bounds));
            }
        });
    }
}

fn primary_ui(ui: &mut egui::Ui, primary: &Primary, document: &Document) {
    let accent = icons::kind_color(primary.part_kind);
    ui.horizontal(|ui| {
        let (uri, bytes) = icons::kind_icon(primary.part_kind);
        ui.add(
            egui::Image::from_bytes(uri, bytes)
                .fit_to_exact_size(egui::vec2(14.0, 14.0))
                .tint(accent),
        );
        ui.add(egui::Label::new(egui::RichText::new(&primary.part_name).strong()).truncate());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(primary.part_kind.label())
                    .small()
                    .color(accent.gamma_multiply(0.9)),
            );
        });
    });

    egui::Grid::new("selection_primary").num_columns(2).show(ui, |ui| match &primary.element {
        Element::Part { faces, edges, vertices, area, volume } => {
            let topology = [
                count_phrase(*faces, "face", "faces"),
                count_phrase(*edges, "edge", "edges"),
                count_phrase(*vertices, "vertex", "vertices"),
            ];
            row(ui, "Topology", topology.join(" · "));
            row(ui, "Area", number(*area));
            if let Some(volume) = volume {
                row(ui, "Volume", number(*volume));
            }
            if let Some(bounds) = document.scene().nodes_bounding(primary.node).bounds {
                row(ui, "Size", size(&bounds));
            }
        }
        Element::Face { index, ty, dimensions, normal, area } => {
            row(ui, "Element", format!("Face {index} · {}", face_type_name(*ty)));
            dimensions_rows(ui, dimensions.as_ref());
            if let Some(normal) = normal {
                row(ui, "Normal", triple(normal.x, normal.y, normal.z));
            }
            row(ui, "Area", number(*area));
        }
        Element::Edge { index, ty, dimensions, length } => {
            row(ui, "Element", format!("Edge {index} · {}", edge_type_name(*ty)));
            dimensions_rows(ui, dimensions.as_ref());
            row(ui, "Length", number(*length));
        }
        Element::Vertex { index, position } => {
            row(ui, "Element", format!("Vertex {index}"));
            row(ui, "Position", triple(position.x, position.y, position.z));
        }
    });
}

fn dimensions_rows(ui: &mut egui::Ui, dimensions: Option<&Dimensions>) {
    match dimensions {
        Some(Dimensions::Radius(radius)) => row(ui, "Radius", number(*radius)),
        Some(Dimensions::MajorMinor(major, minor)) => {
            row(ui, "Major radius", number(*major));
            row(ui, "Minor radius", number(*minor));
        }
        Some(Dimensions::Cone { radius, half_angle }) => {
            row(ui, "Ref. radius", number(*radius));
            row(ui, "Half-angle", format!("{:.2}°", half_angle.to_degrees()));
        }
        None => {}
    }
}

fn row(ui: &mut egui::Ui, label: &str, value: String) {
    ui.weak(label);
    ui.label(value);
    ui.end_row();
}

fn number(value: Real) -> String {
    format!("{value:.3}")
}

fn triple(x: Real, y: Real, z: Real) -> String {
    format!("({}, {}, {})", number(x), number(y), number(z))
}

fn size(bounds: &Aabb) -> String {
    let extent = bounds.max - bounds.min;
    format!("{} × {} × {}", number(extent.x), number(extent.y), number(extent.z))
}

fn count_phrase(count: usize, singular: &str, plural: &str) -> String {
    format!("{count} {}", if count == 1 { singular } else { plural })
}

fn face_type_name(ty: FaceType) -> &'static str {
    match ty {
        FaceType::Plane => "Planar",
        FaceType::Cylinder => "Cylindrical",
        FaceType::Cone => "Conical",
        FaceType::Sphere => "Spherical",
        FaceType::Torus => "Toroidal",
        FaceType::BezierSurface => "Bézier",
        FaceType::BSplineSurface => "B-spline",
        FaceType::SurfaceOfRevolution => "Revolved",
        FaceType::SurfaceOfExtrusion => "Extruded",
        FaceType::OffsetSurface => "Offset",
        FaceType::OtherSurface => "Other",
    }
}

fn edge_type_name(ty: EdgeType) -> &'static str {
    match ty {
        EdgeType::Line => "Line",
        EdgeType::Circle => "Circle",
        EdgeType::Ellipse => "Ellipse",
        EdgeType::Hyperbola => "Hyperbola",
        EdgeType::Parabola => "Parabola",
        EdgeType::BezierCurve => "Bézier",
        EdgeType::BSplineCurve => "B-spline",
        EdgeType::OffsetCurve => "Offset",
        EdgeType::OtherCurve => "Other",
    }
}

impl Counts {
    fn total(&self) -> usize {
        self.parts + self.faces + self.edges + self.vertices
    }

    /// e.g. "2 parts, 1 face".
    fn describe(&self) -> String {
        [
            (self.parts, "part", "parts"),
            (self.faces, "face", "faces"),
            (self.edges, "edge", "edges"),
            (self.vertices, "vertex", "vertices"),
        ]
        .into_iter()
        .filter(|(count, ..)| *count > 0)
        .map(|(count, singular, plural)| count_phrase(count, singular, plural))
        .collect::<Vec<_>>()
        .join(", ")
    }
}

/// Measure every selected item that resolves to a part. A face of a part that
/// is also selected whole adds no area of its own.
fn summarize(document: &Document, selection: &SelectionManager) -> SelectionSummary {
    let whole: HashSet<NodeId> = selection
        .iter()
        .filter_map(|item| match item {
            SelectionItem::Node(node) => Some(*node),
            SelectionItem::SubGeometry { .. } => None,
        })
        .collect();

    let mut summary = SelectionSummary::default();
    for item in selection.iter() {
        let node = item.node_id();
        let Some(part) = document.part_for_node(node).and_then(|id| document.get_part(id)) else {
            continue;
        };
        let Some(element) = measure(&part.shape, item) else { continue };

        match &element {
            Element::Part { area, volume, .. } => {
                summary.counts.parts += 1;
                summary.parts.push(node);
                accumulate(&mut summary.surface_area, *area);
                if let Some(volume) = volume {
                    accumulate(&mut summary.volume, *volume);
                }
            }
            Element::Face { area, .. } => {
                summary.counts.faces += 1;
                if !whole.contains(&node) {
                    accumulate(&mut summary.surface_area, *area);
                }
            }
            Element::Edge { length, .. } => {
                summary.counts.edges += 1;
                accumulate(&mut summary.edge_length, *length);
            }
            Element::Vertex { .. } => summary.counts.vertices += 1,
        }

        if selection.primary() == Some(*item) {
            summary.primary = Some(Primary {
                node,
                part_name: part.name.clone(),
                part_kind: part.kind(),
                element,
            });
        }
    }
    summary
}

fn accumulate(total: &mut Option<Real>, value: Real) {
    *total = Some(total.unwrap_or(0.0) + value);
}

/// Measure one selected item of `shape`, or `None` if its index is out of range.
fn measure(shape: &Shape, item: &SelectionItem) -> Option<Element> {
    let SelectionItem::SubGeometry { element, .. } = item else {
        return Some(part_element(shape));
    };
    let index = element.index;
    Some(match element.kind {
        SubGeometryKind::Face => face_element(index, &shape.face_at(index as usize)?),
        SubGeometryKind::Edge => edge_element(index, &shape.edge_at(index as usize)?),
        SubGeometryKind::Pointset => Element::Vertex {
            index,
            position: dvec3_to_point3(shape.vertex_at(index as usize)?.point()),
        },
    })
}

fn part_element(shape: &Shape) -> Element {
    Element::Part {
        faces: shape.unique_sub_shape_count(ShapeType::Face),
        edges: shape.unique_sub_shape_count(ShapeType::Edge),
        vertices: shape.unique_sub_shape_count(ShapeType::Vertex),
        area: shape.surface_area() as Real,
        volume: shape.contains_type(ShapeType::Solid).then(|| shape.volume() as Real),
    }
}

fn face_element(index: u32, face: &Face) -> Element {
    let ty = face.face_type();
    let dimensions = match ty {
        FaceType::Cylinder => face.cylinder_radius().map(|r| Dimensions::Radius(r as Real)),
        FaceType::Sphere => face.sphere_radius().map(|r| Dimensions::Radius(r as Real)),
        FaceType::Cone => face.cone_dimensions().map(|(radius, half_angle)| Dimensions::Cone {
            radius: radius as Real,
            half_angle: half_angle.abs() as Real,
        }),
        FaceType::Torus => face
            .torus_radii()
            .map(|(major, minor)| Dimensions::MajorMinor(major as Real, minor as Real)),
        _ => None,
    };
    let normal = match ty {
        FaceType::Plane => face.normal_at_center().ok().map(dvec3_to_vec3),
        _ => None,
    };
    Element::Face { index, ty, dimensions, normal, area: face.surface_area() as Real }
}

fn edge_element(index: u32, edge: &Edge) -> Element {
    let ty = edge.edge_type();
    let dimensions = match ty {
        EdgeType::Circle => edge.circle_radius().map(|r| Dimensions::Radius(r as Real)),
        EdgeType::Ellipse => edge
            .ellipse_radii()
            .map(|(major, minor)| Dimensions::MajorMinor(major as Real, minor as Real)),
        _ => None,
    };
    Element::Edge { index, ty, dimensions, length: edge.length() as Real }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::cad::CadTessellationOptions;
    use duck_engine_scene::resource::SubGeometryElement;
    use duck_engine_scene::Scene;

    const EPSILON: Real = 1e-6;

    /// A document with one part per shape, and the parts' nodes.
    fn document_with(shapes: impl IntoIterator<Item = Shape>) -> (Document, Vec<NodeId>) {
        let mut document = Document::new(Scene::default());
        let options = CadTessellationOptions::default();
        let nodes = shapes
            .into_iter()
            .enumerate()
            .map(|(i, shape)| {
                let part = document.add_part(format!("part-{i}"), shape, &options).unwrap();
                document.node_for_part(part).unwrap()
            })
            .collect();
        (document, nodes)
    }

    fn sub_item(node: NodeId, kind: SubGeometryKind, index: usize) -> SelectionItem {
        SelectionItem::SubGeometry {
            node_id: node,
            element: SubGeometryElement::new(kind, index as u32),
        }
    }

    fn selecting(items: impl IntoIterator<Item = SelectionItem>) -> SelectionManager {
        let mut selection = SelectionManager::new();
        selection.extend(items);
        selection
    }

    fn assert_close(actual: Option<Real>, expected: Real) {
        let actual = actual.expect("a value");
        assert!((actual - expected).abs() < EPSILON, "expected {expected}, got {actual}");
    }

    #[test]
    fn a_part_reports_its_topology_area_and_volume() {
        let (document, nodes) = document_with([Shape::cube(2.0)]);
        let summary = summarize(&document, &selecting([SelectionItem::Node(nodes[0])]));

        let primary = summary.primary.expect("a primary");
        assert_eq!(primary.part_name, "part-0");
        let Element::Part { faces, edges, vertices, area, volume } = primary.element else {
            panic!("expected a part, got {:?}", primary.element);
        };
        assert_eq!((faces, edges, vertices), (6, 12, 8));
        assert_close(Some(area), 24.0);
        assert_close(volume, 8.0);
    }

    #[test]
    fn a_cylinder_side_face_reports_its_radius() {
        let cylinder = Shape::cylinder_radius_height(5.0, 3.0);
        let side = cylinder.faces().position(|f| f.face_type() == FaceType::Cylinder).unwrap();
        let (document, nodes) = document_with([cylinder]);

        let summary =
            summarize(&document, &selecting([sub_item(nodes[0], SubGeometryKind::Face, side)]));

        let element = summary.primary.expect("a primary").element;
        let Element::Face { index, ty, dimensions, normal, .. } = element else {
            panic!("expected a face, got {element:?}");
        };
        assert_eq!((index as usize, ty), (side, FaceType::Cylinder));
        let Some(Dimensions::Radius(radius)) = dimensions else {
            panic!("expected a radius, got {dimensions:?}");
        };
        assert_close(Some(radius), 5.0);
        assert!(normal.is_none(), "only planar faces report a normal");
    }

    #[test]
    fn a_circular_edge_reports_its_radius_and_length() {
        let cylinder = Shape::cylinder_radius_height(2.0, 3.0);
        let rim = cylinder.edges().position(|e| e.edge_type() == EdgeType::Circle).unwrap();
        let (document, nodes) = document_with([cylinder]);

        let summary =
            summarize(&document, &selecting([sub_item(nodes[0], SubGeometryKind::Edge, rim)]));

        let element = summary.primary.expect("a primary").element;
        let Element::Edge { ty, dimensions, length, .. } = element else {
            panic!("expected an edge, got {element:?}");
        };
        assert_eq!(ty, EdgeType::Circle);
        let Some(Dimensions::Radius(radius)) = dimensions else {
            panic!("expected a radius, got {dimensions:?}");
        };
        assert_close(Some(radius), 2.0);
        assert_close(Some(length), 4.0 * std::f64::consts::PI as Real);
    }

    #[test]
    fn totals_sum_across_parts() {
        let (document, nodes) = document_with([Shape::cube(2.0), Shape::cube(1.0)]);
        let summary = summarize(&document, &selecting(nodes.iter().map(|&n| SelectionItem::Node(n))));

        assert_eq!(summary.counts, Counts { parts: 2, ..Default::default() });
        assert_close(summary.surface_area, 24.0 + 6.0);
        assert_close(summary.volume, 8.0 + 1.0);
        assert!(summary.edge_length.is_none(), "no edges selected");
    }

    #[test]
    fn a_face_of_a_whole_part_adds_no_area() {
        let (document, nodes) = document_with([Shape::cube(2.0), Shape::cube(1.0)]);
        let summary = summarize(
            &document,
            &selecting([
                SelectionItem::Node(nodes[0]),
                sub_item(nodes[0], SubGeometryKind::Face, 0),
                sub_item(nodes[1], SubGeometryKind::Face, 0),
            ]),
        );

        assert_eq!(summary.counts, Counts { parts: 1, faces: 2, ..Default::default() });
        assert_close(summary.surface_area, 24.0 + 1.0);
        assert_eq!(summary.counts.describe(), "1 part, 2 faces");
    }

    #[test]
    fn items_off_the_document_are_skipped() {
        let (mut document, nodes) = document_with([Shape::cube(2.0)]);
        let part = document.part_for_node(nodes[0]).unwrap();
        document.remove_part(part);

        let summary = summarize(&document, &selecting([SelectionItem::Node(nodes[0])]));
        assert_eq!(summary.counts.total(), 0);
        assert!(summary.primary.is_none());
    }
}
