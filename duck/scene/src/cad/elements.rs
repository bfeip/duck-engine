use opencascade::primitives::{AncestorMap, Edge, Face, Shape, ShapeMap, SubShape, Vertex, Wire};

/// A shape's faces, edges and vertices as a user sees them, numbered: the order
/// a tessellated mesh's sub-geometry ranges follow, so a picked index is an
/// element.
///
/// Each kind is numbered from 0 in the order OCCT first meets them exploring the
/// shape. Seams, degenerate edges and the vertices only they need are left out,
/// and the edges a seam split out of one curve are one [`EdgeElement`].
pub struct ShapeElements {
    shape: Shape,
    face_map: ShapeMap<Face>,
    faces: Vec<Face>,
    edges: Vec<EdgeElement>,
    vertices: Vec<Vertex>,
    seams: Vec<Edge>,
    edge_lookup: Lookup<Edge>,
    vertex_lookup: Lookup<Vertex>,
}

/// An edge as a user sees it: one of the shape's edges, or several where a seam
/// split one curve.
pub struct EdgeElement {
    pieces: Vec<Edge>,
    /// The pieces joined into one, when there are several.
    joined: Option<Edge>,
}

impl EdgeElement {
    /// The shape's edges this element covers, in order along its curve: one, or
    /// more where a seam split it.
    pub fn pieces(&self) -> &[Edge] {
        &self.pieces
    }

    /// This element as one edge: the shape's own edge, or a new edge joining the
    /// pieces that is not part of the shape.
    pub fn curve(&self) -> &Edge {
        self.joined.as_ref().unwrap_or(&self.pieces[0])
    }
}

impl ShapeElements {
    pub fn new(shape: &Shape) -> Self {
        let mut splits = SeamSplits::new(shape);

        let mut edges = Vec::new();
        for run in std::mem::take(&mut splits.chains) {
            let pieces: Vec<Edge> = run.iter().map(|end| splits.edge(end.edge)).collect();
            if pieces.len() == 1 {
                edges.push(EdgeElement { pieces, joined: None });
                continue;
            }
            match join(&pieces) {
                Some(joined) => edges.push(EdgeElement { pieces, joined: Some(joined) }),
                // Pieces that make no one edge are edges of their own, meeting at
                // vertices of their own.
                None => {
                    for end in &run {
                        splits.reveal(end.other());
                    }
                    edges.extend(
                        pieces.into_iter().map(|piece| EdgeElement { pieces: vec![piece], joined: None }),
                    );
                },
            }
        }

        let vertices: Vec<Vertex> =
            (0..splits.hidden.len()).filter(|&v| !splits.hidden[v]).map(|v| splits.vertex(v)).collect();
        let seams = (0..splits.seams.len())
            .filter(|&e| splits.seams[e] && !splits.degenerate[e])
            .map(|e| splits.edge(e))
            .collect();

        let face_map = ShapeMap::new(shape);
        let faces = face_map.iter().collect();
        let mut edge_lookup = Lookup::new(shape);
        for (index, element) in edges.iter().enumerate() {
            for piece in &element.pieces {
                edge_lookup.set(piece, index);
            }
        }
        let mut vertex_lookup = Lookup::new(shape);
        for (index, vertex) in vertices.iter().enumerate() {
            vertex_lookup.set(vertex, index);
        }
        Self { shape: shape.clone(), face_map, faces, edges, vertices, seams, edge_lookup, vertex_lookup }
    }

    /// The shape these are the elements of.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn faces(&self) -> &[Face] {
        &self.faces
    }

    pub fn edges(&self) -> &[EdgeElement] {
        &self.edges
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// The shape's seam edges: no elements, but drawn to debug a surface's
    /// parametrization.
    pub fn seams(&self) -> &[Edge] {
        &self.seams
    }

    /// The element that is `face` of the shape.
    pub fn face_index(&self, face: &Face) -> Option<usize> {
        self.face_map.index_of(face)
    }

    /// The element that covers `edge` of the shape.
    pub fn edge_index(&self, edge: &Edge) -> Option<usize> {
        self.edge_lookup.get(edge)
    }

    /// The element that is `vertex` of the shape.
    pub fn vertex_index(&self, vertex: &Vertex) -> Option<usize> {
        self.vertex_lookup.get(vertex)
    }
}

/// `pieces`, in order along one curve, as one edge; `None` if they make no single
/// edge.
fn join(pieces: &[Edge]) -> Option<Edge> {
    let wire = Wire::from_edges(pieces).ok()?;
    let joined = Shape::from(wire).clean().ok()?;
    let mut edges = joined.edges();
    match (edges.next(), edges.next()) {
        (Some(edge), None) => Some(edge),
        _ => None,
    }
}

/// Which element each of a shape's sub-shapes of one kind is, or belongs to.
struct Lookup<T: SubShape> {
    map: ShapeMap<T>,
    elements: Vec<Option<usize>>,
}

impl<T: SubShape> Lookup<T> {
    fn new(shape: &Shape) -> Self {
        let map = ShapeMap::new(shape);
        let elements = vec![None; map.len()];
        Self { map, elements }
    }

    fn set(&mut self, sub_shape: &T, element: usize) {
        if let Some(slot) = self.map.index_of(sub_shape) {
            self.elements[slot] = Some(element);
        }
    }

    fn get(&self, sub_shape: &T) -> Option<usize> {
        self.elements[self.map.index_of(sub_shape)?]
    }
}

/// One end of an edge of a [`SeamSplits`], by the edge's index.
#[derive(Clone, Copy)]
struct EdgeEnd {
    edge: usize,
    /// The end at the curve's first parameter, rather than its last.
    start: bool,
}

impl EdgeEnd {
    /// Position of this end in a per-end table.
    fn slot(self) -> usize {
        2 * self.edge + usize::from(!self.start)
    }

    fn other(self) -> Self {
        Self { start: !self.start, ..self }
    }
}

/// A shape's edges and vertices as they read with its seams hidden: the
/// vertices that only parametrize a surface, and the runs of edges a seam split
/// out of one curve. Edges and vertices are numbered as their maps number them.
struct SeamSplits {
    edges: ShapeMap<Edge>,
    /// Each vertex, with the edges it bounds.
    vertex_edges: AncestorMap<Vertex, Edge>,
    /// Per edge: it is a seam of some face.
    seams: Vec<bool>,
    /// Per edge: it has no 3D curve.
    degenerate: Vec<bool>,
    /// Per vertex: it is a seam vertex.
    hidden: Vec<bool>,
    /// Runs of edges joined through seam vertices, each in order along its
    /// curve, entered by the listed end. Every edge that is neither a seam nor
    /// degenerate is in exactly one run.
    chains: Vec<Vec<EdgeEnd>>,
    /// Per edge: the run it belongs to.
    chain_of: Vec<Option<usize>>,
}

impl SeamSplits {
    /// Edges leaving a vertex within this angle (radians) of opposite directions
    /// carry one curve through it.
    const SMOOTH_ANGLE: f64 = 1e-3;

    fn new(shape: &Shape) -> Self {
        let edges = ShapeMap::<Edge>::new(shape);
        let vertex_edges = AncestorMap::<Vertex, Edge>::new(shape);
        let (edge_count, vertex_count) = (edges.len(), vertex_edges.len());
        let degenerate = edges.iter().map(|edge| edge.is_degenerated()).collect();

        let mut splits = Self {
            edges,
            vertex_edges,
            seams: vec![false; edge_count],
            degenerate,
            hidden: vec![false; vertex_count],
            chains: Vec::new(),
            chain_of: vec![None; edge_count],
        };
        for face in shape.faces() {
            for edge in face.edges() {
                if let Some(index) = splits.edges.index_of(&edge).filter(|_| edge.is_seam(&face)) {
                    splits.seams[index] = true;
                }
            }
        }
        let joins = splits.find_seam_vertices();
        splits.chain(&joins);
        splits
    }

    fn edge(&self, index: usize) -> Edge {
        self.edges.get(index).expect("an edge of the shape")
    }

    fn vertex(&self, index: usize) -> Vertex {
        self.vertex_edges.get(index).expect("a vertex of the shape")
    }

    /// Whether the edge only parametrizes its face.
    fn parametric(&self, index: usize) -> bool {
        self.seams[index] || self.degenerate[index]
    }

    /// Marks the seam vertices, and returns, per [`EdgeEnd::slot`], the edge end
    /// that continues each end through one.
    fn find_seam_vertices(&mut self) -> Vec<Option<EdgeEnd>> {
        let mut joins = vec![None; 2 * self.seams.len()];
        for v in 0..self.hidden.len() {
            let vertex = self.vertex(v);
            let mut parametric = false;
            let mut ends = Vec::new();
            for edge in self.vertex_edges.ancestors(v).into_iter().flatten() {
                let Some(index) = self.edges.index_of(&edge) else { continue };
                if self.parametric(index) {
                    parametric = true;
                    continue;
                }
                // A closed edge has both its ends here.
                if edge.start_vertex().is_same(&vertex) {
                    ends.push(EdgeEnd { edge: index, start: true });
                }
                if edge.end_vertex().is_same(&vertex) {
                    ends.push(EdgeEnd { edge: index, start: false });
                }
            }

            self.hidden[v] = parametric
                && match ends[..] {
                    [] => true,
                    [a, b] => self.continues(a, b),
                    _ => false,
                };
            if let (&[a, b], true) = (ends.as_slice(), self.hidden[v]) {
                joins[a.slot()] = Some(b);
                joins[b.slot()] = Some(a);
            }
        }
        joins
    }

    /// Whether the edges leaving a vertex by `a` and `b` carry one smooth curve
    /// straight through it.
    fn continues(&self, a: EdgeEnd, b: EdgeEnd) -> bool {
        let away = |end: EdgeEnd| {
            let edge = self.edge(end.edge);
            if end.start {
                edge.start_tangent()
            } else {
                -edge.end_tangent()
            }
        };
        away(a).dot(away(b)) < -Self::SMOOTH_ANGLE.cos()
    }

    /// Groups the edges that are neither seams nor degenerate into runs along
    /// `joins`.
    fn chain(&mut self, joins: &[Option<EdgeEnd>]) {
        for edge in 0..self.seams.len() {
            if self.parametric(edge) || self.chain_of[edge].is_some() {
                continue;
            }
            // Back up to where the run begins; a closed run begins anywhere.
            let mut entry = EdgeEnd { edge, start: true };
            while let Some(previous) = joins[entry.slot()].filter(|previous| previous.edge != edge)
            {
                entry = previous.other();
            }

            let chain = self.chains.len();
            let mut run = Vec::new();
            loop {
                self.chain_of[entry.edge] = Some(chain);
                run.push(entry);
                match joins[entry.other().slot()] {
                    Some(next) if self.chain_of[next.edge].is_none() => entry = next,
                    _ => break,
                }
            }
            self.chains.push(run);
        }
    }

    /// Shows the vertex at `end` again.
    fn reveal(&mut self, end: EdgeEnd) {
        let edge = self.edge(end.edge);
        let vertex = match end.start {
            true => edge.start_vertex(),
            false => edge.end_vertex(),
        };
        if let Some(v) = self.vertex_edges.index_of(&vertex) {
            self.hidden[v] = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{dvec3, DVec3};
    use opencascade::primitives::EdgeType;

    /// A ball with a bite out of its side, where the seam meridian runs through
    /// the bite: the seam ends on the bite's rim twice, splitting the rim circle.
    fn ball_bitten_across_its_seam() -> Shape {
        let ball = Shape::sphere(2.0).build();
        let bite = Shape::sphere(1.0).at(dvec3(2.0, 0.0, 0.3)).build();
        ball.subtract(&bite).unwrap().shape.clean().unwrap()
    }

    fn polyline_length(points: &[DVec3]) -> f64 {
        points.windows(2).map(|pair| pair[0].distance(pair[1])).sum()
    }

    fn counts(elements: &ShapeElements) -> (usize, usize, usize, usize) {
        (elements.faces().len(), elements.edges().len(), elements.vertices().len(), elements.seams().len())
    }

    #[test]
    fn a_seam_split_rim_is_one_edge() {
        let elements = ShapeElements::new(&ball_bitten_across_its_seam());
        // Every vertex is a pole or a seam crossing; the seam is in two pieces.
        assert_eq!(counts(&elements), (2, 1, 0, 2));

        let rim = &elements.edges()[0];
        assert_eq!(rim.pieces().len(), 2, "the seam splits the rim in two");
        for piece in rim.pieces() {
            assert_eq!(elements.edge_index(piece), Some(0));
        }
        for seam in elements.seams() {
            assert_eq!(elements.edge_index(seam), None, "a seam is no element");
        }
        let length: f64 = rim.pieces().iter().map(Edge::length).sum();
        assert_eq!(rim.curve().edge_type(), EdgeType::Circle);
        assert!(
            (rim.curve().length() - length).abs() < 1e-6,
            "expected {length}, got {}",
            rim.curve().length()
        );
    }

    #[test]
    fn a_seam_split_rim_draws_as_one_closed_loop() {
        let elements = ShapeElements::new(&ball_bitten_across_its_seam());
        let rim = &elements.edges()[0];
        let circumference: f64 = rim.pieces().iter().map(Edge::length).sum();

        let points = rim.curve().polyline(0.1);
        assert!(points.first().unwrap().distance(*points.last().unwrap()) < 1e-7, "the loop is open");
        assert!((polyline_length(&points) - circumference).abs() < 0.01 * circumference);
    }

    #[test]
    fn a_sphere_is_one_face() {
        let elements = ShapeElements::new(&Shape::sphere(1.0).build());
        assert_eq!(counts(&elements), (1, 0, 0, 1), "the poles and the seam are no elements");
    }

    /// A cylinder's rims start where its seam meets them: those vertices are no
    /// elements, and each rim is one edge though both of its faces list it.
    #[test]
    fn a_cylinder_is_three_faces_and_two_rims() {
        let elements = ShapeElements::new(&Shape::cylinder_radius_height(1.0, 2.0));
        assert_eq!(counts(&elements), (3, 2, 0, 1));
    }

    #[test]
    fn a_box_numbers_each_face_edge_and_corner_once() {
        let cube = Shape::cube(1.0);
        let elements = ShapeElements::new(&cube);
        assert_eq!(counts(&elements), (6, 12, 8, 0));

        for (index, element) in elements.edges().iter().enumerate() {
            let [piece] = element.pieces() else { panic!("edge {index} is split") };
            assert!(element.curve().is_same(piece), "edge {index} is its own curve");
        }
        // Every occurrence of an edge finds its one element.
        for edge in cube.edges() {
            let index = elements.edge_index(&edge).expect("every box edge is an element");
            assert!(elements.edges()[index].pieces()[0].is_same(&edge));
        }
        for (index, face) in elements.faces().iter().enumerate() {
            assert_eq!(elements.face_index(face), Some(index));
        }
        for (index, vertex) in elements.vertices().iter().enumerate() {
            assert_eq!(elements.vertex_index(vertex), Some(index));
        }
    }

    /// A V-shaped notch cut across a cylinder's top, its point on the seam: the
    /// notch's sides meet there at a corner, which stays a vertex.
    #[test]
    fn a_corner_on_a_seam_stays_a_vertex() {
        let cylinder = Shape::cylinder_radius_height(1.0, 2.0);
        let v = Wire::from_ordered_points([
            dvec3(-2.0, 0.0, 1.5),
            dvec3(-2.0, 2.0, 3.5),
            dvec3(-2.0, -2.0, 3.5),
        ])
        .unwrap();
        let notch = Shape::from(Face::from_wire(&v).unwrap().extrude(dvec3(4.0, 0.0, 0.0)));
        let elements = ShapeElements::new(&cylinder.subtract(&notch).unwrap().shape);

        // The seam runs along +X, so the notch's point on the cylinder's surface
        // there is on the seam.
        let point = dvec3(1.0, 0.0, 1.5);
        let ends_at_point = |s: &Edge| s.start_point().distance(point).min(s.end_point().distance(point)) < 1e-7;
        assert!(elements.seams().iter().any(ends_at_point), "the seam misses the notch");
        assert!(
            elements.vertices().iter().any(|v| v.point().distance(point) < 1e-7),
            "the notch's point is hidden"
        );
        for element in elements.edges() {
            assert_eq!(element.pieces().len(), 1, "nothing on the notch was split");
        }
    }

    #[test]
    fn arcs_of_one_circle_join_into_it() {
        let halves = [
            Edge::arc(dvec3(1.0, 0.0, 0.0), dvec3(0.0, 1.0, 0.0), dvec3(-1.0, 0.0, 0.0)).unwrap(),
            Edge::arc(dvec3(-1.0, 0.0, 0.0), dvec3(0.0, -1.0, 0.0), dvec3(1.0, 0.0, 0.0)).unwrap(),
        ];
        let circle = join(&halves).expect("the halves join");
        assert_eq!(circle.edge_type(), EdgeType::Circle);
        assert!((circle.length() - 2.0 * std::f64::consts::PI).abs() < 1e-9);
    }

    /// A line running smoothly into an arc is two curves: they don't join.
    #[test]
    fn a_line_and_a_tangent_arc_do_not_join() {
        let line = Edge::segment(dvec3(-1.0, 0.0, 0.0), DVec3::ZERO).unwrap();
        let arc = Edge::arc(DVec3::ZERO, dvec3(1.0, 1.0, 0.0), dvec3(0.0, 2.0, 0.0)).unwrap();
        assert!(join(&[line, arc]).is_none());
    }
}
