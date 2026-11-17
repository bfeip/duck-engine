//! Snapping system: operators resolve a cursor ray to a meaningful world
//! position (and possibly orientation) by consulting a set of pluggable
//! [`SnapProvider`]s.
//!
//! # Design
//!
//! A provider is a pure strategy: given a [`SnapInput`] (the cursor in its view
//! plus the active construction context) and the [`Document`] (its scene, and the B-rep
//! behind each part), it proposes candidate [`Snap`]s. The [`SnapEngine`] runs
//! every enabled/requested provider, then **ranks** the candidates by tier
//! first, then screen-space distance to the cursor, returning the single best
//! [`Snap`].
//!
//! While a tool draws a [`Stroke`], the edges near the cursor and the face
//! under it also relate to the stroke's last point: the point closest to it,
//! the foot of a perpendicular from it. Such a relation is a snap like any
//! other, taken only with the cursor within the pixel tolerance of it, and
//! ranks one tier above the element it lies on.
//!
//! The construction plane is an always-present, lowest-tier fallback, so an
//! operator constrained to that plane always gets a position — but any
//! higher-tier snap (grid axis, corner, ...) wins whenever one is within the
//! pixel tolerance.
//!
//! Providers measure nothing themselves. What the scene shows near the cursor
//! comes from [`near_cursor`](crate::geometry::near_cursor), and the points of
//! a part's B-rep from [`PartEdge`](crate::geometry::PartEdge) and
//! [`PartFace`](crate::geometry::PartFace); a provider chooses which of them
//! to offer, as which kind.
//!
//! # Extending
//!
//! To add a snap: add a [`SnapKind`] variant (plus its one-line [`Snap::tier`]
//! and [`SnapKind::flag`] arms and a [`SnapFlags`] bit), implement
//! [`SnapProvider`], and register it in [`SnapEngine::with_defaults`] (or via
//! [`SnapEngine::add_provider`]). A provider is asked only for the kinds in
//! [`SnapInput::requested`]: those the caller wants that are enabled. A snap
//! relating the next point to the points placed before it reads
//! [`SnapInput::stroke`], and proposes nothing without one. [`Snap::direction`]
//! already carries an optional tangent/axis for line snaps and future
//! orientation snaps.

mod providers;
mod stroke;

pub use stroke::Stroke;

use bitflags::bitflags;

use duck_engine_viewer::common::{Axis, InnerSpace, Plane, Point3, Ray, Real, Vector3};
use duck_engine_viewer::scene::PositionedCamera;
use duck_engine_viewer::scene::resource::{NodeId, SubGeometryKind};

use crate::document::Document;
use crate::geometry::PartElement;
use crate::grid::GridConfig;

/// What a single snap locks onto. Maps to a ranking [`SnapTier`] and a
/// [`SnapFlags`] flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SnapKind {
    /// Free point on the active construction plane (always-present fallback).
    ConstructionPlane,
    /// The construction-frame origin (0, 0, 0).
    Origin,
    /// A gridline intersection on the construction plane.
    GridGuide,
    /// A construction-frame principal axis (U, V, or the plane normal N).
    GridAxis,
    /// A B-rep corner / vertex of existing geometry.
    Corner,
    /// A point on an existing geometry edge/wire (smooth closest-point snap).
    Edge,
    /// The point halfway along an existing geometry edge, from its B-rep curve.
    Midpoint,
    /// The centre of the flat face under the cursor, from its B-rep.
    FaceCenter,
    /// The centre of a circular edge (a circle or an arc of one), from its
    /// B-rep, along the circle's axis.
    Center,
    /// A point on an existing geometry face (ray–surface hit under the cursor).
    Face,
    /// The start point of an in-progress wire, offered so the path can be
    /// closed back onto itself.
    WireStart,
    /// A construction-frame axis (U, V or N) through the stroke's last point.
    AxisAligned,
    /// A line through the stroke's last point, parallel to one of its earlier
    /// segments or to the straight edge it last snapped onto.
    Parallel,
    /// The point of the hovered edge or face nearest the stroke's last point.
    Closest,
    /// A point of the hovered edge or face that a segment from the stroke's
    /// last point meets at a right angle.
    Perpendicular,
    /// The point where a segment from the circle the stroke's last point lies
    /// on touches the hovered circle, tangent to both. Taking it moves the last
    /// point to where the segment leaves its circle.
    TangentTangent,
}

impl SnapKind {
    fn flag(self) -> SnapFlags {
        use SnapKind::*;
        match self {
            ConstructionPlane => SnapFlags::CONSTRUCTION_PLANE,
            Origin => SnapFlags::ORIGIN,
            GridGuide => SnapFlags::GRID_GUIDE,
            GridAxis => SnapFlags::GRID_AXIS,
            Corner => SnapFlags::CORNER,
            Edge => SnapFlags::EDGE,
            Midpoint => SnapFlags::MIDPOINT,
            FaceCenter => SnapFlags::FACE_CENTER,
            Center => SnapFlags::CENTER,
            Face => SnapFlags::FACE,
            WireStart => SnapFlags::WIRE_START,
            AxisAligned => SnapFlags::AXIS_ALIGNED,
            Parallel => SnapFlags::PARALLEL,
            Closest => SnapFlags::CLOSEST,
            Perpendicular => SnapFlags::PERPENDICULAR,
            TangentTangent => SnapFlags::TANGENT_TANGENT,
        }
    }
}

/// How far a direction may turn off a world axis, as `1 - cos θ` (about
/// 0.08°), and still lie along it.
const AXIS_TOLERANCE: Real = 1e-6;

/// Ranking precedence, *defined by declaration order* (low → high): a higher
/// tier always beats a lower one when both are within tolerance; ties break on
/// screen distance, or on depth where they coincide on screen. Reordering
/// these variants is the single place snap precedence is expressed.
///
/// (Rust derives [`Ord`] on a field-less enum by discriminant, i.e. declaration
/// order, so `Free < Guide < Axis < ... < Keypoint`.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SnapTier {
    /// Free placement on the construction plane.
    Free,
    /// A grid guide point (low-value hint).
    Guide,
    /// A line through a construction-frame or stroke point.
    Axis,
    /// A point on a face, wherever the cursor is over it. Real geometry beats
    /// the grid, but any edge or keypoint within tolerance still wins.
    Face,
    /// A point of a face defined by the stroke's last point, such as the foot
    /// of the perpendicular from that point to the face. It beats the face it
    /// lies on, but an edge within tolerance still wins.
    FaceRelation,
    /// A point on an edge or wire, wherever the cursor is along it. Real
    /// geometry beats the grid, but a keypoint within tolerance still wins.
    Edge,
    /// A point of an edge defined by the stroke's last point, such as the foot
    /// of the perpendicular from that point to the edge. It beats the edge it
    /// lies on, but a keypoint within tolerance still wins.
    EdgeRelation,
    /// A single definite point: a corner, the middle of an edge, the centre of
    /// a circle or a flat face, the frame origin, the stroke's first point.
    Keypoint,
}

bitflags! {
    /// A *set* of snap kinds: which the engine has enabled, which an operator
    /// requests, and which a provider can emit.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SnapFlags: u32 {
        const CONSTRUCTION_PLANE = 1 << 0;
        const ORIGIN             = 1 << 1;
        const GRID_GUIDE         = 1 << 2;
        const GRID_AXIS          = 1 << 3;
        const CORNER             = 1 << 4;
        const EDGE               = 1 << 5;
        const WIRE_START         = 1 << 6;
        const FACE               = 1 << 7;
        const MIDPOINT           = 1 << 8;
        const FACE_CENTER        = 1 << 9;
        const CENTER             = 1 << 10;
        const AXIS_ALIGNED       = 1 << 11;
        const PARALLEL           = 1 << 12;
        const CLOSEST            = 1 << 13;
        const PERPENDICULAR      = 1 << 14;
        const TANGENT_TANGENT    = 1 << 15;
    }
}

/// Per-call inputs describing where the user is pointing and the active
/// construction context. Cheap to build each cursor move; everything is borrowed
/// so changing the construction plane at runtime needs no provider rebuild.
pub struct SnapInput<'a> {
    /// Cursor position in screen pixels.
    pub cursor: (f32, f32),
    /// Viewport size in pixels (width, height).
    pub viewport: (u32, u32),
    /// Active camera, used to project candidates back to screen for ranking.
    pub camera: &'a PositionedCamera,
    /// Active construction plane: the fallback surface and the grid/frame plane.
    pub plane: &'a Plane,
    /// Grid configuration (spacing, extent) for grid snapping.
    pub grid: &'a GridConfig,
    /// Which kinds the calling operator wants. A provider sees those of them
    /// the engine has enabled.
    pub requested: SnapFlags,
    /// Nodes to ignore (e.g. the operator's own in-progress preview geometry).
    pub exclude_nodes: &'a [NodeId],
    /// The path the calling tool is drawing, for the snaps that relate the
    /// next point to those placed before it; `None` if it draws none.
    pub stroke: Option<&'a Stroke>,
}

impl<'a> SnapInput<'a> {
    /// The input for `cursor` in a `viewport`-sized view through `camera`,
    /// constructing on `plane` with `grid`: every kind requested, no node
    /// ignored, no stroke.
    pub fn new(
        camera: &'a PositionedCamera,
        viewport: (u32, u32),
        cursor: (f32, f32),
        plane: &'a Plane,
        grid: &'a GridConfig,
    ) -> Self {
        Self { cursor, viewport, camera, plane, grid, requested: SnapFlags::all(), exclude_nodes: &[], stroke: None }
    }

    /// The camera ray through the cursor, in world space.
    pub fn ray(&self) -> Ray {
        let (x, y) = self.cursor;
        let (width, height) = self.viewport;
        self.camera.ray_from_screen_point(x, y, width, height)
    }
}

/// A snap location: where the cursor snaps to, and what it locks onto there.
#[derive(Debug, Clone, Copy)]
pub struct Snap {
    /// World-space position to snap to.
    pub position: Point3,
    /// Optional associated direction: the tangent for line snaps (edges, grid
    /// axes) and the surface normal for face snaps.
    pub direction: Option<Vector3>,
    /// What this snap locks onto (drives tier + filtering).
    pub kind: SnapKind,
    /// The part element the snap lies on, where it lies on one.
    pub on: Option<PartElement>,
    /// Where the stroke's last point moves to when this snap is taken.
    pub moves_last: Option<Point3>,
}

impl Snap {
    /// A `kind` snap at `position`, with no direction, on no part element.
    pub fn new(position: Point3, kind: SnapKind) -> Self {
        Self { position, direction: None, kind, on: None, moves_last: None }
    }

    pub fn with_direction(self, direction: Vector3) -> Self {
        Self { direction: Some(direction), ..self }
    }

    pub fn with_on(self, on: PartElement) -> Self {
        Self { on: Some(on), ..self }
    }

    /// What the snap locks onto, in a few words, as the pointer tags it: the
    /// axis an axis snap runs along, the world plane a free point lies on, and
    /// what a relation relates to. `construction` is the construction plane.
    pub fn describe(&self, construction: &Plane) -> String {
        use SnapKind::*;
        let axis = self.direction.and_then(|direction| Axis::along(direction, AXIS_TOLERANCE));
        let on = self.on.map(|on| match on.element.kind {
            SubGeometryKind::Face => "face",
            SubGeometryKind::Edge => "edge",
            SubGeometryKind::Pointset => "point",
        });
        match self.kind {
            ConstructionPlane => match Axis::along(construction.normal, AXIS_TOLERANCE) {
                Some(normal) => format!("{} plane", normal.plane_name()),
                None => "Plane".to_owned(),
            },
            Origin => "Origin".to_owned(),
            GridGuide => "Grid".to_owned(),
            GridAxis => axis.map_or("Axis".to_owned(), |axis| format!("{} axis", axis.name())),
            Corner => "Corner".to_owned(),
            Edge => "Edge".to_owned(),
            Midpoint => "Midpoint".to_owned(),
            FaceCenter => "Face center".to_owned(),
            Center => "Center".to_owned(),
            Face => "Face".to_owned(),
            WireStart => "Close path".to_owned(),
            AxisAligned => axis.map_or("Aligned".to_owned(), |axis| format!("Along {}", axis.name())),
            Parallel => "Parallel".to_owned(),
            Closest => on.map_or("Closest".to_owned(), |on| format!("Closest on {on}")),
            Perpendicular => on.map_or("Perpendicular".to_owned(), |on| format!("Perpendicular to {on}")),
            TangentTangent => "Tangent to circle".to_owned(),
        }
    }

    /// How this snap ranks. See [`SnapTier`].
    fn tier(&self) -> SnapTier {
        use SnapKind::*;
        match self.kind {
            ConstructionPlane => SnapTier::Free,
            GridGuide => SnapTier::Guide,
            GridAxis | AxisAligned | Parallel => SnapTier::Axis,
            Face => SnapTier::Face,
            Edge => SnapTier::Edge,
            Closest | Perpendicular => match self.on {
                Some(on) if on.element.kind == SubGeometryKind::Face => SnapTier::FaceRelation,
                _ => SnapTier::EdgeRelation,
            },
            TangentTangent => SnapTier::EdgeRelation,
            Origin | Corner | Midpoint | FaceCenter | Center | WireStart => SnapTier::Keypoint,
        }
    }
}

/// A strategy that proposes candidate [`Snap`]s for a given input.
///
/// Implementors are pure: all live context arrives via the arguments. Register
/// one with [`SnapEngine::add_provider`] to add a new kind of snap.
pub trait SnapProvider {
    /// The kind(s) this provider can emit. Used to skip it entirely when those
    /// kinds are disabled or not requested.
    fn produces(&self) -> SnapFlags;

    /// Candidate snaps found for `input`, of the kinds it requests.
    fn collect(&self, input: &SnapInput, document: &Document, settings: &SnapSettings) -> Vec<Snap>;
}

/// User-facing snap configuration.
#[derive(Debug, Clone, Copy)]
pub struct SnapSettings {
    /// Master switch. When `false`, [`SnapEngine::snap`] returns only the
    /// construction-plane fallback.
    pub enabled: bool,
    /// Globally-enabled snap kinds.
    pub enabled_kinds: SnapFlags,
    /// Screen-space radius, in pixels, within which a non-fallback candidate may
    /// snap.
    pub pixel_tolerance: f32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            enabled_kinds: SnapFlags::all(),
            pixel_tolerance: 12.0,
        }
    }
}

/// Holds the registered providers and the user [`SnapSettings`], and resolves a
/// [`SnapInput`] to the single best [`Snap`].
pub struct SnapEngine {
    providers: Vec<Box<dyn SnapProvider>>,
    /// User-configurable settings (enable flags, tolerance).
    pub settings: SnapSettings,
}

impl Default for SnapEngine {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl SnapEngine {
    /// An engine pre-populated with the built-in providers: construction plane,
    /// origin, grid (axes + guides), geometry corners, edges, midpoints, faces
    /// and face centres, and the snaps relating a stroke's next point to those
    /// before it.
    pub fn with_defaults() -> Self {
        let mut engine = Self {
            providers: Vec::new(),
            settings: SnapSettings::default(),
        };
        engine.add_provider(Box::new(providers::ConstructionPlaneSnap));
        engine.add_provider(Box::new(providers::OriginSnap));
        engine.add_provider(Box::new(providers::GridSnap));
        engine.add_provider(Box::new(providers::GeometrySnap));
        engine.add_provider(Box::new(stroke::StrokeSnap));
        engine
    }

    /// Registers a custom provider.
    pub fn add_provider(&mut self, provider: Box<dyn SnapProvider>) {
        self.providers.push(provider);
    }

    /// Resolves the best snap for `input`, or `None` if nothing (not even the
    /// plane fallback) applies.
    pub fn snap(&self, input: &SnapInput, document: &Document) -> Option<Snap> {
        // Effective kinds = requested ∩ enabled, but the construction-plane
        // fallback is always available so operators still get a position.
        let mut active = if self.settings.enabled {
            input.requested & self.settings.enabled_kinds
        } else {
            SnapFlags::empty()
        };
        active |= SnapFlags::CONSTRUCTION_PLANE; // Fallback, always active
        let input = &SnapInput { requested: active, ..*input };

        let mut candidates = Vec::new();
        for provider in &self.providers {
            if provider.produces().intersects(active) {
                let provider_candidates = provider.collect(input, document, &self.settings);
                candidates.extend(provider_candidates);
            }
        }
        // A provider may emit several kinds; keep only the active ones.
        candidates.retain(|c| active.contains(c.kind.flag()));

        self.rank(input, &candidates)
    }

    /// Ranks candidates by `(tier, then nearest screen distance)`, dropping any
    /// behind the camera or — for non-fallback kinds — beyond the pixel
    /// tolerance. Of two that coincide on screen, the nearer the eye wins.
    fn rank(&self, input: &SnapInput, candidates: &[Snap]) -> Option<Snap> {
        /// Screen distance within which two candidates are one place.
        const COINCIDENT: f32 = 1.0;
        struct Ranked {
            tier: SnapTier,
            screen: (f32, f32),
            dist: f32,
            depth: Real,
            snap: Snap,
        }

        let cam = input.camera;
        let view = cam.forward();
        let (vw, vh) = input.viewport;
        let (cursor_x, cursor_y) = input.cursor;

        let mut best: Option<Ranked> = None;
        for &snap in candidates {
            // Reject candidates behind the camera; their projection is meaningless.
            let depth = (snap.position - cam.eye).dot(view);
            if depth <= 0.0 {
                continue;
            }
            let screen = cam.project_point_screen(snap.position, vw, vh);
            let (screen_x, screen_y) = (screen.x as f32, screen.y as f32);
            let dist = (screen_x - cursor_x).hypot(screen_y - cursor_y);

            // The fallback projects onto the cursor itself, so it is never gated;
            // every real snap must be within tolerance.
            if snap.kind != SnapKind::ConstructionPlane && dist > self.settings.pixel_tolerance {
                continue;
            }

            let tier = snap.tier();
            let better = match &best {
                None => true,
                Some(best) if tier != best.tier => tier > best.tier,
                Some(best) => {
                    let (best_x, best_y) = best.screen;
                    let apart = (screen_x - best_x).hypot(screen_y - best_y);
                    match apart <= COINCIDENT {
                        true => depth < best.depth,
                        false => dist < best.dist,
                    }
                }
            };
            if better {
                best = Some(Ranked { tier, screen: (screen_x, screen_y), dist, depth, snap });
            }
        }

        best.map(|best| best.snap)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::*;
    use duck_engine_viewer::common::Real;
    use duck_engine_viewer::scene::resource::SubGeometryElement;
    use duck_engine_viewer::scene::{Projection, Scene};
    use glam::DVec3;
    use opencascade::primitives::Shape;

    use crate::document::PartId;

    /// A camera looking down −Z from (0, 0, 10) at the origin, 800×600 viewport.
    fn test_camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 10.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 800.0 / 600.0,
            projection: Projection::Perspective { fovy: 45.0, znear: 0.01, zfar: 1000.0 },
        }
    }

    /// The screen pixel a world point projects to, used to aim candidates/cursor.
    fn screen_of(cam: &PositionedCamera, p: Point3) -> (f32, f32) {
        let s = cam.project_point_screen(p, 800, 600);
        (s.x as f32, s.y as f32)
    }

    /// A camera looking straight down −Y from (0, 10, 0), so cursor rays meet
    /// the XZ plane.
    fn top_down_camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 10.0, 0.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 0.0, -1.0),
            aspect: 800.0 / 600.0,
            projection: Projection::Perspective { fovy: 45.0, znear: 0.01, zfar: 1000.0 },
        }
    }

    fn bare_engine() -> SnapEngine {
        SnapEngine {
            providers: Vec::new(),
            settings: SnapSettings::default(),
        }
    }

    #[test]
    fn tier_ordering_low_to_high() {
        assert!(SnapTier::Free < SnapTier::Guide);
        assert!(SnapTier::Guide < SnapTier::Axis);
        assert!(SnapTier::Axis < SnapTier::Face);
        assert!(SnapTier::Face < SnapTier::FaceRelation);
        assert!(SnapTier::FaceRelation < SnapTier::Edge);
        assert!(SnapTier::Edge < SnapTier::EdgeRelation);
        assert!(SnapTier::EdgeRelation < SnapTier::Keypoint);
    }

    #[test]
    fn kind_flag_roundtrip() {
        for kind in [
            SnapKind::ConstructionPlane,
            SnapKind::Origin,
            SnapKind::GridGuide,
            SnapKind::GridAxis,
            SnapKind::Corner,
            SnapKind::Edge,
            SnapKind::Midpoint,
            SnapKind::Face,
            SnapKind::FaceCenter,
            SnapKind::Center,
            SnapKind::WireStart,
            SnapKind::AxisAligned,
            SnapKind::Parallel,
            SnapKind::Closest,
            SnapKind::Perpendicular,
            SnapKind::TangentTangent,
        ] {
            assert!(SnapFlags::all().contains(kind.flag()));
        }
        assert_ne!(SnapKind::Corner.flag(), SnapKind::Origin.flag());
    }

    #[test]
    fn higher_tier_beats_closer_lower_tier() {
        let cam = test_camera();
        let engine = bare_engine();
        // A corner exactly under the cursor competes with a guide 3px away.
        let corner = Point3::new(0.0, 0.0, 0.0);
        let cursor = screen_of(&cam, corner);
        let guide_world = cam
            .unproject_point_screen(cursor.0 + 3.0, cursor.1, 0.5, 800, 600)
            .unwrap();
        let candidates = vec![
            Snap::new(guide_world, SnapKind::GridGuide),
            Snap::new(corner, SnapKind::Corner),
        ];
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let inp = SnapInput::new(&cam, (800, 600), cursor, &plane, &grid);
        assert_eq!(engine.rank(&inp, &candidates).unwrap().kind, SnapKind::Corner);
    }

    /// Two corners on one line of sight, a back one a little nearer the cursor
    /// on screen: the front one is what the user sees there.
    #[test]
    fn of_coincident_candidates_the_nearer_wins() {
        let cam = test_camera();
        let engine = bare_engine();
        let front = Point3::new(1.0, 0.0, 0.0);
        let (x, y) = screen_of(&cam, front);
        let through = cam.unproject_point_screen(x + 0.5, y, 0.5, 800, 600).unwrap();
        let back = cam.eye + (through - cam.eye).normalize() * 15.0;
        let cursor = (x + 0.6, y);
        let corner = |position| Snap::new(position, SnapKind::Corner);
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let inp = SnapInput::new(&cam, (800, 600), cursor, &plane, &grid);
        for candidates in [[corner(front), corner(back)], [corner(back), corner(front)]] {
            assert_eq!(engine.rank(&inp, &candidates).unwrap().position, front);
        }
    }

    /// A `kind` relation at `position`, on an element of `element` kind.
    fn relation(kind: SnapKind, element: SubGeometryKind, position: Point3) -> Snap {
        let on = PartElement { part: PartId::new(), element: SubGeometryElement::new(element, 0) };
        Snap::new(position, kind).with_on(on)
    }

    /// The world point `pixels` to the right of `point` on screen.
    fn beside(cam: &PositionedCamera, point: Point3, pixels: f32) -> Point3 {
        let (x, y) = screen_of(cam, point);
        cam.unproject_point_screen(x + pixels, y, 0.5, 800, 600).unwrap()
    }

    #[test]
    fn an_axis_snap_names_the_world_axis_it_runs_along_either_way() {
        let at = Point3::new(1.0, 0.0, 0.0);
        let plane = Plane::xz();
        let axis = Snap::new(at, SnapKind::GridAxis).with_direction(-Vector3::unit_y());
        assert_eq!(axis.describe(&plane), "Y axis");
        let aligned = Snap::new(at, SnapKind::AxisAligned).with_direction(Vector3::unit_x());
        assert_eq!(aligned.describe(&plane), "Along X");
    }

    #[test]
    fn an_axis_snap_off_the_world_axes_names_none() {
        let at = Point3::new(1.0, 0.0, 0.0);
        let plane = Plane::xz();
        let skew = Vector3::new(1.0, 1.0, 0.0).normalize();
        assert_eq!(Snap::new(at, SnapKind::GridAxis).with_direction(skew).describe(&plane), "Axis");
        assert_eq!(Snap::new(at, SnapKind::AxisAligned).with_direction(skew).describe(&plane), "Aligned");
    }

    #[test]
    fn a_free_point_names_the_world_plane_it_lies_on() {
        let free = Snap::new(Point3::new(1.0, 2.0, 5.0), SnapKind::ConstructionPlane);
        let raised = Plane::from_point(-Vector3::unit_z(), Point3::new(0.0, 0.0, 5.0));
        assert_eq!(free.describe(&raised), "XY plane");
        let tilted = Plane::from_point(Vector3::new(0.0, 1.0, 1.0), Point3::new(0.0, 0.0, 0.0));
        assert_eq!(free.describe(&tilted), "Plane");
    }

    #[test]
    fn a_relation_names_what_it_relates_to() {
        let at = Point3::new(1.0, 0.0, 0.0);
        let plane = Plane::xz();
        let foot = relation(SnapKind::Perpendicular, SubGeometryKind::Edge, at);
        assert_eq!(foot.describe(&plane), "Perpendicular to edge");
        let nearest = relation(SnapKind::Closest, SubGeometryKind::Face, at);
        assert_eq!(nearest.describe(&plane), "Closest on face");
    }

    #[test]
    fn a_relation_near_the_cursor_beats_the_edge_it_lies_on() {
        let cam = test_camera();
        let engine = bare_engine();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let under = Point3::new(0.0, 0.0, 0.0);
        let foot = beside(&cam, under, 3.0);
        let candidates = [
            Snap::new(under, SnapKind::Edge),
            relation(SnapKind::Perpendicular, SubGeometryKind::Edge, foot),
        ];
        let inp = SnapInput::new(&cam, (800, 600), screen_of(&cam, under), &plane, &grid);

        let snap = engine.rank(&inp, &candidates).unwrap();

        assert_eq!(snap.kind, SnapKind::Perpendicular);
        assert_eq!(snap.position, foot);
    }

    #[test]
    fn a_relation_beyond_tolerance_leaves_the_edge_it_lies_on() {
        let cam = test_camera();
        let engine = bare_engine();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let under = Point3::new(0.0, 0.0, 0.0);
        let foot = beside(&cam, under, engine.settings.pixel_tolerance + 1.0);
        let candidates = [
            Snap::new(under, SnapKind::Edge),
            relation(SnapKind::Perpendicular, SubGeometryKind::Edge, foot),
        ];
        let inp = SnapInput::new(&cam, (800, 600), screen_of(&cam, under), &plane, &grid);

        assert_eq!(engine.rank(&inp, &candidates).unwrap().kind, SnapKind::Edge);
    }

    #[test]
    fn a_corner_near_the_cursor_beats_a_nearer_relation() {
        let cam = test_camera();
        let engine = bare_engine();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let under = Point3::new(0.0, 0.0, 0.0);
        let candidates = [
            relation(SnapKind::Closest, SubGeometryKind::Edge, under),
            Snap::new(beside(&cam, under, 3.0), SnapKind::Corner),
        ];
        let inp = SnapInput::new(&cam, (800, 600), screen_of(&cam, under), &plane, &grid);

        assert_eq!(engine.rank(&inp, &candidates).unwrap().kind, SnapKind::Corner);
    }

    /// The face's relation is under the cursor and the edge's beside it, but
    /// an edge's relation outranks the face's, as the edge outranks the face.
    #[test]
    fn an_edge_relation_beats_a_face_relation() {
        let cam = test_camera();
        let engine = bare_engine();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let under = Point3::new(0.0, 0.0, 0.0);
        let candidates = [
            relation(SnapKind::Closest, SubGeometryKind::Face, under),
            relation(SnapKind::Closest, SubGeometryKind::Edge, beside(&cam, under, 3.0)),
        ];
        let inp = SnapInput::new(&cam, (800, 600), screen_of(&cam, under), &plane, &grid);

        let snap = engine.rank(&inp, &candidates).unwrap();

        assert_eq!(snap.on.map(|on| on.element.kind), Some(SubGeometryKind::Edge));
    }

    /// Looking down on a 2-unit box centred on the origin, with the last point
    /// out beyond its top edge z = 1: the cursor snaps square onto that edge
    /// near the foot of the perpendicular, and anywhere else along it, onto
    /// the edge where it is.
    #[test]
    fn an_edge_relates_to_the_last_point_only_near_the_relation() {
        const EPSILON: Real = 1e-6;
        let cam = top_down_camera();
        let engine = SnapEngine::with_defaults();
        let (document, _node) = crate::testing::doc_with_box();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let mut stroke = Stroke::new(true);
        stroke.place(Snap::new(Point3::new(0.3, 1.0, 4.0), SnapKind::ConstructionPlane));
        let foot = Point3::new(0.3, 1.0, 1.0);
        // Both clear of the edge's corners and midpoint; the first within the
        // pixel tolerance of the foot, the second far along the edge from it.
        let near = Point3::new(0.35, 1.0, 1.0);
        let far = Point3::new(-0.6, 1.0, 1.0);

        let mut inp = SnapInput::new(&cam, (800, 600), screen_of(&cam, near), &plane, &grid);
        assert_eq!(engine.snap(&inp, &document).unwrap().kind, SnapKind::Edge, "no stroke, no relation");

        inp.stroke = Some(&stroke);
        let snap = engine.snap(&inp, &document).unwrap();
        assert!(matches!(snap.kind, SnapKind::Closest | SnapKind::Perpendicular), "{snap:?}");
        assert!((snap.position - foot).magnitude() < EPSILON, "{:?}", snap.position);
        assert_eq!(snap.on.map(|on| on.element.kind), Some(SubGeometryKind::Edge));

        let mut inp = SnapInput::new(&cam, (800, 600), screen_of(&cam, far), &plane, &grid);
        inp.stroke = Some(&stroke);
        let snap = engine.snap(&inp, &document).unwrap();
        assert_eq!(snap.kind, SnapKind::Edge);
        assert!((snap.position - far).magnitude() < EPSILON, "{:?}", snap.position);
    }

    #[test]
    fn fallback_wins_when_nothing_in_tolerance() {
        let cam = test_camera();
        let engine = bare_engine();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let cursor = (400.0, 300.0);
        // A corner far from the cursor (well beyond tolerance) plus the fallback.
        let far = cam.unproject_point_screen(700.0, 500.0, 0.5, 800, 600).unwrap();
        let fallback = cam.unproject_point_screen(cursor.0, cursor.1, 0.5, 800, 600).unwrap();
        let candidates = vec![
            Snap::new(far, SnapKind::Corner),
            Snap::new(fallback, SnapKind::ConstructionPlane),
        ];
        let inp = SnapInput::new(&cam, (800, 600), cursor, &plane, &grid);
        assert_eq!(
            engine.rank(&inp, &candidates).unwrap().kind,
            SnapKind::ConstructionPlane
        );
    }

    #[test]
    fn behind_camera_candidate_rejected() {
        let cam = test_camera();
        let engine = bare_engine();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let cursor = (400.0, 300.0);
        // Behind the camera (eye at z=10 looking toward −z; z=20 is behind it).
        let candidates = vec![Snap::new(Point3::new(0.0, 0.0, 20.0), SnapKind::Corner)];
        let inp = SnapInput::new(&cam, (800, 600), cursor, &plane, &grid);
        assert!(engine.rank(&inp, &candidates).is_none());
    }

    #[test]
    fn disabled_engine_returns_only_fallback() {
        // A top-down camera, so the cursor ray actually meets the XZ plane — the
        // default test camera looks parallel to it.
        let cam = top_down_camera();
        let mut engine = SnapEngine::with_defaults();
        engine.settings.enabled = false;
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let document = Document::new(Scene::default());
        // Aim straight at the origin; with snapping on this would be Origin, but
        // disabled it must fall through to the construction plane.
        let cursor = screen_of(&cam, Point3::new(0.0, 0.0, 0.0));
        let inp = SnapInput::new(&cam, (800, 600), cursor, &plane, &grid);
        assert_eq!(
            engine.snap(&inp, &document).unwrap().kind,
            SnapKind::ConstructionPlane
        );
    }

    /// A provider of every kind, which notes the kinds it is asked for.
    struct Asked(Rc<Cell<SnapFlags>>);

    impl SnapProvider for Asked {
        fn produces(&self) -> SnapFlags {
            SnapFlags::all()
        }

        fn collect(&self, input: &SnapInput, _document: &Document, _settings: &SnapSettings) -> Vec<Snap> {
            self.0.set(input.requested);
            Vec::new()
        }
    }

    #[test]
    fn a_provider_is_asked_only_for_requested_kinds_that_are_enabled() {
        let cam = top_down_camera();
        let asked = Rc::new(Cell::new(SnapFlags::empty()));
        let mut engine = bare_engine();
        engine.add_provider(Box::new(Asked(Rc::clone(&asked))));
        engine.settings.enabled_kinds = SnapFlags::all() - SnapFlags::FACE;
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let document = Document::new(Scene::default());
        let mut inp = SnapInput::new(&cam, (800, 600), (400.0, 300.0), &plane, &grid);
        inp.requested = SnapFlags::CORNER | SnapFlags::FACE;

        engine.snap(&inp, &document);

        assert_eq!(asked.get(), SnapFlags::CORNER | SnapFlags::CONSTRUCTION_PLANE);
    }

    #[test]
    fn midpoint_beats_the_edge_it_lies_on() {
        const EPSILON: Real = 1e-6;
        // Looking down on a 2-unit box centred on the origin, with the cursor
        // just off the middle of its top edge x = 1, y = 1.
        let cam = top_down_camera();
        let engine = SnapEngine::with_defaults();
        let (document, _node) = crate::testing::doc_with_box();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let cursor = screen_of(&cam, Point3::new(1.0, 1.0, 0.05));
        let inp = SnapInput::new(&cam, (800, 600), cursor, &plane, &grid);

        let snap = engine.snap(&inp, &document).unwrap();

        assert_eq!(snap.kind, SnapKind::Midpoint);
        assert!((snap.position - Point3::new(1.0, 1.0, 0.0)).magnitude() < EPSILON, "{:?}", snap.position);
    }

    #[test]
    fn face_center_beats_the_face_it_lies_on() {
        const EPSILON: Real = 1e-6;
        // Looking down on a 2-unit box off the origin (whose snap would otherwise
        // tie with the centre on screen), with the cursor just off the middle of
        // its top face y = 2.
        let cam = top_down_camera();
        let engine = SnapEngine::with_defaults();
        let (low, high) = (DVec3::new(2.0, 0.0, -1.0), DVec3::new(4.0, 2.0, 1.0));
        let (document, _node) = crate::testing::doc_with_shape(Shape::box_from_corners(low, high));
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let cursor = screen_of(&cam, Point3::new(3.05, 2.0, 0.05));
        let inp = SnapInput::new(&cam, (800, 600), cursor, &plane, &grid);

        let snap = engine.snap(&inp, &document).unwrap();

        assert_eq!(snap.kind, SnapKind::FaceCenter);
        assert!((snap.position - Point3::new(3.0, 2.0, 0.0)).magnitude() < EPSILON, "{:?}", snap.position);
    }

    /// Looking down a cylinder's axis, both rims' centres are under the
    /// cursor; the top one is in front.
    #[test]
    fn the_near_rim_center_beats_the_far_one() {
        const EPSILON: Real = 1e-6;
        let cam = PositionedCamera {
            eye: Point3::new(3.0, 10.0, 0.0),
            target: Point3::new(3.0, 0.0, 0.0),
            ..top_down_camera()
        };
        let engine = SnapEngine::with_defaults();
        let cylinder = Shape::cylinder(DVec3::new(3.0, 0.0, 0.0), 1.0, DVec3::Y, 1.0);
        let (document, _node) = crate::testing::doc_with_shape(cylinder);
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let mut inp = SnapInput::new(&cam, (800, 600), screen_of(&cam, Point3::new(3.02, 1.0, 0.0)), &plane, &grid);
        inp.requested = SnapFlags::CENTER;

        let snap = engine.snap(&inp, &document).unwrap();

        assert_eq!(snap.kind, SnapKind::Center);
        assert!((snap.position - Point3::new(3.0, 1.0, 0.0)).magnitude() < EPSILON, "{:?}", snap.position);
    }
}
