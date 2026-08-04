//! Snapping system: operators resolve a cursor ray to a meaningful world
//! position (and, possibly orientation) by consulting a set of pluggable
//! [`SnapProvider`]s.
//!
//! # Design
//!
//! A provider is a pure strategy: given a [`SnapInput`] (the cursor ray plus the
//! active construction context) and the [`Scene`], it proposes candidate
//! [`Snap`]s. The [`SnapEngine`] runs every enabled/requested provider, then
//! **ranks** the candidates by *tier first, then screen-space distance to the
//! cursor*, returning the single best [`Snap`].
//!
//! The construction plane is an always-present, lowest-tier fallback, so an
//! operator constrained to that plane always gets a position — but any
//! higher-tier snap (grid axis, corner, ...) wins whenever one is within the
//! pixel tolerance.
//!
//! # Extending
//!
//! To add a snap: add a [`SnapKind`] variant (plus its one-line [`SnapKind::tier`]
//! and [`SnapKind::flag`] arms and a [`SnapKinds`] bit), implement
//! [`SnapProvider`], and register it in [`SnapEngine::with_defaults`] (or via
//! [`SnapEngine::add_provider`]). [`Snap::direction`] already carries an optional
//! tangent/axis for line snaps and future orientation snaps.

mod providers;

pub(crate) use providers::WireStartSnap;

use bitflags::bitflags;

use duck_engine_viewer::common::{InnerSpace, Plane, Point3, Ray, Vector3};
use duck_engine_viewer::scene::{NodeId, PositionedCamera, Scene};

use crate::grid::GridConfig;

/// What a single snap locks onto. Maps to a ranking [`SnapTier`] and a
/// [`SnapKinds`] flag.
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
    /// A point on an existing geometry face (ray–surface hit under the cursor).
    Face,
    /// The start point of an in-progress wire, offered so the path can be
    /// closed back onto itself.
    WireStart,
}

impl SnapKind {
    fn tier(self) -> SnapTier {
        use SnapKind::*;
        match self {
            ConstructionPlane => SnapTier::Free,
            GridGuide => SnapTier::Guide,
            GridAxis => SnapTier::Axis,
            Face => SnapTier::Face,
            Edge => SnapTier::Edge,
            Origin | Corner | WireStart => SnapTier::Feature,
        }
    }

    fn flag(self) -> SnapFlags {
        use SnapKind::*;
        match self {
            ConstructionPlane => SnapFlags::CONSTRUCTION_PLANE,
            Origin => SnapFlags::ORIGIN,
            GridGuide => SnapFlags::GRID_GUIDE,
            GridAxis => SnapFlags::GRID_AXIS,
            Corner => SnapFlags::CORNER,
            Edge => SnapFlags::EDGE,
            Face => SnapFlags::FACE,
            WireStart => SnapFlags::WIRE_START,
        }
    }
}

/// Ranking precedence, *defined by declaration order* (low → high): a higher
/// tier always beats a lower one when both are within tolerance; ties break on
/// screen distance. Reordering these variants is the single place snap
/// precedence is expressed.
///
/// (Rust derives [`Ord`] on a field-less enum by discriminant, i.e. declaration
/// order, so `Free < Guide < Axis < Feature`.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SnapTier {
    /// Free placement on the construction plane.
    Free,
    /// A grid guide point (low-value hint).
    Guide,
    /// A construction-frame principal axis.
    Axis,
    /// A point on an existing geometry face (real geometry beats the grid, but
    /// any edge or feature within tolerance still wins over a mid-face point).
    Face,
    /// A point on an existing geometry edge/wire (real geometry beats the grid,
    /// but a discrete corner still wins over a mid-edge point).
    Edge,
    /// A geometry feature or the frame origin (a deliberate, high-value target).
    Feature,
}

bitflags! {
    /// A *set* of snap kinds: which the engine has enabled, which an operator
    /// requests, and which a provider can emit. Modeled on `NodeFlags` in the
    /// scene crate — explicit bits, so nothing relies on enum discriminant order.
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
    }
}

/// Per-call inputs describing where the user is pointing and the active
/// construction context. Cheap to build each cursor move; everything is borrowed
/// so changing the construction plane at runtime needs no provider rebuild.
pub struct SnapInput<'a> {
    /// Camera ray through the cursor, in world space.
    pub ray: Ray,
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
    /// Which kinds the calling operator wants; intersected with the engine's
    /// globally-enabled kinds.
    pub requested: SnapFlags,
    /// Nodes to ignore (e.g. the operator's own in-progress preview geometry).
    pub exclude_nodes: &'a [NodeId],
}

/// A snap location: proposed by a [`SnapProvider`] as a candidate, and returned
/// by [`SnapEngine::snap`] as the chosen result (the two are the same thing).
#[derive(Debug, Clone, Copy)]
pub struct Snap {
    /// World-space position to snap to.
    pub position: Point3,
    /// Optional associated direction: the tangent for line snaps (edges, grid
    /// axes) and the surface normal for face snaps.
    pub direction: Option<Vector3>,
    /// What this snap locks onto (drives tier + filtering).
    pub kind: SnapKind,
}

/// A strategy that proposes candidate [`Snap`]s for a given input.
///
/// Implementors are pure: all live context arrives via the arguments. Register
/// one with [`SnapEngine::add_provider`] to add a new kind of snap.
pub trait SnapProvider {
    /// The kind(s) this provider can emit. Used to skip it entirely when those
    /// kinds are disabled or not requested.
    fn produces(&self) -> SnapFlags;

    /// Candidate snaps found for `input`.
    fn collect(&self, input: &SnapInput, scene: &Scene, settings: &SnapSettings) -> Vec<Snap>;
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
    /// origin, grid (axes + guides), geometry corners, and geometry edges.
    pub fn with_defaults() -> Self {
        let mut engine = Self {
            providers: Vec::new(),
            settings: SnapSettings::default(),
        };
        engine.add_provider(Box::new(providers::ConstructionPlaneSnap));
        engine.add_provider(Box::new(providers::OriginSnap));
        engine.add_provider(Box::new(providers::GridSnap));
        engine.add_provider(Box::new(providers::GeometrySnap));
        engine
    }

    /// Registers a custom provider.
    pub fn add_provider(&mut self, provider: Box<dyn SnapProvider>) {
        self.providers.push(provider);
    }

    /// Resolves the best snap for `input`, or `None` if nothing (not even the
    /// plane fallback) applies. `extra` holds caller-supplied per-call providers
    /// (e.g. a wire-start snap), consulted alongside the registered ones.
    pub fn snap(
        &self,
        input: &SnapInput,
        scene: &Scene,
        extra: &[&dyn SnapProvider],
    ) -> Option<Snap> {
        // Effective kinds = requested ∩ enabled, but the construction-plane
        // fallback is always available so operators still get a position.
        let mut active = if self.settings.enabled {
            input.requested & self.settings.enabled_kinds
        } else {
            SnapFlags::empty()
        };
        active |= SnapFlags::CONSTRUCTION_PLANE;

        let mut candidates = Vec::new();
        let builtin = self.providers.iter().map(|p| p.as_ref());
        for provider in builtin.chain(extra.iter().copied()) {
            if provider.produces().intersects(active) {
                let provider_candidates = provider.collect(input, &scene, &self.settings);
                candidates.extend(provider_candidates);
            }
        }
        // A provider may emit several kinds; keep only the active ones.
        candidates.retain(|c| active.contains(c.kind.flag()));

        self.rank(input, &candidates)
    }

    /// Ranks candidates by `(tier, then nearest screen distance)`, dropping any
    /// behind the camera or — for non-fallback kinds — beyond the pixel
    /// tolerance.
    fn rank(&self, input: &SnapInput, candidates: &[Snap]) -> Option<Snap> {
        let cam = input.camera;
        let view = cam.forward();
        let (vw, vh) = input.viewport;

        let mut best: Option<(SnapTier, f32, Snap)> = None;
        for &c in candidates {
            // Reject candidates behind the camera; their projection is meaningless.
            if (c.position - cam.eye).dot(view) <= 0.0 {
                continue;
            }
            let screen = cam.project_point_screen(c.position, vw, vh);
            let dx = screen.x - input.cursor.0;
            let dy = screen.y - input.cursor.1;
            let dist = (dx * dx + dy * dy).sqrt();

            // The fallback projects onto the cursor itself, so it is never gated;
            // every real snap must be within tolerance.
            if c.kind != SnapKind::ConstructionPlane && dist > self.settings.pixel_tolerance {
                continue;
            }

            let tier = c.kind.tier();
            let better = match best {
                None => true,
                Some((best_tier, best_dist, _)) => {
                    tier > best_tier || (tier == best_tier && dist < best_dist)
                }
            };
            if better {
                best = Some((tier, dist, c));
            }
        }

        best.map(|(_, _, snap)| snap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera looking down −Z from (0, 0, 10) at the origin, 800×600 viewport.
    fn test_camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 10.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 800.0 / 600.0,
            fovy: 45.0,
            znear: 0.01,
            zfar: 1000.0,
            ortho: false,
        }
    }

    fn input<'a>(
        cam: &'a PositionedCamera,
        plane: &'a Plane,
        grid: &'a GridConfig,
        cursor: (f32, f32),
    ) -> SnapInput<'a> {
        SnapInput {
            ray: cam.ray_from_screen_point(cursor.0, cursor.1, 800, 600),
            cursor,
            viewport: (800, 600),
            camera: cam,
            plane,
            grid,
            requested: SnapFlags::all(),
            exclude_nodes: &[],
        }
    }

    /// The screen pixel a world point projects to, used to aim candidates/cursor.
    fn screen_of(cam: &PositionedCamera, p: Point3) -> (f32, f32) {
        let s = cam.project_point_screen(p, 800, 600);
        (s.x, s.y)
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
        assert!(SnapTier::Face < SnapTier::Edge);
        assert!(SnapTier::Edge < SnapTier::Feature);
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
            SnapKind::Face,
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
            Snap { position: guide_world, direction: None, kind: SnapKind::GridGuide },
            Snap { position: corner, direction: None, kind: SnapKind::Corner },
        ];
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let inp = input(&cam, &plane, &grid, cursor);
        assert_eq!(engine.rank(&inp, &candidates).unwrap().kind, SnapKind::Corner);
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
            Snap { position: far, direction: None, kind: SnapKind::Corner },
            Snap { position: fallback, direction: None, kind: SnapKind::ConstructionPlane },
        ];
        let inp = input(&cam, &plane, &grid, cursor);
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
        let candidates = vec![Snap {
            position: Point3::new(0.0, 0.0, 20.0),
            direction: None,
            kind: SnapKind::Corner,
        }];
        let inp = input(&cam, &plane, &grid, cursor);
        assert!(engine.rank(&inp, &candidates).is_none());
    }

    #[test]
    fn disabled_engine_returns_only_fallback() {
        // A top-down camera, so the cursor ray actually meets the XZ plane — the
        // default test camera looks parallel to it.
        let cam = PositionedCamera {
            eye: Point3::new(0.0, 10.0, 0.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 0.0, -1.0),
            aspect: 800.0 / 600.0,
            fovy: 45.0,
            znear: 0.01,
            zfar: 1000.0,
            ortho: false,
        };
        let mut engine = SnapEngine::with_defaults();
        engine.settings.enabled = false;
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let scene = Scene::default();
        // Aim straight at the origin; with snapping on this would be Origin, but
        // disabled it must fall through to the construction plane.
        let cursor = screen_of(&cam, Point3::new(0.0, 0.0, 0.0));
        let inp = input(&cam, &plane, &grid, cursor);
        assert_eq!(
            engine.snap(&inp, &scene, &[]).unwrap().kind,
            SnapKind::ConstructionPlane
        );
    }
}
