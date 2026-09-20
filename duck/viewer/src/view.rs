//! Views: independently rendered, independently interactive regions of a
//! viewer's target.
//!
//! A [`Viewer`](crate::Viewer) hosts any number of views. Each view owns its
//! camera and pairs a scene handle with a renderer (workflow, background,
//! depth/MSAA targets), operator stack, and cursor state — a view is as capable
//! as a whole single-view viewer. Views are placed by a [`ViewLayout`] and may
//! tile the target or overlap it; the viewer composites them in stack order
//! (last view on top) and routes input to the view under the cursor.
//!
//! Views of the same scene share that scene's GPU resources and selection
//! (selecting in one pane highlights in all), while each keeps its own camera.
//! A view over its own small scene with a transparent background acts as an
//! overlay — e.g. an axis triad in a corner.

use duck_engine_common::{Deg, Quaternion, Rotation3};
use duck_engine_scene::Scene;

use crate::{
    camera_transition::CameraTransition,
    event::EventDispatcher,
    renderer::Renderer,
    scene::{
        Light, PositionedCamera, PositionedLight, SceneData,
        common::{RgbaColor, Transform},
    },
};

/// Identifies a view within its [`Viewer`](crate::Viewer). Viewer-local and
/// monotonically assigned; not a scene resource id.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ViewId(pub(crate) u64);

/// A corner of the viewer target, for anchored view placement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Where a view sits on the viewer target. Resolved to a [`PixelRect`]
/// whenever the target size changes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ViewLayout {
    /// Fractions of the target in `[0, 1]`; origin top-left, y-down (matching
    /// cursor coordinates). Adjacent fractional views tile without gaps or
    /// overlap: shared edges resolve to the same pixel.
    Fractional { x: f32, y: f32, width: f32, height: f32 },
    /// A fixed pixel size anchored to a corner with a pixel margin.
    Anchored {
        corner: Corner,
        /// View size in physical pixels (width, height).
        size: (u32, u32),
        /// Distance from the anchored corner in physical pixels (x, y).
        margin: (u32, u32),
    },
}

impl ViewLayout {
    /// The whole target.
    pub const FULL: Self = Self::Fractional { x: 0.0, y: 0.0, width: 1.0, height: 1.0 };

    /// Resolve to pixels against a target size. Edges round so that adjacent
    /// fractional views share pixel boundaries; the result is clamped to the
    /// target and never zero-sized.
    pub(crate) fn resolve(&self, target: (u32, u32)) -> PixelRect {
        let (tw, th) = (target.0.max(1), target.1.max(1));
        match *self {
            Self::Fractional { x, y, width, height } => {
                let edge = |frac: f32, extent: u32| -> u32 {
                    ((frac * extent as f32).round().max(0.0) as u32).min(extent)
                };
                let x0 = edge(x, tw);
                let x1 = edge(x + width, tw);
                let y0 = edge(y, th);
                let y1 = edge(y + height, th);
                PixelRect {
                    x: x0.min(tw - 1),
                    y: y0.min(th - 1),
                    width: x1.saturating_sub(x0).max(1),
                    height: y1.saturating_sub(y0).max(1),
                }
            }
            Self::Anchored { corner, size, margin } => {
                let width = size.0.clamp(1, tw);
                let height = size.1.clamp(1, th);
                let x = match corner {
                    Corner::TopLeft | Corner::BottomLeft => margin.0.min(tw - width),
                    Corner::TopRight | Corner::BottomRight => {
                        tw.saturating_sub(width + margin.0)
                    }
                };
                let y = match corner {
                    Corner::TopLeft | Corner::TopRight => margin.1.min(th - height),
                    Corner::BottomLeft | Corner::BottomRight => {
                        th.saturating_sub(height + margin.1)
                    }
                };
                PixelRect { x, y, width, height }
            }
        }
    }
}

/// A resolved view placement in physical pixels, origin top-left, y-down.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl PixelRect {
    /// True if the point (in target coordinates) lies inside this rect.
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x as f32
            && y >= self.y as f32
            && x < (self.x + self.width) as f32
            && y < (self.y + self.height) as f32
    }
}

/// The color texture one view renders into, later composited onto the viewer
/// target. Recreated whenever the view's pixel size changes.
pub(crate) struct ViewTarget {
    pub texture: wgpu::Texture,
    pub render_view: wgpu::TextureView,
    /// Compositor bind group sampling `render_view`.
    pub bind_group: wgpu::BindGroup,
}

/// When a view's headlight rig ([`View::headlight_rig`]) contributes to its
/// rendering.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum HeadlightMode {
    /// Active only while the view has no lights of its own, its scene has no
    /// lights, and no environment map is active (evaluated each frame).
    #[default]
    Auto,
    On,
    Off,
}

/// The default headlight rig: a three-point directional setup (warm key, cool
/// fill, back rim) over a hemisphere ambient gradient, all in camera space.
///
/// Colors are linear. Intensities suit the Reinhard tonemap in the lit surface
/// shader, which takes no 1/PI on incoming radiance.
pub(crate) fn default_headlight_rig() -> Vec<PositionedLight> {
    // Rotations aim the light's -Z axis in camera space, where +X is right, +Y
    // is up and -Z points away from the viewer.
    let directional = |color, intensity, rotation| {
        PositionedLight::camera(Light::directional(color, intensity), Transform::from_rotation(rotation))
    };
    vec![
        // Key: from the upper left, in front of the subject.
        directional(
            RgbaColor { r: 1.0, g: 0.96, b: 0.90, a: 1.0 },
            9.0,
            Quaternion::from_angle_x(Deg(-40.0)) * Quaternion::from_angle_y(Deg(-30.0)),
        ),
        // Fill: cooler and dimmer, from the lower right.
        directional(
            RgbaColor { r: 0.72, g: 0.80, b: 1.0, a: 1.0 },
            2.6,
            Quaternion::from_angle_x(Deg(26.0)) * Quaternion::from_angle_y(Deg(37.0)),
        ),
        // Rim: from above and behind, separating the silhouette.
        directional(
            RgbaColor { r: 0.85, g: 0.90, b: 1.0, a: 1.0 },
            5.0,
            Quaternion::from_angle_x(Deg(36.0)) * Quaternion::from_angle_y(Deg(160.0)),
        ),
        // Ambient: sky axis near camera up, tilted toward the viewer.
        PositionedLight::camera(
            Light::hemisphere(
                RgbaColor { r: 0.16, g: 0.19, b: 0.24, a: 1.0 },
                RgbaColor { r: 0.045, g: 0.042, b: 0.040, a: 1.0 },
                0.9,
            ),
            Transform::from_rotation(Quaternion::from_angle_x(Deg(-70.0))),
        ),
    ]
}

/// One independently rendered, independently interactive region of a viewer.
/// See the [module docs](self) for the concept; create views with
/// [`Viewer::add_view`](crate::Viewer::add_view) and mutate them through
/// [`Viewer::view_mut`](crate::Viewer::view_mut).
pub struct View {
    pub(crate) id: ViewId,
    pub(crate) name: String,
    pub(crate) scene: Scene,
    /// The camera this view renders from. Its aspect is kept in sync with the
    /// view's pixel size by the viewer.
    pub(crate) camera: PositionedCamera,
    /// Lights this view alone contributes, on top of its scene's lights.
    pub(crate) lights: Vec<PositionedLight>,
    /// When [`headlight_rig`](Self::headlight_rig) contributes to rendering.
    pub(crate) headlight: HeadlightMode,
    /// The fallback rig, kept apart from [`lights`](Self::lights) so that
    /// [`HeadlightMode::Auto`]'s "no other lights" test does not see itself.
    pub(crate) headlight_rig: Vec<PositionedLight>,
    pub(crate) layout: ViewLayout,
    /// Cached `layout.resolve()` against the current target size.
    pub(crate) rect: PixelRect,
    pub(crate) visible: bool,
    pub(crate) renderer: Renderer,
    pub(crate) dispatcher: EventDispatcher,
    /// Cursor position in view-local physical pixels. May leave the rect (and
    /// go negative) while a drag that started here is captured.
    pub(crate) cursor_position: Option<(f32, f32)>,
    pub(crate) target: ViewTarget,
    /// In-flight animated camera move, advanced by the viewer each update.
    pub(crate) transition: Option<CameraTransition>,
}

impl View {
    pub fn id(&self) -> ViewId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// A clone of this view's scene handle.
    pub fn scene(&self) -> Scene {
        self.scene.clone()
    }

    pub fn layout(&self) -> ViewLayout {
        self.layout
    }

    /// Current placement on the viewer target, in physical pixels.
    pub fn rect(&self) -> PixelRect {
        self.rect
    }

    pub fn visible(&self) -> bool {
        self.visible
    }

    /// View size in physical pixels (width, height).
    pub fn size(&self) -> (u32, u32) {
        (self.rect.width, self.rect.height)
    }

    /// View aspect ratio.
    pub fn aspect(&self) -> f32 {
        self.rect.width as f32 / self.rect.height.max(1) as f32
    }

    /// The camera this view renders from.
    pub fn camera(&self) -> &PositionedCamera {
        &self.camera
    }

    /// When this view's headlight rig contributes to rendering.
    pub fn headlight(&self) -> HeadlightMode {
        self.headlight
    }

    /// Lights this view alone contributes, on top of its scene's lights.
    pub fn lights(&self) -> &[PositionedLight] {
        &self.lights
    }

    /// This view's fallback headlight rig.
    pub fn headlight_rig(&self) -> &[PositionedLight] {
        &self.headlight_rig
    }

    /// The lights to render this view with: its scene's lights, then its own,
    /// then the headlight rig if it applies.
    ///
    /// The rig goes last so that it, rather than a light the app deliberately
    /// added, is what falls off the end of
    /// [`MAX_LIGHTS`](crate::scene::MAX_LIGHTS).
    pub(crate) fn effective_lights(
        &self,
        scene_lights: &[PositionedLight],
        scene: &SceneData,
    ) -> Vec<PositionedLight> {
        let rig_on = match self.headlight {
            HeadlightMode::On => true,
            HeadlightMode::Off => false,
            HeadlightMode::Auto => {
                scene_lights.is_empty()
                    && self.lights.is_empty()
                    && scene.active_environment_map().is_none()
            }
        };
        let rig = if rig_on { self.headlight_rig.as_slice() } else { &[] };
        scene_lights.iter().chain(&self.lights).chain(rig).cloned().collect()
    }

    /// The view's operator stack.
    pub fn dispatcher(&self) -> &EventDispatcher {
        &self.dispatcher
    }

    /// The color texture this view renders into, before compositing.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.target.texture
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_quadrants_tile_without_gaps() {
        // An odd target size forces rounding; shared edges must agree.
        let target = (801, 601);
        let q = |x, y| ViewLayout::Fractional { x, y, width: 0.5, height: 0.5 };
        let tl = q(0.0, 0.0).resolve(target);
        let tr = q(0.5, 0.0).resolve(target);
        let bl = q(0.0, 0.5).resolve(target);
        let br = q(0.5, 0.5).resolve(target);

        assert_eq!(tl.x + tl.width, tr.x);
        assert_eq!(tl.y + tl.height, bl.y);
        assert_eq!(tr.x + tr.width, target.0);
        assert_eq!(bl.y + bl.height, target.1);
        assert_eq!(br.x, tr.x);
        assert_eq!(br.y, bl.y);
        assert_eq!(br.x + br.width, target.0);
        assert_eq!(br.y + br.height, target.1);
    }

    #[test]
    fn full_layout_covers_target() {
        let rect = ViewLayout::FULL.resolve((800, 600));
        assert_eq!(rect, PixelRect { x: 0, y: 0, width: 800, height: 600 });
    }

    #[test]
    fn resolve_never_returns_zero_size() {
        let tiny = ViewLayout::Fractional { x: 0.5, y: 0.5, width: 0.0001, height: 0.0001 };
        let rect = tiny.resolve((100, 100));
        assert!(rect.width >= 1 && rect.height >= 1);

        let degenerate_target = ViewLayout::FULL.resolve((0, 0));
        assert!(degenerate_target.width >= 1 && degenerate_target.height >= 1);
    }

    #[test]
    fn anchored_corners_resolve_with_margin() {
        let target = (800, 600);
        let layout = |corner| ViewLayout::Anchored { corner, size: (120, 100), margin: (10, 20) };

        assert_eq!(
            layout(Corner::TopLeft).resolve(target),
            PixelRect { x: 10, y: 20, width: 120, height: 100 }
        );
        assert_eq!(
            layout(Corner::TopRight).resolve(target),
            PixelRect { x: 670, y: 20, width: 120, height: 100 }
        );
        assert_eq!(
            layout(Corner::BottomLeft).resolve(target),
            PixelRect { x: 10, y: 480, width: 120, height: 100 }
        );
        assert_eq!(
            layout(Corner::BottomRight).resolve(target),
            PixelRect { x: 670, y: 480, width: 120, height: 100 }
        );
    }

    #[test]
    fn anchored_clamps_to_small_targets() {
        let rect = ViewLayout::Anchored {
            corner: Corner::BottomRight,
            size: (500, 500),
            margin: (10, 10),
        }
        .resolve((200, 100));
        assert_eq!(rect.width, 200);
        assert_eq!(rect.height, 100);
        assert_eq!((rect.x, rect.y), (0, 0));
    }

    #[test]
    fn pixel_rect_contains_is_half_open() {
        let rect = PixelRect { x: 10, y: 10, width: 20, height: 20 };
        assert!(rect.contains(10.0, 10.0));
        assert!(rect.contains(29.9, 29.9));
        assert!(!rect.contains(30.0, 30.0));
        assert!(!rect.contains(9.9, 15.0));
    }
}
