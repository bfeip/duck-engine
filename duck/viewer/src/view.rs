//! Views: independently rendered, independently interactive regions of a
//! viewer's target.
//!
//! A [`Viewer`](crate::Viewer) hosts any number of views. Each view pairs a
//! scene handle with its own camera node, renderer (workflow, background,
//! depth/MSAA targets), operator stack, and cursor state — a view is as capable
//! as a whole single-view viewer. Views are placed by a [`ViewLayout`] and may
//! tile the target or overlap it; the viewer composites them in stack order
//! (last view on top) and routes input to the view under the cursor.
//!
//! Views of the same scene share that scene's GPU resources and selection
//! (selecting in one pane highlights in all), while each keeps its own camera.
//! A view over its own small scene with a transparent background acts as an
//! overlay — e.g. an axis triad in a corner.

use duck_engine_scene::resource::NodeId;
use duck_engine_scene::Scene;

use crate::{
    event::EventDispatcher,
    renderer::Renderer,
    scene::PositionedCamera,
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

/// One independently rendered, independently interactive region of a viewer.
/// See the [module docs](self) for the concept; create views with
/// [`Viewer::add_view`](crate::Viewer::add_view) and mutate them through
/// [`Viewer::view_mut`](crate::Viewer::view_mut).
pub struct View {
    pub(crate) id: ViewId,
    pub(crate) name: String,
    pub(crate) scene: Scene,
    /// The scene node this view renders from.
    pub(crate) camera_node: NodeId,
    /// True when the view created its camera node (and removes it on drop);
    /// false when it adopted the scene's own active camera.
    pub(crate) owns_camera_node: bool,
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

    /// The scene node this view renders from.
    pub fn camera_node(&self) -> NodeId {
        self.camera_node
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

    /// Builds a [`PositionedCamera`] for this view's camera node at the view's
    /// aspect ratio. `None` if the node has been removed from the scene.
    pub fn camera(&self) -> Option<PositionedCamera> {
        self.scene.camera_for_node(self.camera_node, self.aspect())
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
