// Screen overlay: rectangles drawn over the composited views.
//
// Each instance is one rectangle in target pixels, expanded to two triangles.
// Colors arrive premultiplied, matching the pipeline's blend.

struct Rect {
    // Top-left and bottom-right corners, in target pixels.
    @location(0) min: vec2<f32>,
    @location(1) max: vec2<f32>,
    // Interior and border colors, linear and premultiplied.
    @location(2) fill_color: vec4<f32>,
    @location(3) stroke_color: vec4<f32>,
    // Border width in pixels, inside the rectangle.
    @location(4) stroke_width: f32,
    // The whole target's width and height in pixels.
    @location(5) target_size: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) min: vec2<f32>,
    @location(1) @interpolate(flat) max: vec2<f32>,
    @location(2) @interpolate(flat) fill_color: vec4<f32>,
    @location(3) @interpolate(flat) stroke_color: vec4<f32>,
    @location(4) @interpolate(flat) stroke_width: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32, rect: Rect) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let pixel = mix(rect.min, rect.max, corners[index]);
    // Target pixels (y-down) to NDC (y-up).
    let ndc = vec2<f32>(pixel.x / rect.target_size.x * 2.0 - 1.0, 1.0 - pixel.y / rect.target_size.y * 2.0);

    var out: VertexOutput;
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.min = rect.min;
    out.max = rect.max;
    out.fill_color = rect.fill_color;
    out.stroke_color = rect.stroke_color;
    out.stroke_width = rect.stroke_width;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Distance in pixels from this fragment's center to the nearest edge.
    let p = in.position.xy;
    let edge = min(min(p.x - in.min.x, in.max.x - p.x), min(p.y - in.min.y, in.max.y - p.y));
    return select(in.fill_color, in.stroke_color, edge < in.stroke_width);
}
