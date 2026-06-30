use std::sync::{Arc, Mutex};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use duck_engine_viewer::{SurfacedViewer, winit_support};
use duck_engine_viewer::operator::{NavigationOperator, SelectionOperator, TransformMode, TransformOperator};

/// Application state for the winit event loop
struct App<'a> {
    window: Option<Arc<Window>>,
    viewer: Option<SurfacedViewer<'a>>,
}

impl<'a> App<'a> {
    /// Initialize the window and viewer
    fn initialize(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("Duck Engine - Winit Basic Example");

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        // Create viewer with the window
        let size = window.inner_size();
        let mut viewer = pollster::block_on(SurfacedViewer::new(
            Arc::clone(&window),
            size.width,
            size.height,
        ));

        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(TransformOperator::new(TransformMode::Translate))));
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(TransformOperator::new(TransformMode::Rotate))));
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(TransformOperator::new(TransformMode::Scale))));
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(SelectionOperator::new())));
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(NavigationOperator::new())));

        window.request_redraw();

        self.window = Some(window);
        self.viewer = Some(viewer);
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Initialize on first resume
        if self.window.is_none() {
            self.initialize(event_loop);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let viewer = self.viewer.as_mut().unwrap();

                // Dispatch Update event for continuous operations (WASD movement, etc.)
                viewer.update();

                // Render the scene
                if let Err(e) = viewer.render() {
                    log::error!("Render error: {}", e);
                }

                // Request next frame for continuous rendering
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }

        // Convert and handle all window events
        if let Some(app_event) = winit_support::convert_window_event(event) {
            let viewer = self.viewer.as_mut().unwrap();
            viewer.handle_event(&app_event);

            // Check for exit on Escape key
            if let duck_engine_viewer::event::Event::Device(
                duck_engine_viewer::event::DeviceEvent::KeyboardInput { event: key_event, .. },
            ) = &app_event
            {
                if matches!(
                    key_event.logical_key,
                    duck_engine_viewer::input::Key::Named(duck_engine_viewer::input::NamedKey::Escape)
                ) {
                    event_loop.exit();
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // Convert and handle device events (e.g., mouse motion)
        if let Some(app_event) = winit_support::convert_device_event(event) {
            self.viewer.as_mut().unwrap().handle_event(&app_event);
        }
    }
}

fn main() {
    // Initialize logging
    env_logger::init();

    // Create event loop
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    // Create application state
    let mut app = App {
        window: None,
        viewer: None,
    };

    // Run the event loop
    event_loop.run_app(&mut app).unwrap();
}
