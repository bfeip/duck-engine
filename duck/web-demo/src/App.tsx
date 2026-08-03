import { useCallback, useRef, useState } from "react";
import { Viewer } from "./components/Viewer";
import { LoadingOverlay } from "./components/LoadingOverlay";
import { WebLoadPhase, type WebViewer } from "../pkg/duck_engine_viewer";
import "./App.css";

export function App() {
  const viewerRef = useRef<WebViewer | null>(null);
  const [sceneInfo, setSceneInfo] = useState({ nodes: 0, meshes: 0 });
  const [loading, setLoading] = useState<{ pct: number; phase: WebLoadPhase } | null>(null);

  const handleViewerReady = useCallback(async (viewer: WebViewer) => {
    viewerRef.current = viewer;
    try {
      const resp = await fetch("/default-scene.glb");
      if (resp.ok) {
        const data = new Uint8Array(await resp.arrayBuffer());
        viewer.start_load(data);
        setLoading({ pct: 0, phase: WebLoadPhase.Pending });
      }
    } catch (e) {
      console.warn("Failed to load default scene:", e);
    }
  }, []);

  const handleLoadProgress = useCallback((pct: number, phase: WebLoadPhase) => {
    setLoading({ pct, phase });
  }, []);

  const handleLoadComplete = useCallback((success: boolean) => {
    setLoading(null);
    if (success) {
      const viewer = viewerRef.current;
      if (viewer) {
        setSceneInfo({
          nodes: viewer.node_count(),
          meshes: viewer.mesh_count(),
        });
      }
    }
  }, []);

  const handleLoadFile = useCallback(async () => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".gltf,.glb";
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      const data = new Uint8Array(await file.arrayBuffer());
      viewer.start_load(data);
      setLoading({ pct: 0, phase: 0 });
    };
    input.click();
  }, []);

  const handleClear = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.clear_scene();
    setSceneInfo({ nodes: 0, meshes: 0 });
  }, []);

  return (
    <div className="app">
      <div className="sidebar">
        <h1>Duck Engine</h1>

        <div className="controls">
          <button onClick={handleLoadFile}>Load Scene</button>
          <button onClick={handleClear}>Clear</button>
        </div>

        {sceneInfo.meshes > 0 && (
          <div className="scene-info">
            <h2>Scene</h2>
            <p>Nodes: {sceneInfo.nodes}</p>
            <p>Meshes: {sceneInfo.meshes}</p>
          </div>
        )}
      </div>
      <div className="viewer-area">
        <Viewer
          onReady={handleViewerReady}
          onLoadProgress={handleLoadProgress}
          onLoadComplete={handleLoadComplete}
        />
        {loading && <LoadingOverlay pct={loading.pct} phase={loading.phase} />}
      </div>
    </div>
  );
}
