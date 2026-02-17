"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-draw";
import "leaflet-draw/dist/leaflet.draw.css";
import type {
  LabelingTask,
  AnnotationRegion,
  LabelingAnnotation,
} from "@/lib/types";
import { LULC_CLASSES } from "@/lib/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface TaskWithUrl extends LabelingTask {
  image_url: string;
  annotations: LabelingAnnotation[];
}

type GroupedTasks = Record<string, LabelingTask[]>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function groupBySite(tasks: LabelingTask[]): GroupedTasks {
  const grouped: GroupedTasks = {};
  for (const t of tasks) {
    const key = t.site_display_name || t.site_name;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(t);
  }
  return grouped;
}

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

function classColor(name: string): string {
  return LULC_CLASSES.find((c) => c.name === name)?.color || "#999999";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function LabelingApp() {
  // State
  const [tasks, setTasks] = useState<LabelingTask[]>([]);
  const [selectedTask, setSelectedTask] = useState<TaskWithUrl | null>(null);
  const [loading, setLoading] = useState(true);
  const [taskLoading, setTaskLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [annotator, setAnnotator] = useState("");
  const [activeClass, setActiveClass] = useState<string>(LULC_CLASSES[0].name);
  const [regions, setRegions] = useState<AnnotationRegion[]>([]);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string>("");

  // Refs
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const imageOverlayRef = useRef<L.ImageOverlay | null>(null);
  const drawLayerRef = useRef<L.FeatureGroup>(new L.FeatureGroup());
  const solarLayerRef = useRef<L.LayerGroup>(new L.LayerGroup());
  const drawControlRef = useRef<L.Control.Draw | null>(null);
  const activeClassRef = useRef(activeClass);
  const regionsRef = useRef(regions);

  // Keep refs in sync
  useEffect(() => {
    activeClassRef.current = activeClass;
  }, [activeClass]);
  useEffect(() => {
    regionsRef.current = regions;
  }, [regions]);

  // Load annotator from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("labeling_annotator");
    if (saved) setAnnotator(saved);
  }, []);

  // Persist annotator
  useEffect(() => {
    if (annotator) localStorage.setItem("labeling_annotator", annotator);
  }, [annotator]);

  // Fetch task list
  useEffect(() => {
    fetch("/api/labeling/tasks")
      .then((r) => r.json())
      .then((data) => {
        setTasks(data.tasks || []);
        setLoading(false);
      })
      .catch((e) => {
        console.error("Failed to load tasks:", e);
        setLoading(false);
      });
  }, []);

  // Initialize map
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current, {
      crs: L.CRS.Simple,
      minZoom: -2,
      maxZoom: 4,
      zoomControl: true,
      attributionControl: false,
    });

    drawLayerRef.current.addTo(map);
    solarLayerRef.current.addTo(map);

    const drawControl = new L.Control.Draw({
      position: "topleft",
      draw: {
        polygon: {
          allowIntersection: false,
          shapeOptions: {
            color: classColor(activeClassRef.current),
            weight: 2,
            fillOpacity: 0.3,
          },
        },
        polyline: false,
        rectangle: false,
        circle: false,
        circlemarker: false,
        marker: false,
      },
      edit: {
        featureGroup: drawLayerRef.current,
        remove: true,
      },
    });
    drawControl.addTo(map);
    drawControlRef.current = drawControl;

    // Handle new polygon drawn
    map.on(L.Draw.Event.CREATED, (e: any) => {
      const layer = e.layer as L.Polygon;
      const latlngs = (layer.getLatLngs()[0] as L.LatLng[]);
      const points: [number, number][] = latlngs.map((ll) => [ll.lng, ll.lat]);

      const regionId = uid();
      const className = activeClassRef.current;
      const color = classColor(className);

      // Style the layer
      layer.setStyle({
        color,
        fillColor: color,
        fillOpacity: 0.3,
        weight: 2,
      });
      (layer as L.Polygon & { regionId?: string }).regionId = regionId;

      // Add tooltip
      layer.bindTooltip(
        LULC_CLASSES.find((c) => c.name === className)?.label || className,
        { permanent: false, direction: "center" }
      );

      drawLayerRef.current.addLayer(layer);

      const newRegion: AnnotationRegion = {
        id: regionId,
        class_name: className,
        points,
      };
      setRegions((prev) => [...prev, newRegion]);
    });

    // Handle polygon deleted
    map.on(L.Draw.Event.DELETED, (e: any) => {
      const deletedIds: string[] = [];
      e.layers.eachLayer((layer: L.Layer) => {
        const rid = (layer as L.Polygon & { regionId?: string }).regionId;
        if (rid) deletedIds.push(rid);
      });
      if (deletedIds.length > 0) {
        setRegions((prev) => prev.filter((r) => !deletedIds.includes(r.id)));
      }
    });

    // Handle polygon edited
    map.on(L.Draw.Event.EDITED, (e: any) => {
      const updates: Record<string, [number, number][]> = {};
      e.layers.eachLayer((layer: L.Layer) => {
        const poly = layer as L.Polygon & { regionId?: string };
        if (poly.regionId) {
          const latlngs = (poly.getLatLngs()[0] as L.LatLng[]);
          updates[poly.regionId] = latlngs.map((ll) => [ll.lng, ll.lat]);
        }
      });
      setRegions((prev) =>
        prev.map((r) => (updates[r.id] ? { ...r, points: updates[r.id] } : r))
      );
    });

    // Click handler for selecting regions
    map.on("click", () => {
      setSelectedRegionId(null);
    });

    mapRef.current = map;
    map.setView([0, 0], 0);

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Update draw control color when active class changes
  useEffect(() => {
    if (!drawControlRef.current || !mapRef.current) return;
    mapRef.current.removeControl(drawControlRef.current);
    const newControl = new L.Control.Draw({
      position: "topleft",
      draw: {
        polygon: {
          allowIntersection: false,
          shapeOptions: {
            color: classColor(activeClass),
            weight: 2,
            fillOpacity: 0.3,
          },
        },
        polyline: false,
        rectangle: false,
        circle: false,
        circlemarker: false,
        marker: false,
      },
      edit: {
        featureGroup: drawLayerRef.current,
        remove: true,
      },
    });
    newControl.addTo(mapRef.current);
    drawControlRef.current = newControl;
  }, [activeClass]);

  // Load task detail + image
  const loadTask = useCallback(async (taskId: number) => {
    setTaskLoading(true);
    setSaveStatus("");
    try {
      const res = await fetch(`/api/labeling/tasks/${taskId}`);
      const data: TaskWithUrl = await res.json();
      setSelectedTask(data);

      // Clear existing layers
      drawLayerRef.current.clearLayers();
      solarLayerRef.current.clearLayers();

      if (!mapRef.current) return;
      const map = mapRef.current;

      // Remove old image overlay
      if (imageOverlayRef.current) {
        map.removeLayer(imageOverlayRef.current);
      }

      // Load image
      const bounds: L.LatLngBoundsExpression = [
        [0, 0],
        [data.image_height, data.image_width],
      ];
      const overlay = L.imageOverlay(data.image_url, bounds);
      overlay.addTo(map);
      imageOverlayRef.current = overlay;
      map.fitBounds(bounds);

      // Draw solar polygon reference (if post-construction)
      if (data.solar_polygon_pixels) {
        for (const ring of data.solar_polygon_pixels) {
          const latlngs: L.LatLngExpression[] = ring.map(
            ([x, y]: number[]) => [y, x] as L.LatLngExpression
          );
          const poly = L.polygon(latlngs, {
            color: "#ff0000",
            weight: 2,
            fillOpacity: 0,
            dashArray: "6 4",
            interactive: false,
          });
          poly.bindTooltip("Solar Installation", {
            permanent: false,
            direction: "center",
          });
          solarLayerRef.current.addLayer(poly);
        }
      }

      // Load existing annotations
      if (data.annotations && data.annotations.length > 0) {
        const ann = data.annotations[0]; // Most recent
        const loadedRegions: AnnotationRegion[] = ann.regions || [];
        setRegions(loadedRegions);

        // Render regions on map
        for (const region of loadedRegions) {
          const color = classColor(region.class_name);
          const latlngs: L.LatLngExpression[] = region.points.map(
            ([x, y]) => [y, x] as L.LatLngExpression
          );
          const poly = L.polygon(latlngs, {
            color,
            fillColor: color,
            fillOpacity: 0.3,
            weight: 2,
          });
          (poly as L.Polygon & { regionId?: string }).regionId = region.id;
          poly.bindTooltip(
            LULC_CLASSES.find((c) => c.name === region.class_name)?.label ||
              region.class_name,
            { permanent: false, direction: "center" }
          );
          poly.on("click", (e) => {
            L.DomEvent.stopPropagation(e);
            setSelectedRegionId(region.id);
          });
          drawLayerRef.current.addLayer(poly);
        }
      } else {
        setRegions([]);
      }
    } catch (e) {
      console.error("Failed to load task:", e);
    } finally {
      setTaskLoading(false);
    }
  }, []);

  // Save annotations
  const handleSave = useCallback(async () => {
    if (!selectedTask || !annotator.trim()) {
      setSaveStatus("Please enter your name first");
      return;
    }
    setSaving(true);
    setSaveStatus("");
    try {
      const res = await fetch(
        `/api/labeling/tasks/${selectedTask.id}/annotations`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            annotator: annotator.trim(),
            regions,
          }),
        }
      );
      if (res.ok) {
        setSaveStatus("Saved!");
        // Update annotation count in task list
        setTasks((prev) =>
          prev.map((t) =>
            t.id === selectedTask.id
              ? { ...t, annotation_count: (t.annotation_count || 0) + 1 }
              : t
          )
        );
        setTimeout(() => setSaveStatus(""), 2000);
      } else {
        setSaveStatus("Save failed");
      }
    } catch {
      setSaveStatus("Save failed");
    } finally {
      setSaving(false);
    }
  }, [selectedTask, annotator, regions]);

  // Change class of selected region
  const changeRegionClass = useCallback(
    (newClass: string) => {
      if (!selectedRegionId) return;
      const color = classColor(newClass);
      setRegions((prev) =>
        prev.map((r) =>
          r.id === selectedRegionId ? { ...r, class_name: newClass } : r
        )
      );
      // Update layer style
      drawLayerRef.current.eachLayer((layer) => {
        const poly = layer as L.Polygon & { regionId?: string };
        if (poly.regionId === selectedRegionId) {
          poly.setStyle({ color, fillColor: color });
          poly.unbindTooltip();
          poly.bindTooltip(
            LULC_CLASSES.find((c) => c.name === newClass)?.label || newClass,
            { permanent: false, direction: "center" }
          );
        }
      });
      setSelectedRegionId(null);
    },
    [selectedRegionId]
  );

  // Keyboard shortcut: 1-9,0 for classes
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      const idx =
        e.key === "0" ? 9 : parseInt(e.key) - 1;
      if (idx >= 0 && idx < LULC_CLASSES.length) {
        const cls = LULC_CLASSES[idx].name;
        if (selectedRegionId) {
          changeRegionClass(cls);
        } else {
          setActiveClass(cls);
        }
      }
      if (e.key === "s" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleSave();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedRegionId, changeRegionClass, handleSave]);

  const grouped = groupBySite(tasks);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "system-ui" }}>
      {/* Left sidebar: Task list */}
      <div
        style={{
          width: 280,
          borderRight: "1px solid #e0e0e0",
          overflowY: "auto",
          background: "#fafafa",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            padding: "12px 16px",
            borderBottom: "1px solid #e0e0e0",
            background: "#fff",
          }}
        >
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>
            LULC Labeling
          </h2>
          <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
            {tasks.length} tasks &middot;{" "}
            {tasks.filter((t) => (t.annotation_count || 0) > 0).length} labeled
          </div>
        </div>

        {/* Annotator name */}
        <div style={{ padding: "8px 16px", borderBottom: "1px solid #e0e0e0" }}>
          <input
            type="text"
            placeholder="Your name"
            value={annotator}
            onChange={(e) => setAnnotator(e.target.value)}
            style={{
              width: "100%",
              padding: "6px 8px",
              border: "1px solid #ccc",
              borderRadius: 4,
              fontSize: 13,
              boxSizing: "border-box",
            }}
          />
        </div>

        {loading ? (
          <div style={{ padding: 16, color: "#666" }}>Loading tasks...</div>
        ) : (
          Object.entries(grouped).map(([siteName, siteTasks]) => (
            <div key={siteName}>
              <div
                style={{
                  padding: "8px 16px 4px",
                  fontSize: 12,
                  fontWeight: 600,
                  color: "#444",
                  textTransform: "uppercase",
                  letterSpacing: "0.5px",
                }}
              >
                {siteName}
              </div>
              {siteTasks.map((task) => (
                <div
                  key={task.id}
                  onClick={() => loadTask(task.id)}
                  style={{
                    padding: "6px 16px 6px 24px",
                    cursor: "pointer",
                    fontSize: 13,
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    background:
                      selectedTask?.id === task.id ? "#e3f2fd" : "transparent",
                    borderLeft:
                      selectedTask?.id === task.id
                        ? "3px solid #1976d2"
                        : "3px solid transparent",
                  }}
                >
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      background:
                        (task.annotation_count || 0) > 0
                          ? "#4caf50"
                          : "#e0e0e0",
                      flexShrink: 0,
                    }}
                  />
                  <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {task.buffer_km}km &middot; {task.year}
                    {task.month ? `/${String(task.month).padStart(2, "0")}` : ""}
                    &middot;{" "}
                    <span
                      style={{
                        color:
                          task.period === "post"
                            ? "#d32f2f"
                            : task.period === "pre"
                            ? "#1976d2"
                            : "#666",
                        fontWeight: 500,
                      }}
                    >
                      {task.period}
                    </span>
                  </span>
                </div>
              ))}
            </div>
          ))
        )}

        {/* Export link */}
        <div
          style={{
            padding: "12px 16px",
            borderTop: "1px solid #e0e0e0",
            marginTop: "auto",
          }}
        >
          <a
            href="/api/labeling/export"
            target="_blank"
            style={{ fontSize: 12, color: "#1976d2" }}
          >
            Export all annotations (JSON)
          </a>
        </div>
      </div>

      {/* Center: Map/Image */}
      <div style={{ flex: 1, position: "relative" }}>
        {taskLoading && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: "rgba(255,255,255,0.8)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 1000,
              fontSize: 14,
            }}
          >
            Loading image...
          </div>
        )}
        {!selectedTask && !taskLoading && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#999",
              fontSize: 14,
              zIndex: 500,
            }}
          >
            Select a task from the sidebar to begin labeling
          </div>
        )}
        <div
          ref={mapContainerRef}
          style={{ width: "100%", height: "100%", background: "#1a1a1a" }}
        />

        {/* Save bar */}
        {selectedTask && (
          <div
            style={{
              position: "absolute",
              bottom: 16,
              left: "50%",
              transform: "translateX(-50%)",
              zIndex: 1000,
              display: "flex",
              alignItems: "center",
              gap: 8,
              background: "#fff",
              padding: "8px 16px",
              borderRadius: 8,
              boxShadow: "0 2px 12px rgba(0,0,0,0.15)",
            }}
          >
            <span style={{ fontSize: 13, color: "#666" }}>
              {regions.length} region{regions.length !== 1 ? "s" : ""}
            </span>
            <button
              onClick={handleSave}
              disabled={saving || !annotator.trim()}
              style={{
                padding: "6px 16px",
                background: saving ? "#ccc" : "#1976d2",
                color: "#fff",
                border: "none",
                borderRadius: 4,
                cursor: saving ? "default" : "pointer",
                fontSize: 13,
                fontWeight: 500,
              }}
            >
              {saving ? "Saving..." : "Save (Ctrl+S)"}
            </button>
            {saveStatus && (
              <span
                style={{
                  fontSize: 12,
                  color: saveStatus === "Saved!" ? "#4caf50" : "#d32f2f",
                }}
              >
                {saveStatus}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Right sidebar: Class selector */}
      <div
        style={{
          width: 200,
          borderLeft: "1px solid #e0e0e0",
          overflowY: "auto",
          background: "#fafafa",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            padding: "12px 12px 8px",
            borderBottom: "1px solid #e0e0e0",
            background: "#fff",
          }}
        >
          <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600 }}>Classes</h3>
          <div style={{ fontSize: 11, color: "#888", marginTop: 2 }}>
            Keys 1-9, 0 to select
          </div>
        </div>

        {LULC_CLASSES.map((cls, idx) => (
          <div
            key={cls.name}
            onClick={() => {
              if (selectedRegionId) {
                changeRegionClass(cls.name);
              } else {
                setActiveClass(cls.name);
              }
            }}
            style={{
              padding: "8px 12px",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 8,
              background: activeClass === cls.name ? "#e8f5e9" : "transparent",
              borderLeft:
                activeClass === cls.name
                  ? `3px solid ${cls.color}`
                  : "3px solid transparent",
            }}
          >
            <div
              style={{
                width: 16,
                height: 16,
                borderRadius: 3,
                background: cls.color,
                border: cls.name === "snow" || cls.name === "no_data" ? "1px solid #ccc" : "none",
                flexShrink: 0,
              }}
            />
            <span style={{ fontSize: 13, flex: 1 }}>{cls.label}</span>
            <span style={{ fontSize: 11, color: "#aaa" }}>
              {idx === 9 ? "0" : idx + 1}
            </span>
          </div>
        ))}

        {/* Selected region info */}
        {selectedRegionId && (
          <div
            style={{
              padding: "12px",
              borderTop: "1px solid #e0e0e0",
              background: "#fff3e0",
            }}
          >
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
              Selected Region
            </div>
            <div style={{ fontSize: 12, color: "#666" }}>
              Class:{" "}
              {LULC_CLASSES.find(
                (c) =>
                  c.name ===
                  regions.find((r) => r.id === selectedRegionId)?.class_name
              )?.label || "?"}
            </div>
            <div style={{ fontSize: 11, color: "#999", marginTop: 4 }}>
              Click a class above to reassign
            </div>
          </div>
        )}

        {/* Task info */}
        {selectedTask && (
          <div
            style={{
              padding: "12px",
              borderTop: "1px solid #e0e0e0",
              background: "#fff",
            }}
          >
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
              Current Task
            </div>
            <div style={{ fontSize: 12, color: "#666" }}>
              {selectedTask.site_display_name || selectedTask.site_name}
            </div>
            <div style={{ fontSize: 12, color: "#666" }}>
              {selectedTask.buffer_km}km &middot; {selectedTask.year}
              {selectedTask.month
                ? `/${String(selectedTask.month).padStart(2, "0")}`
                : ""}{" "}
              &middot; {selectedTask.period}
            </div>
            <div style={{ fontSize: 12, color: "#666" }}>
              {selectedTask.image_width} x {selectedTask.image_height} px
            </div>
            {selectedTask.solar_polygon_pixels && (
              <div
                style={{
                  fontSize: 11,
                  color: "#d32f2f",
                  marginTop: 4,
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                }}
              >
                <span
                  style={{
                    display: "inline-block",
                    width: 12,
                    height: 2,
                    background: "#ff0000",
                    borderTop: "1px dashed #ff0000",
                  }}
                />
                Solar polygon shown
              </div>
            )}
          </div>
        )}

        {/* Instructions */}
        <div style={{ padding: "12px", borderTop: "1px solid #e0e0e0" }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
            Instructions
          </div>
          <ol
            style={{
              fontSize: 11,
              color: "#666",
              margin: 0,
              paddingLeft: 16,
              lineHeight: 1.6,
            }}
          >
            <li>Select a class (right panel)</li>
            <li>Click polygon tool (left toolbar)</li>
            <li>Click points to draw boundary</li>
            <li>Double-click to finish polygon</li>
            <li>Click existing polygon to select it</li>
            <li>Click a class to reassign</li>
            <li>Save when done (Ctrl+S)</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
