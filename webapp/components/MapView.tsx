"use client";

import { useEffect, useRef, useCallback } from "react";
import type { Project, GeoJSONGeometry } from "@/lib/types";

interface MapViewProps {
  selectedProject: Project | null;
  editMode: "none" | "edit" | "draw";
  onPolygonEdited: (geometry: GeoJSONGeometry) => void;
  onPolygonDrawn: (geometry: GeoJSONGeometry) => void;
}

// Polygon colors by source
const POLYGON_STYLES = {
  grw: { color: "#3b82f6", fillColor: "#3b82f6", fillOpacity: 0.15, weight: 2, dashArray: "6 4" },
  drawn: { color: "#ef4444", fillColor: "#ef4444", fillOpacity: 0.2, weight: 2 },
};

export default function MapView({
  selectedProject,
  editMode,
  onPolygonEdited,
  onPolygonDrawn,
}: MapViewProps) {
  // Use refs with generic types to avoid Leaflet type issues
  const mapRef = useRef<ReturnType<typeof import("leaflet")["map"]> | null>(null);
  const leafletRef = useRef<typeof import("leaflet") | null>(null);
  const polygonLayerRef = useRef<unknown>(null);
  const markerRef = useRef<unknown>(null);
  const drawControlRef = useRef<unknown>(null);
  const editableLayerRef = useRef<unknown>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize map
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return;

    const initMap = async () => {
      const L = await import("leaflet");
      await import("leaflet-draw");
      leafletRef.current = L;

      // Fix default icon paths
      // @ts-expect-error Leaflet internal
      delete L.Icon.Default.prototype._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
        iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
        shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
      });

      const map = L.map(containerRef.current!, {
        center: [23.5, 80],
        zoom: 5,
        zoomControl: true,
      });

      // Google Satellite tiles
      L.tileLayer("https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", {
        attribution: "&copy; Google",
        maxZoom: 20,
      }).addTo(map);

      // Labels overlay
      L.tileLayer("https://mt1.google.com/vt/lyrs=h&x={x}&y={y}&z={z}", {
        maxZoom: 20,
        pane: "overlayPane",
      }).addTo(map);

      mapRef.current = map;
    };

    initMap();

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  // Clean up draw control helper
  const cleanupDrawControl = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    if (drawControlRef.current) {
      // @ts-expect-error dynamic Leaflet control
      map.removeControl(drawControlRef.current);
      drawControlRef.current = null;
    }
    if (editableLayerRef.current) {
      // @ts-expect-error dynamic Leaflet layer
      map.removeLayer(editableLayerRef.current);
      editableLayerRef.current = null;
    }
  }, []);

  // Update polygons when project changes
  useEffect(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L) return;

    // Clear existing layers
    if (polygonLayerRef.current) {
      // @ts-expect-error dynamic Leaflet layer
      map.removeLayer(polygonLayerRef.current);
      polygonLayerRef.current = null;
    }
    if (markerRef.current) {
      // @ts-expect-error dynamic Leaflet layer
      map.removeLayer(markerRef.current);
      markerRef.current = null;
    }
    cleanupDrawControl();

    if (!selectedProject) return;

    // Add marker for GSPT coordinates
    const marker = L.marker([
      selectedProject.latitude,
      selectedProject.longitude,
    ]).addTo(map);
    marker.bindPopup(
      `<b>${selectedProject.project_name}</b><br>${selectedProject.capacity_mw} MW`
    );
    markerRef.current = marker;

    // Add polygon layers
    const polygonGroup = L.featureGroup();

    // Show merged polygon if available
    if (selectedProject.merged_polygon) {
      L.geoJSON(selectedProject.merged_polygon as GeoJSON.GeoJsonObject, {
        style: POLYGON_STYLES.grw,
      }).addTo(polygonGroup);
    }

    // Also show individual GRW polygons if available
    if (selectedProject.grw_polygons && selectedProject.grw_polygons.length > 0) {
      for (const poly of selectedProject.grw_polygons) {
        L.geoJSON(poly as unknown as GeoJSON.GeoJsonObject, {
          style: { ...POLYGON_STYLES.grw, fillOpacity: 0.08, weight: 1 },
        }).addTo(polygonGroup);
      }
    }

    polygonGroup.addTo(map);
    polygonLayerRef.current = polygonGroup;

    // Fit bounds to show everything
    const allLayers = L.featureGroup([marker, polygonGroup]);
    if (allLayers.getBounds().isValid()) {
      map.fitBounds(allLayers.getBounds(), { padding: [50, 50], maxZoom: 16 });
    } else {
      map.setView([selectedProject.latitude, selectedProject.longitude], 14);
    }
  }, [selectedProject, cleanupDrawControl]);

  // Handle edit/draw mode
  useEffect(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L) return;

    cleanupDrawControl();

    if (editMode === "none") return;

    const editableLayer = new L.FeatureGroup();
    map.addLayer(editableLayer);
    editableLayerRef.current = editableLayer;

    if (editMode === "edit" && selectedProject?.merged_polygon) {
      // Add existing polygon to editable layer
      const existingGeoJson = L.geoJSON(selectedProject.merged_polygon as GeoJSON.GeoJsonObject);
      existingGeoJson.eachLayer((layer: L.Layer) => {
        editableLayer.addLayer(layer);
      });

      const drawControl = new L.Control.Draw({
        edit: { featureGroup: editableLayer, remove: true },
        draw: {
          polygon: false,
          polyline: false,
          circle: false,
          rectangle: false,
          marker: false,
          circlemarker: false,
        },
      });
      map.addControl(drawControl);
      drawControlRef.current = drawControl;

      map.on(L.Draw.Event.EDITED, (e: L.LeafletEvent) => {
        const event = e as L.LeafletEvent & { layers: L.LayerGroup };
        event.layers.eachLayer((layer) => {
          const geojson = (layer as L.Polygon).toGeoJSON();
          onPolygonEdited(geojson.geometry as GeoJSONGeometry);
        });
      });
    } else if (editMode === "draw") {
      const drawControl = new L.Control.Draw({
        edit: { featureGroup: editableLayer, remove: true },
        draw: {
          polygon: { allowIntersection: false, shapeOptions: POLYGON_STYLES.drawn },
          polyline: false,
          circle: false,
          rectangle: false,
          marker: false,
          circlemarker: false,
        },
      });
      map.addControl(drawControl);
      drawControlRef.current = drawControl;

      map.on(L.Draw.Event.CREATED, (e: L.LeafletEvent) => {
        const event = e as L.LeafletEvent & { layer: L.Layer };
        editableLayer.addLayer(event.layer);
        const geojson = (event.layer as L.Polygon).toGeoJSON();
        onPolygonDrawn(geojson.geometry as GeoJSONGeometry);
      });
    }

    return () => {
      map.off(L.Draw.Event.CREATED);
      map.off(L.Draw.Event.EDITED);
    };
  }, [editMode, selectedProject, onPolygonEdited, onPolygonDrawn, cleanupDrawControl]);

  return (
    <div ref={containerRef} className="w-full h-full" />
  );
}
