"use client";

import { useEffect, useRef, useCallback } from "react";
import type { Project, GrwFeature, GeoJSONGeometry, OverviewPoint } from "@/lib/types";

interface MapViewProps {
  selectedProject: Project | null;
  selectedGrwFeature: GrwFeature | null;
  editMode: "none" | "edit" | "draw";
  onPolygonEdited: (geometry: GeoJSONGeometry) => void;
  onPolygonDrawn: (geometry: GeoJSONGeometry) => void;
  showOverview: boolean;
  overviewPoints: OverviewPoint[];
  onOverviewPointClick: (point: OverviewPoint) => void;
}

// Polygon colors by source
const POLYGON_STYLES = {
  grw: { color: "#3b82f6", fillColor: "#3b82f6", fillOpacity: 0.15, weight: 2, dashArray: "6 4" },
  drawn: { color: "#ef4444", fillColor: "#ef4444", fillOpacity: 0.2, weight: 2 },
  grwFeature: { color: "#9333ea", fillColor: "#9333ea", fillOpacity: 0.2, weight: 2, dashArray: "6 4" },
};

// Overview point colors
const OVERVIEW_COLORS: Record<string, string> = {
  matched: "#22c55e",
  gem_only: "#f97316",
  grw_only: "#9333ea",
};

export default function MapView({
  selectedProject,
  selectedGrwFeature,
  editMode,
  onPolygonEdited,
  onPolygonDrawn,
  showOverview,
  overviewPoints,
  onOverviewPointClick,
}: MapViewProps) {
  // Use refs with generic types to avoid Leaflet type issues
  const mapRef = useRef<ReturnType<typeof import("leaflet")["map"]> | null>(null);
  const leafletRef = useRef<typeof import("leaflet") | null>(null);
  const polygonLayerRef = useRef<unknown>(null);
  const markerRef = useRef<unknown>(null);
  const drawControlRef = useRef<unknown>(null);
  const editableLayerRef = useRef<unknown>(null);
  const overviewLayerRef = useRef<unknown>(null);
  const legendRef = useRef<unknown>(null);
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
        preferCanvas: true,
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

  // Update polygons when project or GRW feature changes
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

    if (selectedProject) {
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

      if (selectedProject.merged_polygon) {
        L.geoJSON(selectedProject.merged_polygon as GeoJSON.GeoJsonObject, {
          style: POLYGON_STYLES.grw,
        }).addTo(polygonGroup);
      }

      if (selectedProject.grw_polygons && selectedProject.grw_polygons.length > 0) {
        for (const poly of selectedProject.grw_polygons) {
          L.geoJSON(poly as unknown as GeoJSON.GeoJsonObject, {
            style: { ...POLYGON_STYLES.grw, fillOpacity: 0.08, weight: 1 },
          }).addTo(polygonGroup);
        }
      }

      polygonGroup.addTo(map);
      polygonLayerRef.current = polygonGroup;

      const allLayers = L.featureGroup([marker, polygonGroup]);
      if (allLayers.getBounds().isValid()) {
        map.fitBounds(allLayers.getBounds(), { padding: [50, 50], maxZoom: 16 });
      } else {
        map.setView([selectedProject.latitude, selectedProject.longitude], 14);
      }
    } else if (selectedGrwFeature) {
      // Show GRW feature polygon
      const marker = L.marker([
        selectedGrwFeature.centroid_lat,
        selectedGrwFeature.centroid_lon,
      ]).addTo(map);
      marker.bindPopup(
        `<b>${selectedGrwFeature.user_name || `GRW #${selectedGrwFeature.fid}`}</b><br>${
          selectedGrwFeature.area_m2 ? (selectedGrwFeature.area_m2 / 10000).toFixed(1) + " ha" : ""
        }`
      );
      markerRef.current = marker;

      const polygonGroup = L.featureGroup();

      if (selectedGrwFeature.polygon) {
        L.geoJSON(selectedGrwFeature.polygon as unknown as GeoJSON.GeoJsonObject, {
          style: POLYGON_STYLES.grwFeature,
        }).addTo(polygonGroup);
      }

      polygonGroup.addTo(map);
      polygonLayerRef.current = polygonGroup;

      const allLayers = L.featureGroup([marker, polygonGroup]);
      if (allLayers.getBounds().isValid()) {
        map.fitBounds(allLayers.getBounds(), { padding: [50, 50], maxZoom: 16 });
      } else {
        map.setView([selectedGrwFeature.centroid_lat, selectedGrwFeature.centroid_lon], 14);
      }
    }
  }, [selectedProject, selectedGrwFeature, cleanupDrawControl]);

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

  // Overview layer
  useEffect(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L) return;

    // Clean up existing overlay
    if (overviewLayerRef.current) {
      // @ts-expect-error dynamic Leaflet layer
      map.removeLayer(overviewLayerRef.current);
      overviewLayerRef.current = null;
    }
    if (legendRef.current) {
      // @ts-expect-error dynamic Leaflet control
      map.removeControl(legendRef.current);
      legendRef.current = null;
    }

    if (!showOverview || overviewPoints.length === 0) return;

    const overviewGroup = L.layerGroup();

    for (const point of overviewPoints) {
      const color = OVERVIEW_COLORS[point.type] || "#666";
      const circle = L.circleMarker([point.lat, point.lon], {
        radius: 4,
        fillColor: color,
        color: color,
        weight: 1,
        fillOpacity: 0.7,
        renderer: L.canvas(),
      });
      circle.bindTooltip(point.label, { direction: "top", offset: [0, -6] });
      circle.on("click", () => onOverviewPointClick(point));
      circle.addTo(overviewGroup);
    }

    overviewGroup.addTo(map);
    overviewLayerRef.current = overviewGroup;

    // Add legend
    const LegendControl = L.Control.extend({
      onAdd: () => {
        const div = L.DomUtil.create("div", "leaflet-control");
        div.style.cssText =
          "background:white;padding:8px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.3);font-size:11px;line-height:1.6;";
        div.innerHTML = `
          <div style="font-weight:600;margin-bottom:4px;">Overview</div>
          <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${OVERVIEW_COLORS.matched};margin-right:6px;vertical-align:middle;"></span>Matched (GEM+GRW)</div>
          <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${OVERVIEW_COLORS.gem_only};margin-right:6px;vertical-align:middle;"></span>GEM only</div>
          <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${OVERVIEW_COLORS.grw_only};margin-right:6px;vertical-align:middle;"></span>GRW only</div>
          <div style="color:#999;margin-top:4px;font-size:10px;">${overviewPoints.length.toLocaleString()} points</div>
        `;
        L.DomEvent.disableClickPropagation(div);
        return div;
      },
    });

    const legend = new LegendControl({ position: "bottomright" });
    legend.addTo(map);
    legendRef.current = legend;

    return () => {
      if (overviewLayerRef.current) {
        // @ts-expect-error dynamic Leaflet layer
        map.removeLayer(overviewLayerRef.current);
        overviewLayerRef.current = null;
      }
      if (legendRef.current) {
        // @ts-expect-error dynamic Leaflet control
        map.removeControl(legendRef.current);
        legendRef.current = null;
      }
    };
  }, [showOverview, overviewPoints, onOverviewPointClick]);

  return (
    <div ref={containerRef} className="w-full h-full" />
  );
}
