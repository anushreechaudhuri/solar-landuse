"use client";

import { useEffect, useRef } from "react";
import L from "leaflet";
import { SolarSite } from "@/lib/types";

interface ConflictMapInnerProps {
  sites: SolarSite[];
  selectedSite: SolarSite | null;
  onSelectSite: (site: SolarSite) => void;
}

function getMarkerColor(site: SolarSite): string {
  if (site.status === "Proposed") return "#718078";
  if (site.has_conflict) return "#b54736";
  return "#39735a";
}

function getMarkerRadius(capacity_mw: number): number {
  return Math.max(6, Math.min(20, Math.sqrt(capacity_mw) * 1.5));
}

export default function ConflictMapInner({
  sites,
  selectedSite,
  onSelectSite,
}: ConflictMapInnerProps) {
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const markersRef = useRef<L.CircleMarker[]>([]);
  const polygonLayerRef = useRef<L.GeoJSON | null>(null);

  // Initialize the map
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: [23.8, 90.4],
      zoom: 7,
      zoomControl: true,
      scrollWheelZoom: true,
    });

    L.tileLayer("https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", {
      attribution: "Google Satellite",
      maxZoom: 18,
    }).addTo(map);

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Add markers
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Clear existing markers
    markersRef.current.forEach((m) => m.remove());
    markersRef.current = [];

    sites.forEach((site) => {
      if (site.lat === null || site.lon === null) return;

      const color = getMarkerColor(site);
      const radius = getMarkerRadius(site.capacity_mw);

      const marker = L.circleMarker([site.lat, site.lon], {
        radius,
        fillColor: color,
        fillOpacity: 0.7,
        color: "#ffffff",
        weight: 2,
        opacity: 0.9,
      }).addTo(map);

      // Popup
      const conflictStatus = site.has_conflict
        ? '<span style="color: #b54736; font-weight: 600;">Conflict documented</span>'
        : '<span style="color: #39735a; font-weight: 600;">No documented conflict</span>';

      marker.bindPopup(
        `<h3>${site.name}</h3>
         <p><strong>${site.capacity_mw} MW</strong> &mdash; ${site.district}</p>
         <p>${conflictStatus}</p>`,
        { closeButton: true, maxWidth: 250 }
      );

      marker.on("click", () => {
        onSelectSite(site);
      });

      markersRef.current.push(marker);
    });
  }, [sites, onSelectSite]);

  // Show polygon for selected site
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Remove previous polygon
    if (polygonLayerRef.current) {
      polygonLayerRef.current.remove();
      polygonLayerRef.current = null;
    }

    if (selectedSite?.polygon) {
      const geojson = L.geoJSON(selectedSite.polygon as GeoJSON.Polygon, {
        style: {
          color: "#b54736",
          weight: 2,
          fillColor: "#b54736",
          fillOpacity: 0.2,
        },
      }).addTo(map);

      polygonLayerRef.current = geojson;

      // Fly to the selected site
      if (selectedSite.lat && selectedSite.lon) {
        map.flyTo([selectedSite.lat, selectedSite.lon], 13, {
          duration: 1.5,
        });
      }
    } else if (selectedSite?.lat && selectedSite?.lon) {
      map.flyTo([selectedSite.lat, selectedSite.lon], 11, {
        duration: 1.5,
      });
    }
  }, [selectedSite]);

  return (
    <div className="relative">
      <div
        ref={containerRef}
        className="h-[500px] w-full overflow-hidden rounded-[10px] border border-[var(--line)] sm:h-[600px]"
      />
      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-[1000] rounded-md border border-[var(--line)] bg-white p-3 text-xs shadow-[0_8px_24px_rgba(23,37,31,0.16)]">
        <p className="mb-2 font-semibold text-[var(--foreground)]">Legend</p>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-full bg-[#b54736]" />
            <span className="text-[var(--muted)]">Conflict documented</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-full bg-[#39735a]" />
            <span className="text-[var(--muted)]">No documented conflict</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-full bg-[#718078]" />
            <span className="text-[var(--muted)]">Proposed</span>
          </div>
          <div className="mt-1.5 flex items-center gap-2 border-t border-[var(--line)] pt-1.5">
            <span className="text-[var(--muted)]">Circle size = capacity (MW)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
