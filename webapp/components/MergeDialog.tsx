"use client";

import { useState, useEffect } from "react";

interface NearbyProject {
  id: string;
  project_name: string;
  capacity_mw: number;
  status: string;
  latitude: number;
  longitude: number;
  match_confidence: string | null;
  has_polygon: boolean;
}

interface NearbyGrwFeature {
  id: number;
  fid: number;
  country: string;
  centroid_lat: number;
  centroid_lon: number;
  area_m2: number;
  construction_year: number | null;
  user_name: string | null;
}

interface MergeDialogProps {
  mode: "grw_to_gem" | "gem_to_grw";
  sourceLat: number;
  sourceLon: number;
  sourceLabel: string;
  sourceId: string | number;
  onMerge: (targetId: string | number) => void;
  onCancel: () => void;
  onPreview: (lat: number, lon: number) => void;
}

export default function MergeDialog({
  mode,
  sourceLat,
  sourceLon,
  sourceLabel,
  sourceId,
  onMerge,
  onCancel,
  onPreview,
}: MergeDialogProps) {
  const [candidates, setCandidates] = useState<(NearbyProject | NearbyGrwFeature)[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCandidate, setSelectedCandidate] = useState<string | number | null>(null);
  const [radiusKm, setRadiusKm] = useState(20);

  useEffect(() => {
    const fetchNearby = async () => {
      setLoading(true);
      const type = mode === "grw_to_gem" ? "gem" : "grw";
      try {
        const res = await fetch(
          `/api/nearby?lat=${sourceLat}&lon=${sourceLon}&radius_km=${radiusKm}&type=${type}`
        );
        const data = await res.json();
        setCandidates(data.results || []);
      } catch (err) {
        console.error("Failed to fetch nearby:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchNearby();
  }, [sourceLat, sourceLon, radiusKm, mode]);

  return (
    <div className="bg-white border-t border-gray-200 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-purple-800">
          {mode === "grw_to_gem" ? "Link to GEM Project" : "Find GRW Polygon"}
        </h4>
        <button onClick={onCancel} className="text-gray-400 hover:text-gray-600 text-xs">
          Cancel
        </button>
      </div>

      <div className="text-xs text-gray-500">
        Source: <span className="font-medium">{sourceLabel}</span> (ID: {String(sourceId)})
      </div>

      <div className="flex items-center gap-2">
        <label className="text-xs text-gray-500">Radius:</label>
        <select
          value={radiusKm}
          onChange={(e) => setRadiusKm(parseInt(e.target.value))}
          className="text-xs px-2 py-1 border border-gray-300 rounded"
        >
          <option value={5}>5 km</option>
          <option value={10}>10 km</option>
          <option value={20}>20 km</option>
          <option value={50}>50 km</option>
        </select>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-500" />
        </div>
      ) : candidates.length === 0 ? (
        <div className="text-xs text-gray-400 text-center py-4">
          No candidates found within {radiusKm} km
        </div>
      ) : (
        <div className="max-h-48 overflow-y-auto sidebar-scroll space-y-1">
          {candidates.map((c) => {
            const isProject = "project_name" in c;
            const id = isProject ? (c as NearbyProject).id : (c as NearbyGrwFeature).id;
            const label = isProject
              ? `${(c as NearbyProject).project_name} (${(c as NearbyProject).capacity_mw} MW)`
              : (c as NearbyGrwFeature).user_name || `GRW #${(c as NearbyGrwFeature).fid}`;
            const subtitle = isProject
              ? `${(c as NearbyProject).status} | Confidence: ${(c as NearbyProject).match_confidence || "none"}`
              : `${((c as NearbyGrwFeature).area_m2 / 10000).toFixed(1)} ha | ${(c as NearbyGrwFeature).construction_year || "?"}`;
            const lat = isProject ? (c as NearbyProject).latitude : (c as NearbyGrwFeature).centroid_lat;
            const lon = isProject ? (c as NearbyProject).longitude : (c as NearbyGrwFeature).centroid_lon;

            return (
              <button
                key={String(id)}
                onClick={() => {
                  setSelectedCandidate(id);
                  onPreview(lat, lon);
                }}
                className={`w-full text-left px-2 py-2 rounded text-xs border transition-colors ${
                  selectedCandidate === id
                    ? "border-purple-500 bg-purple-50"
                    : "border-gray-200 hover:bg-gray-50"
                }`}
              >
                <div className="font-medium truncate">{label}</div>
                <div className="text-gray-400">{subtitle}</div>
              </button>
            );
          })}
        </div>
      )}

      <button
        onClick={() => selectedCandidate !== null && onMerge(selectedCandidate)}
        disabled={selectedCandidate === null}
        className="w-full px-3 py-2 bg-purple-600 text-white text-xs font-medium rounded hover:bg-purple-700 disabled:opacity-50"
      >
        {selectedCandidate !== null ? "Link Selected" : "Select a candidate above"}
      </button>
    </div>
  );
}
