"use client";

import { useState } from "react";
import type { GrwFeature, MergeHistoryEntry } from "@/lib/types";

interface GrwFeatureDetailProps {
  feature: GrwFeature & { merge_history?: MergeHistoryEntry[] };
  onClose: () => void;
  onUpdated: () => void;
  onMergeClick: () => void;
}

export default function GrwFeatureDetail({
  feature,
  onClose,
  onUpdated,
  onMergeClick,
}: GrwFeatureDetailProps) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(feature.user_name || "");
  const [capacity, setCapacity] = useState(feature.user_capacity_mw?.toString() || "");
  const [status, setStatus] = useState(feature.user_status || "");
  const [notes, setNotes] = useState(feature.user_notes || "");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const areaHa = feature.area_m2 ? (feature.area_m2 / 10000).toFixed(2) : "?";
  const displayName = feature.user_name || `GRW #${feature.fid || feature.id}`;

  const handleSave = async () => {
    setSaving(true);
    setMessage(null);
    try {
      const res = await fetch(`/api/grw-features/${feature.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_name: name.trim() || null,
          user_capacity_mw: capacity ? parseFloat(capacity) : null,
          user_status: status.trim() || null,
          user_notes: notes.trim() || null,
        }),
      });
      if (!res.ok) throw new Error("Failed to save");
      setMessage({ type: "success", text: "Saved!" });
      setEditing(false);
      onUpdated();
    } catch {
      setMessage({ type: "error", text: "Failed to save changes" });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="bg-white border-t border-gray-200 overflow-y-auto sidebar-scroll" style={{ maxHeight: "55%" }}>
      {/* Header */}
      <div className="flex items-start justify-between p-3 border-b border-gray-100">
        <div>
          <h3 className="text-sm font-semibold">{displayName}</h3>
          <div className="text-xs text-gray-500">FID: {feature.fid}</div>
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 p-1">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="p-3 space-y-2 text-xs">
        {/* Fixed metadata */}
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <div>
            <span className="text-gray-500">Country:</span>{" "}
            <span className="font-medium">{feature.country || "Unknown"}</span>
          </div>
          <div>
            <span className="text-gray-500">Area:</span>{" "}
            <span className="font-medium">{areaHa} ha</span>
          </div>
          {feature.construction_year && (
            <div>
              <span className="text-gray-500">Built:</span>{" "}
              <span className="font-medium">
                {feature.construction_year}
                {feature.construction_quarter ? `Q${feature.construction_quarter}` : ""}
              </span>
            </div>
          )}
          {feature.landcover && (
            <div>
              <span className="text-gray-500">Landcover:</span>{" "}
              <span className="font-medium">{feature.landcover}</span>
            </div>
          )}
        </div>

        {/* Coordinates */}
        <div className="bg-gray-50 rounded p-2">
          <span className="text-gray-500">Centroid:</span>{" "}
          <span className="font-mono text-[11px]">
            {feature.centroid_lat.toFixed(6)}, {feature.centroid_lon.toFixed(6)}
          </span>
        </div>

        {/* Editable fields */}
        {editing ? (
          <div className="bg-purple-50 rounded p-2 space-y-2">
            <div className="text-gray-600 font-medium">Edit Details</div>
            <div>
              <label className="text-gray-500 block mb-0.5">Name:</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Rajshahi Solar Park"
                className="w-full px-2 py-1 border border-gray-300 rounded text-xs"
              />
            </div>
            <div>
              <label className="text-gray-500 block mb-0.5">Capacity (MW):</label>
              <input
                type="number"
                value={capacity}
                onChange={(e) => setCapacity(e.target.value)}
                placeholder="e.g. 50"
                className="w-full px-2 py-1 border border-gray-300 rounded text-xs"
              />
            </div>
            <div>
              <label className="text-gray-500 block mb-0.5">Status:</label>
              <select
                value={status}
                onChange={(e) => setStatus(e.target.value)}
                className="w-full px-2 py-1 border border-gray-300 rounded text-xs"
              >
                <option value="">Unknown</option>
                <option value="operating">Operating</option>
                <option value="construction">Construction</option>
                <option value="decommissioned">Decommissioned</option>
              </select>
            </div>
            <div>
              <label className="text-gray-500 block mb-0.5">Notes:</label>
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Any notes..."
                className="w-full px-2 py-1 border border-gray-300 rounded text-xs"
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-3 py-1.5 bg-purple-600 text-white text-xs font-medium rounded hover:bg-purple-700 disabled:opacity-50"
              >
                {saving ? "Saving..." : "Save"}
              </button>
              <button
                onClick={() => setEditing(false)}
                className="px-3 py-1.5 bg-gray-200 text-gray-700 text-xs font-medium rounded hover:bg-gray-300"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="bg-gray-50 rounded p-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-gray-500 font-medium">User Details</span>
              <button
                onClick={() => setEditing(true)}
                className="text-xs text-purple-600 hover:text-purple-800 underline"
              >
                Edit
              </button>
            </div>
            {feature.user_name && <div>Name: <span className="font-medium">{feature.user_name}</span></div>}
            {feature.user_capacity_mw && <div>Capacity: <span className="font-medium">{feature.user_capacity_mw} MW</span></div>}
            {feature.user_status && <div>Status: <span className="font-medium capitalize">{feature.user_status}</span></div>}
            {feature.user_notes && <div>Notes: <span className="text-gray-600">{feature.user_notes}</span></div>}
            {!feature.user_name && !feature.user_capacity_mw && !feature.user_status && (
              <div className="text-gray-400 italic">No details added yet</div>
            )}
          </div>
        )}

        {/* Link status */}
        <div className="bg-gray-50 rounded p-2">
          {feature.linked_project_id ? (
            <div>
              <span className="text-green-700 font-medium">Linked to project:</span>{" "}
              <span className="font-mono text-[11px]">{feature.linked_project_id}</span>
            </div>
          ) : (
            <button
              onClick={onMergeClick}
              className="w-full px-3 py-2 bg-purple-600 text-white text-xs font-medium rounded hover:bg-purple-700"
            >
              Link to GEM Project
            </button>
          )}
        </div>

        {/* Merge history */}
        {feature.merge_history && feature.merge_history.length > 0 && (
          <div className="border-t border-gray-200 pt-2 mt-2">
            <div className="text-gray-500 font-medium mb-1">Merge History:</div>
            {feature.merge_history.map((h: MergeHistoryEntry) => (
              <div key={h.id} className="flex items-start gap-2 py-1 text-[11px]">
                <span className="text-gray-400 flex-shrink-0">
                  {new Date(h.created_at).toLocaleDateString()}
                </span>
                <span className="font-medium">{h.performed_by}</span>
                <span className="capitalize text-gray-600">{h.action}</span>
                {h.notes && <span className="text-gray-400 truncate">- {h.notes}</span>}
              </div>
            ))}
          </div>
        )}

        {message && (
          <div
            className={`text-xs p-2 rounded ${
              message.type === "success" ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"
            }`}
          >
            {message.text}
          </div>
        )}
      </div>
    </div>
  );
}
