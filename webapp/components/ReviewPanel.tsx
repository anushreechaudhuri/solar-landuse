"use client";

import { useState, useEffect } from "react";
import type { Project, GeoJSONGeometry } from "@/lib/types";

interface ReviewPanelProps {
  project: Project;
  isActiveTab: boolean;
  editMode: "none" | "edit" | "draw";
  onEditModeChange: (mode: "none" | "edit" | "draw") => void;
  pendingPolygon: GeoJSONGeometry | null;
  onReviewSubmitted: () => void;
}

export default function ReviewPanel({
  project,
  isActiveTab,
  editMode,
  onEditModeChange,
  pendingPolygon,
  onReviewSubmitted,
}: ReviewPanelProps) {
  const [reviewerName, setReviewerName] = useState("");
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Load reviewer name from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("reviewer_name");
    if (saved) setReviewerName(saved);
  }, []);

  // Save reviewer name
  useEffect(() => {
    if (reviewerName) {
      localStorage.setItem("reviewer_name", reviewerName);
    }
  }, [reviewerName]);

  const submitReview = async (action: string, polygon?: GeoJSONGeometry | null) => {
    if (!reviewerName.trim()) {
      setMessage({ type: "error", text: "Please enter your name" });
      return;
    }

    setSubmitting(true);
    setMessage(null);

    try {
      const res = await fetch(`/api/projects/${project.id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reviewer_name: reviewerName.trim(),
          action,
          polygon: polygon || null,
          notes: notes.trim() || null,
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Failed to submit review");
      }

      // If polygon was edited/drawn, also update the project's merged_polygon
      if (polygon && (action === "edited_polygon" || action === "drawn_new")) {
        await fetch(`/api/projects/${project.id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            polygon: {
              type: "Feature",
              geometry: polygon,
              properties: { source: "user", reviewer: reviewerName.trim() },
            },
          }),
        });
      }

      setMessage({ type: "success", text: "Review saved!" });
      setNotes("");
      onEditModeChange("none");
      onReviewSubmitted();
    } catch (err) {
      setMessage({ type: "error", text: (err as Error).message });
    } finally {
      setSubmitting(false);
    }
  };

  const hasPolygon = !!project.merged_polygon;

  return (
    <div className="bg-white border-t border-gray-200 p-3 space-y-3">
      {/* Reviewer name */}
      <div>
        <label className="text-xs text-gray-500 block mb-1">Your name:</label>
        <input
          type="text"
          value={reviewerName}
          onChange={(e) => setReviewerName(e.target.value)}
          placeholder="Enter your name"
          className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>

      {/* Notes */}
      <div>
        <label className="text-xs text-gray-500 block mb-1">Notes (optional):</label>
        <input
          type="text"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Any notes..."
          className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm"
        />
      </div>

      {/* Action buttons — different for active vs proposed tab */}
      {isActiveTab ? (
        <div className="space-y-2">
          {/* Polygon actions */}
          {editMode === "none" ? (
            <div className="grid grid-cols-2 gap-2">
              {hasPolygon && (
                <button
                  onClick={() => submitReview("confirmed")}
                  disabled={submitting}
                  className="px-3 py-2 bg-green-600 text-white text-xs font-medium rounded hover:bg-green-700 disabled:opacity-50"
                >
                  Confirm Polygon
                </button>
              )}
              <button
                onClick={() => submitReview("no_match")}
                disabled={submitting}
                className="px-3 py-2 bg-red-600 text-white text-xs font-medium rounded hover:bg-red-700 disabled:opacity-50"
              >
                No Match
              </button>
              {hasPolygon && (
                <button
                  onClick={() => onEditModeChange("edit")}
                  className="px-3 py-2 bg-orange-500 text-white text-xs font-medium rounded hover:bg-orange-600"
                >
                  Edit Polygon
                </button>
              )}
              <button
                onClick={() => onEditModeChange("draw")}
                className="px-3 py-2 bg-blue-600 text-white text-xs font-medium rounded hover:bg-blue-700"
              >
                Draw New
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="text-xs text-gray-500 bg-yellow-50 p-2 rounded">
                {editMode === "edit"
                  ? "Edit the polygon on the map, then save."
                  : "Draw a new polygon on the map, then save."}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    if (pendingPolygon) {
                      submitReview(
                        editMode === "edit" ? "edited_polygon" : "drawn_new",
                        pendingPolygon
                      );
                    }
                  }}
                  disabled={submitting || !pendingPolygon}
                  className="flex-1 px-3 py-2 bg-green-600 text-white text-xs font-medium rounded hover:bg-green-700 disabled:opacity-50"
                >
                  {pendingPolygon ? "Save Polygon" : "Draw on map first..."}
                </button>
                <button
                  onClick={() => onEditModeChange("none")}
                  className="px-3 py-2 bg-gray-200 text-gray-700 text-xs font-medium rounded hover:bg-gray-300"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Feasibility actions for proposed/other tab */
        <div className="space-y-2">
          <div className="text-xs text-gray-500">
            Could this location feasibly host a utility-scale solar project?
          </div>
          <div className="grid grid-cols-3 gap-2">
            <button
              onClick={() => submitReview("feasibility_yes")}
              disabled={submitting}
              className="px-3 py-2 bg-green-600 text-white text-xs font-medium rounded hover:bg-green-700 disabled:opacity-50"
            >
              Yes
            </button>
            <button
              onClick={() => submitReview("feasibility_maybe")}
              disabled={submitting}
              className="px-3 py-2 bg-yellow-500 text-white text-xs font-medium rounded hover:bg-yellow-600 disabled:opacity-50"
            >
              Maybe
            </button>
            <button
              onClick={() => submitReview("feasibility_no")}
              disabled={submitting}
              className="px-3 py-2 bg-red-600 text-white text-xs font-medium rounded hover:bg-red-700 disabled:opacity-50"
            >
              No
            </button>
          </div>
        </div>
      )}

      {/* Status message */}
      {message && (
        <div
          className={`text-xs p-2 rounded ${
            message.type === "success"
              ? "bg-green-50 text-green-700"
              : "bg-red-50 text-red-700"
          }`}
        >
          {message.text}
        </div>
      )}
    </div>
  );
}
