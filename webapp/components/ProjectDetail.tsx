"use client";

import { useState } from "react";
import type { Project, Review } from "@/lib/types";

interface ProjectDetailProps {
  project: Project;
  reviews: Review[];
  onClose: () => void;
  onMergeClick?: () => void;
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="text-xs text-blue-600 hover:text-blue-800 underline"
      title={`Copy ${label}`}
    >
      {copied ? "Copied!" : label}
    </button>
  );
}

function toDMS(decimal: number, isLat: boolean): string {
  const abs = Math.abs(decimal);
  const deg = Math.floor(abs);
  const minFloat = (abs - deg) * 60;
  const min = Math.floor(minFloat);
  const sec = ((minFloat - min) * 60).toFixed(1);
  const dir = isLat ? (decimal >= 0 ? "N" : "S") : (decimal >= 0 ? "E" : "W");
  return `${deg}\u00B0${min}'${sec}"${dir}`;
}

export default function ProjectDetail({
  project,
  reviews,
  onClose,
  onMergeClick,
}: ProjectDetailProps) {
  const coordStr = `${project.latitude}, ${project.longitude}`;
  const dmsStr = `${toDMS(project.latitude, true)} ${toDMS(project.longitude, false)}`;

  return (
    <div className="bg-white border-t border-gray-200 overflow-y-auto sidebar-scroll" style={{ maxHeight: "50%" }}>
      {/* Header */}
      <div className="flex items-start justify-between p-3 border-b border-gray-100">
        <div>
          <h3 className="text-sm font-semibold">{project.project_name}</h3>
          {project.phase_name && project.phase_name !== "--" && (
            <div className="text-xs text-gray-500">{project.phase_name}</div>
          )}
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 p-1">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Metadata */}
      <div className="p-3 space-y-2 text-xs">
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <div>
            <span className="text-gray-500">Capacity:</span>{" "}
            <span className="font-medium">{project.capacity_mw} MW</span>
            {project.capacity_rating && (
              <span className="text-gray-400"> ({project.capacity_rating})</span>
            )}
          </div>
          <div>
            <span className="text-gray-500">Status:</span>{" "}
            <span className="font-medium capitalize">{project.status}</span>
          </div>
          <div>
            <span className="text-gray-500">Country:</span>{" "}
            <span className="font-medium">{project.country}</span>
          </div>
          {project.state_province && (
            <div>
              <span className="text-gray-500">State:</span>{" "}
              <span className="font-medium">{project.state_province}</span>
            </div>
          )}
          {project.start_year && (
            <div>
              <span className="text-gray-500">Start year:</span>{" "}
              <span className="font-medium">{project.start_year}</span>
            </div>
          )}
          {project.owner && (
            <div>
              <span className="text-gray-500">Owner:</span>{" "}
              <span className="font-medium">{project.owner}</span>
            </div>
          )}
        </div>

        {/* Coordinates */}
        <div className="bg-gray-50 rounded p-2">
          <div className="flex items-center justify-between">
            <span className="text-gray-500">Coordinates:</span>
            <div className="flex gap-2">
              <CopyButton text={coordStr} label="Copy DD" />
              <CopyButton text={dmsStr} label="Copy DMS" />
            </div>
          </div>
          <div className="font-mono text-[11px] mt-1">
            {coordStr}
          </div>
          <div className="font-mono text-[11px] text-gray-500">
            {dmsStr}
          </div>
          {project.location_accuracy && (
            <div className="text-[10px] text-gray-400 mt-0.5">
              Accuracy: {project.location_accuracy}
            </div>
          )}
        </div>

        {/* IDs */}
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500 mb-1">Identifiers:</div>
          <div className="font-mono text-[11px] space-y-0.5">
            <div>Phase: {project.id}</div>
            {project.gem_location_id && <div>Location: {project.gem_location_id}</div>}
            {project.other_ids && <div>Other: {project.other_ids}</div>}
          </div>
        </div>

        {/* Match info */}
        {project.match_confidence && (
          <div className="bg-gray-50 rounded p-2">
            <div className="text-gray-500 mb-1">GRW Match:</div>
            <div className="space-y-0.5">
              <div>
                Confidence:{" "}
                <span className={`font-medium ${
                  project.match_confidence === "high" ? "text-green-700" :
                  project.match_confidence === "medium" ? "text-yellow-700" :
                  "text-red-700"
                }`}>
                  {project.match_confidence}
                </span>
              </div>
              {project.match_distance_km != null && (
                <div>Distance: {project.match_distance_km.toFixed(2)} km</div>
              )}
              {project.grw_construction_date && (
                <div>Construction: {project.grw_construction_date}</div>
              )}
            </div>
          </div>
        )}

        {/* Find GRW polygon button (shown when no match) */}
        {onMergeClick && (
          <button
            onClick={onMergeClick}
            className="w-full px-3 py-2 bg-purple-600 text-white text-xs font-medium rounded hover:bg-purple-700"
          >
            Find GRW Polygon
          </button>
        )}

        {/* Wiki link */}
        {project.wiki_url && (
          <a
            href={project.wiki_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-800 underline block"
          >
            View on GEM Wiki
          </a>
        )}

        {/* Review history */}
        {reviews.length > 0 && (
          <div className="border-t border-gray-200 pt-2 mt-2">
            <div className="text-gray-500 font-medium mb-1">Review History:</div>
            {reviews.map((r) => (
              <div key={r.id} className="flex items-start gap-2 py-1 text-[11px]">
                <span className="text-gray-400 flex-shrink-0">
                  {new Date(r.created_at).toLocaleDateString()}
                </span>
                <span className="font-medium">{r.reviewer_name}</span>
                <span className="capitalize text-gray-600">{r.action.replace("_", " ")}</span>
                {r.notes && <span className="text-gray-400 truncate">- {r.notes}</span>}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
