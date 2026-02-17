"use client";

import type { Project } from "@/lib/types";

interface ProjectListProps {
  projects: Project[];
  selectedId: string | null;
  onSelect: (project: Project) => void;
  loading: boolean;
  total: number;
  page: number;
  perPage: number;
  onPageChange: (page: number) => void;
}

function ConfidenceBadge({ confidence }: { confidence: string | null }) {
  const styles: Record<string, string> = {
    high: "bg-green-100 text-green-800",
    medium: "bg-yellow-100 text-yellow-800",
    low: "bg-orange-100 text-orange-800",
    none: "bg-red-100 text-red-800",
  };
  const label = confidence || "none";
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${styles[label] || styles.none}`}>
      {label}
    </span>
  );
}

function ReviewBadge({ review }: { review: Project["latest_review"] }) {
  if (!review) return null;
  const actionLabels: Record<string, string> = {
    confirmed: "Confirmed",
    edited_polygon: "Edited",
    no_match: "No match",
    drawn_new: "New polygon",
    feasibility_yes: "Feasible",
    feasibility_no: "Not feasible",
    feasibility_maybe: "Maybe",
  };
  return (
    <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-800 font-medium">
      {actionLabels[review.action] || review.action}
    </span>
  );
}

export default function ProjectList({
  projects,
  selectedId,
  onSelect,
  loading,
  total,
  page,
  perPage,
  onPageChange,
}: ProjectListProps) {
  const totalPages = Math.ceil(total / perPage);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (projects.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500 text-sm">
        No projects found
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto sidebar-scroll">
        {projects.map((project) => (
          <button
            key={project.id}
            onClick={() => onSelect(project)}
            className={`w-full text-left px-3 py-2.5 border-b border-gray-100 hover:bg-blue-50 transition-colors ${
              selectedId === project.id ? "bg-blue-50 border-l-2 border-l-blue-500" : ""
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-medium truncate">
                  {project.project_name}
                </div>
                {project.phase_name && project.phase_name !== "--" && (
                  <div className="text-xs text-gray-500 truncate">
                    {project.phase_name}
                  </div>
                )}
                <div className="text-xs text-gray-400 mt-0.5">
                  {project.capacity_mw} MW &middot; {project.country}
                  {project.start_year ? ` &middot; ${project.start_year}` : ""}
                </div>
              </div>
              <div className="flex flex-col items-end gap-1 flex-shrink-0">
                <ConfidenceBadge confidence={project.match_confidence} />
                <ReviewBadge review={project.latest_review} />
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-3 py-2 border-t border-gray-200 bg-gray-50 text-xs">
          <span className="text-gray-500">
            {total} projects
          </span>
          <div className="flex gap-1">
            <button
              onClick={() => onPageChange(page - 1)}
              disabled={page <= 1}
              className="px-2 py-1 rounded border border-gray-300 disabled:opacity-30 hover:bg-white"
            >
              Prev
            </button>
            <span className="px-2 py-1 text-gray-600">
              {page}/{totalPages}
            </span>
            <button
              onClick={() => onPageChange(page + 1)}
              disabled={page >= totalPages}
              className="px-2 py-1 rounded border border-gray-300 disabled:opacity-30 hover:bg-white"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
