"use client";

import type { GrwFeature } from "@/lib/types";

interface GrwFeatureListProps {
  features: GrwFeature[];
  selectedId: number | null;
  onSelect: (feature: GrwFeature) => void;
  loading: boolean;
  total: number;
  page: number;
  perPage: number;
  onPageChange: (page: number) => void;
}

function LinkedBadge({ linked }: { linked: boolean }) {
  return linked ? (
    <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-100 text-green-800 font-medium">
      Linked
    </span>
  ) : (
    <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-100 text-purple-800 font-medium">
      Unmatched
    </span>
  );
}

export default function GrwFeatureList({
  features,
  selectedId,
  onSelect,
  loading,
  total,
  page,
  perPage,
  onPageChange,
}: GrwFeatureListProps) {
  const totalPages = Math.ceil(total / perPage);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500" />
      </div>
    );
  }

  if (features.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500 text-sm">
        No GRW features found
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto sidebar-scroll">
        {features.map((feature) => {
          const areaHa = feature.area_m2 ? (feature.area_m2 / 10000).toFixed(1) : "?";
          const name = feature.user_name || `GRW #${feature.fid || feature.id}`;
          return (
            <button
              key={feature.id}
              onClick={() => onSelect(feature)}
              className={`w-full text-left px-3 py-2.5 border-b border-gray-100 hover:bg-purple-50 transition-colors ${
                selectedId === feature.id ? "bg-purple-50 border-l-2 border-l-purple-500" : ""
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium truncate">{name}</div>
                  <div className="text-xs text-gray-500 truncate">
                    {feature.country || "Unknown"} &middot; {areaHa} ha
                  </div>
                  <div className="text-xs text-gray-400 mt-0.5">
                    {feature.construction_year
                      ? `Built ${feature.construction_year}${feature.construction_quarter ? `Q${feature.construction_quarter}` : ""}`
                      : "Year unknown"}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1 flex-shrink-0">
                  <LinkedBadge linked={!!feature.linked_project_id} />
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between px-3 py-2 border-t border-gray-200 bg-gray-50 text-xs">
          <span className="text-gray-500">{total} features</span>
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
