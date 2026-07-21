"use client";

import { SolarSite } from "@/lib/types";

interface SiteCardProps {
  site: SolarSite;
  onSelect: (site: SolarSite) => void;
}

const TAG_COLORS: Record<string, string> = {
  "Forced Acquisition": "border-[#e2b7af] bg-[#f6e8e5] text-[#87372b]",
  "Three-Crop Land": "border-[#e5d099] bg-[#f6efd9] text-[#765a19]",
  "Farmer Livelihoods": "border-[#e5d099] bg-[#f6efd9] text-[#765a19]",
  "Ecological Impact": "border-[#afd0c1] bg-[#e7f0ec] text-[#285e49]",
  Corruption: "border-[#e2b7af] bg-[#f6e8e5] text-[#87372b]",
  "River Erosion": "border-[#b5cfda] bg-[#e6eff3] text-[#285f78]",
  "Inadequate Compensation": "border-[#e2b7af] bg-[#f6e8e5] text-[#87372b]",
  "Community Protests": "border-[#e2b7af] bg-[#f6e8e5] text-[#87372b]",
  "No Documented Conflict": "border-[#cad5ce] bg-[#edf1ee] text-[#53645b]",
};

function getBorderColor(site: SolarSite): string {
  if (site.status === "Proposed") return "border-[#bdc8c2]";
  if (site.has_conflict) return "border-[#d9a49b]";
  return "border-[#9fc0b0]";
}

function getStatusBadge(site: SolarSite) {
  if (site.status === "Proposed") {
    return (
      <span className="inline-flex items-center rounded-md border border-[#e5d099] bg-[#f6efd9] px-2.5 py-0.5 text-xs font-medium text-[#765a19]">
        Proposed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center rounded-md border border-[#afd0c1] bg-[#e7f0ec] px-2.5 py-0.5 text-xs font-medium text-[#285e49]">
      Operational
    </span>
  );
}

export default function SiteCard({ site, onSelect }: SiteCardProps) {
  const conflictPreview = site.has_conflict
    ? site.conflict_reasons.length > 100
      ? site.conflict_reasons.substring(0, 100) + "..."
      : site.conflict_reasons
    : "No documented conflict";

  return (
    <article
      className={`relative w-full rounded-[10px] border bg-white p-5 text-left transition-colors hover:border-[#7f998b] ${getBorderColor(
        site
      )}`}
    >
      <button
        type="button"
        onClick={() => onSelect(site)}
        className="absolute inset-0 z-10 cursor-pointer rounded-[10px]"
        aria-label={`View details for ${site.name}`}
      />
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-base font-semibold leading-tight text-[var(--foreground)]">
          {site.name}
        </h3>
        {site.capacity_mw && (
        <span className="inline-flex shrink-0 items-center rounded-md border border-[#b5cfda] bg-[#e6eff3] px-2.5 py-0.5 text-xs font-semibold text-[#285f78]">
          {site.capacity_mw} MW
        </span>
        )}
      </div>

      {/* Location */}
      {(site.district || site.upazilla) && (
      <p className="mt-1.5 text-sm text-[var(--muted)]">
        {[site.district, site.upazilla].filter(Boolean).join(", ")}
      </p>
      )}

      {/* Conflict preview */}
      <p className="mt-3 text-sm leading-relaxed text-[#46564e]">
        {conflictPreview}
      </p>

      {/* Tags */}
      <div className="mt-3 flex flex-wrap gap-1.5">
        {site.conflict_tags.map((tag) => (
          <span
            key={tag}
            className={`inline-flex items-center rounded-[4px] border px-2 py-0.5 text-xs font-medium ${
              TAG_COLORS[tag] || "border-[#cad5ce] bg-[#edf1ee] text-[#53645b]"
            }`}
          >
            {tag}
          </span>
        ))}
      </div>

      {/* Status */}
      <div className="mt-3 flex items-center justify-between">
        {getStatusBadge(site)}
        {site.completion_date && (
          <span className="text-xs text-[#728179]">
            {site.completion_date}
          </span>
        )}
      </div>
    </article>
  );
}
