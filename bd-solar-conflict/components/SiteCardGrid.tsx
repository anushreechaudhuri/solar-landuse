"use client";

import { useState } from "react";
import { SolarSite } from "@/lib/types";
import SiteCard from "./SiteCard";

interface SiteCardGridProps {
  sites: SolarSite[];
  onSelect: (site: SolarSite) => void;
}

type FilterType = "all" | "conflict" | "no-conflict" | "proposed";

export default function SiteCardGrid({ sites, onSelect }: SiteCardGridProps) {
  const [filter, setFilter] = useState<FilterType>("all");

  const filteredSites = sites.filter((site) => {
    switch (filter) {
      case "conflict":
        return site.has_conflict && site.status !== "Proposed";
      case "no-conflict":
        return !site.has_conflict && site.status !== "Proposed";
      case "proposed":
        return site.status === "Proposed";
      default:
        return true;
    }
  });

  const filterButtons: { key: FilterType; label: string; count: number }[] = [
    { key: "all", label: "All", count: sites.length },
    {
      key: "conflict",
      label: "Conflict",
      count: sites.filter((s) => s.has_conflict && s.status !== "Proposed")
        .length,
    },
    {
      key: "no-conflict",
      label: "No Conflict",
      count: sites.filter((s) => !s.has_conflict && s.status !== "Proposed")
        .length,
    },
    {
      key: "proposed",
      label: "Proposed",
      count: sites.filter((s) => s.status === "Proposed").length,
    },
  ];

  return (
    <section className="bg-[var(--background)] py-16" id="sites">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-semibold text-[var(--foreground)]">All Sites</h2>
        <p className="mt-2 text-[var(--muted)]">
          Click any site to see satellite imagery, land cover data, and conflict
          details.
        </p>

        {/* Filter bar */}
        <div className="mt-6 inline-flex flex-wrap gap-1 rounded-lg border border-[var(--line)] bg-white p-1">
          {filterButtons.map((btn) => (
            <button
              type="button"
              key={btn.key}
              onClick={() => setFilter(btn.key)}
              className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                filter === btn.key
                  ? "bg-[var(--hero-bg)] text-white"
                  : "text-[var(--muted)] hover:bg-[var(--surface-muted)] hover:text-[var(--foreground)]"
              }`}
            >
              {btn.label}
              <span
                className={`ml-1.5 ${
                  filter === btn.key ? "text-white/60" : "text-[#7a8981]"
                }`}
              >
                ({btn.count})
              </span>
            </button>
          ))}
        </div>

        {/* Card grid */}
        <div className="mt-8 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {filteredSites.map((site) => (
            <SiteCard key={site.id} site={site} onSelect={onSelect} />
          ))}
        </div>

        {filteredSites.length === 0 && (
          <p className="mt-8 text-center text-[#728179]">
            No sites match the current filter.
          </p>
        )}
      </div>
    </section>
  );
}
