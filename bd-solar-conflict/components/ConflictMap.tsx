"use client";

import dynamic from "next/dynamic";
import { SolarSite } from "@/lib/types";

const ConflictMapInner = dynamic(() => import("./ConflictMapInner"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[500px] w-full animate-pulse items-center justify-center rounded-[10px] border border-[var(--line)] bg-[var(--surface-muted)] sm:h-[600px]">
      <p className="text-[#728179]">Loading map...</p>
    </div>
  ),
});

interface ConflictMapProps {
  sites: SolarSite[];
  selectedSite: SolarSite | null;
  onSelectSite: (site: SolarSite) => void;
}

export default function ConflictMap({
  sites,
  selectedSite,
  onSelectSite,
}: ConflictMapProps) {
  return (
    <section className="bg-white py-14" id="map">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-semibold text-[var(--foreground)]">
          Solar Sites Across Bangladesh
        </h2>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          Each circle represents a solar installation. Red indicates documented
          land conflicts. Circle size is proportional to capacity. Click a site
          to explore its satellite data.
        </p>
        <div className="mt-6">
          <ConflictMapInner
            sites={sites}
            selectedSite={selectedSite}
            onSelectSite={onSelectSite}
          />
        </div>
      </div>
    </section>
  );
}
