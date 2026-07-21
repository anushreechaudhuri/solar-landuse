"use client";

import { useState, useEffect } from "react";
import { SolarSite } from "@/lib/types";
import Hero from "@/components/Hero";
import ConflictMap from "@/components/ConflictMap";
import SiteCardGrid from "@/components/SiteCardGrid";
import SiteDetail from "@/components/SiteDetail";
import CaseStudies from "@/components/CaseStudies";
import SatelliteFindings from "@/components/SatelliteFindings";
import Footer from "@/components/Footer";

export default function Home() {
  const [sites, setSites] = useState<SolarSite[]>([]);
  const [selectedSite, setSelectedSite] = useState<SolarSite | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/data/sites.json")
      .then((res) => res.json())
      .then((data: SolarSite[]) => {
        setSites(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load sites:", err);
        setLoading(false);
      });
  }, []);

  const handleSelectSite = (site: SolarSite) => {
    setSelectedSite(site);
  };

  const handleCloseSiteDetail = () => {
    setSelectedSite(null);
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--hero-bg)]">
        <div className="text-center">
          <div className="mx-auto h-12 w-12 animate-spin rounded-full border-4 border-[#d9b85e] border-t-transparent" />
          <p className="mt-4 text-sm text-[var(--hero-muted)]">Loading site data...</p>
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen">
      <Hero sites={sites} />
      <ConflictMap
        sites={sites}
        selectedSite={selectedSite}
        onSelectSite={handleSelectSite}
      />
      <SiteCardGrid sites={sites} onSelect={handleSelectSite} />
      {selectedSite && (
        <SiteDetail site={selectedSite} onClose={handleCloseSiteDetail} />
      )}
      <CaseStudies sites={sites} />
      <SatelliteFindings />
      <Footer />
    </main>
  );
}
