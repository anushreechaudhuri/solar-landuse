"use client";

import { useEffect, useRef } from "react";
import Image from "next/image";
import { SolarSite } from "@/lib/types";
import LulcTimelineChart from "./LulcTimelineChart";
import LulcDonutChart from "./LulcDonutChart";

interface SiteDetailProps {
  site: SolarSite;
  onClose: () => void;
}

function getConstructionYear(site: SolarSite): number | null {
  if (!site.completion_date) return null;
  const year = parseInt(site.completion_date.split("-")[0], 10);
  return isNaN(year) ? null : year;
}

export default function SiteDetail({ site, onClose }: SiteDetailProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [site.id]);

  const constructionYear = getConstructionYear(site);

  return (
    <section
      ref={ref}
      className="border-y border-[var(--line)] bg-white py-12"
      id="site-detail"
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Close button */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-2xl font-semibold text-[var(--foreground)] sm:text-3xl">
              {site.name}
            </h2>
            <p className="mt-1 text-[var(--muted)]">
              {[site.district, site.upazilla].filter(Boolean).join(", ")}
              {site.capacity_mw ? ` — ${site.capacity_mw} MW` : ""}
              {site.status === "Proposed" ? " (Proposed)" : ""}
            </p>
          </div>
          <button
            onClick={onClose}
            className="ml-4 shrink-0 rounded-md border border-transparent p-2 transition-colors hover:border-[var(--line)] hover:bg-[var(--surface-muted)]"
            aria-label="Close site detail"
          >
            <svg
              className="h-6 w-6 text-[#728179]"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Two-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left: satellite image */}
          <div>
            {site.images?.pre_post ? (
              <div className="overflow-hidden rounded-[8px] border border-[var(--line)]">
                <Image
                  src={site.images.pre_post}
                  alt={`Pre and post construction satellite imagery of ${site.name}`}
                  width={800}
                  height={400}
                  className="w-full h-auto"
                  priority
                />
              </div>
            ) : (
              <div className="flex h-64 items-center justify-center rounded-[8px] border border-[var(--line)] bg-[var(--surface-muted)]">
                <p className="text-[#728179]">No satellite imagery available</p>
              </div>
            )}
          </div>

          {/* Right: info */}
          <div className="space-y-6">
            {/* Conflict details */}
            {site.has_conflict && (
              <div>
                <h3 className="mb-2 text-sm font-semibold uppercase tracking-[0.08em] text-[#9c3c30]">
                  Conflict Details
                </h3>
                <p className="leading-relaxed text-[#35473e]">
                  {site.conflict_reasons}
                </p>
              </div>
            )}

            {!site.has_conflict && (
              <div>
                <h3 className="mb-2 text-sm font-semibold uppercase tracking-[0.08em] text-[#2f6b52]">
                  Conflict Status
                </h3>
                <p className="text-[var(--muted)]">
                  No documented land conflict for this site.
                </p>
              </div>
            )}

            {/* Links: Google Maps, GEM, News sources */}
            <div>
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-[0.08em] text-[#46564e]">
                Links
              </h3>
              <div className="flex flex-wrap gap-2">
                {/* Google Maps */}
                {site.google_maps_link && (
                  <a
                    href={site.google_maps_link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 rounded-md border border-[#afd0c1] bg-[#e7f0ec] px-3 py-1.5 text-xs font-medium text-[#285e49] transition-colors hover:bg-[#dbe9e2]"
                  >
                    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
                    </svg>
                    Google Maps
                  </a>
                )}

                {/* GEM Database */}
                {site.gem_url && (
                  <a
                    href={site.gem_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 rounded-md border border-[#e5d099] bg-[#f6efd9] px-3 py-1.5 text-xs font-medium text-[#765a19] transition-colors hover:bg-[#eee3c2]"
                  >
                    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9zm3.75 11.625a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z" />
                    </svg>
                    GEM Database
                  </a>
                )}

                {/* News links */}
                {site.news_links && site.news_links.map((link, i) => {
                  let domain = "";
                  try {
                    domain = new URL(link).hostname.replace("www.", "");
                  } catch {
                    domain = "Source";
                  }
                  return (
                    <a
                      key={i}
                      href={link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 rounded-md border border-[#b5cfda] bg-[#e6eff3] px-3 py-1.5 text-xs font-medium text-[#285f78] transition-colors hover:bg-[#d9e8ee]"
                    >
                      <svg
                        className="w-3 h-3"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                        />
                      </svg>
                      {domain}
                    </a>
                  );
                })}
              </div>
            </div>

            {/* Project metadata */}
            <div className="grid grid-cols-2 gap-4">
              {site.developer && (
              <div>
                <p className="text-xs font-medium uppercase tracking-[0.08em] text-[#728179]">
                  Developer
                </p>
                <p className="mt-1 text-sm text-[#35473e]">{site.developer}</p>
              </div>
              )}
              {site.financing && (
              <div>
                <p className="text-xs font-medium uppercase tracking-[0.08em] text-[#728179]">
                  Financing
                </p>
                <p className="mt-1 text-sm text-[#35473e]">{site.financing}</p>
              </div>
              )}
              {site.completion_date && (
                <div>
                  <p className="text-xs font-medium uppercase tracking-[0.08em] text-[#728179]">
                    Completion Date
                  </p>
                  <p className="mt-1 text-sm text-[#35473e]">
                    {site.completion_date}
                  </p>
                </div>
              )}
              <div>
                <p className="text-xs font-medium uppercase tracking-[0.08em] text-[#728179]">
                  Status
                </p>
                <p className="mt-1 text-sm text-[#35473e]">{site.status}</p>
              </div>
            </div>

            {/* Conflict tags */}
            {site.conflict_tags && site.conflict_tags.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {site.conflict_tags.map((tag) => {
                  const tagColors: Record<string, string> = {
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
                  return (
                    <span
                      key={tag}
                      className={`inline-flex items-center rounded-[4px] border px-2.5 py-1 text-xs font-medium ${
                        tagColors[tag] || "border-[#cad5ce] bg-[#edf1ee] text-[#53645b]"
                      }`}
                    >
                      {tag}
                    </span>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* LULC Timeline chart */}
        {site.annual_lulc && site.annual_lulc.length > 0 && (
          <div className="mt-10 border-t border-[var(--line)] pt-8">
            <LulcTimelineChart
              data={site.annual_lulc}
              constructionYear={constructionYear}
            />
          </div>
        )}

        {/* Pre/post land cover comparison donuts */}
        {site.annual_lulc && site.annual_lulc.length > 0 && (
          <div className="mt-8">
            <LulcDonutChart
              annualLulc={site.annual_lulc}
              constructionYear={constructionYear}
            />
          </div>
        )}
      </div>
    </section>
  );
}
