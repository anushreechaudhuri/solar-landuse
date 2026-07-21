"use client";

import { useState } from "react";
import Image from "next/image";
import { SolarSite } from "@/lib/types";

interface CaseStudiesProps {
  sites: SolarSite[];
}

interface CaseStudy {
  id: string;
  title: string;
  narrative: string;
}

const CASE_STUDIES: CaseStudy[] = [
  {
    id: "teesta",
    title: "Teesta (Gaibandha) — 200 MW",
    narrative:
      "Bangladesh's largest solar installation at 200 MW, built on prime agricultural land in Gaibandha. Satellite imagery reveals the transformation of over 400 acres of cropland. Reports document violent and illegal land acquisition, with farmers losing their livelihoods.",
  },
  {
    id: "feni",
    title: "Feni/Sonagazi — 75 MW",
    narrative:
      "A 75 MW World Bank-financed project in Sonagazi, built on three-crop agricultural land. Farmers protested the acquisition of their productive farmland for solar installation.",
  },
  {
    id: "manikganj",
    title: "Manikganj (Spectra) — 35 MW",
    narrative:
      "A 35 MW project by Spectra Engineers in Shibalaya, built on cropland along the Padma river. Land acquisition involved threats and inadequate compensation, compounded by ongoing river erosion.",
  },
  {
    id: "moulvibazar",
    title: "Moulvibazar — 10 MW",
    narrative:
      "A 10 MW installation in a sensitive haor (seasonal wetland) ecosystem. Community opposition centers on ecological destruction and forced land acquisition, with proposed expansions up to 100 MW threatening additional wetland areas.",
  },
];

const TAG_COLORS: Record<string, string> = {
  "Forced Acquisition": "border-[#e2b7af] bg-[#f6e8e5] text-[#87372b]",
  "Three-Crop Land": "border-[#e5d099] bg-[#f6efd9] text-[#765a19]",
  "Farmer Livelihoods": "border-[#e5d099] bg-[#f6efd9] text-[#765a19]",
  "Ecological Impact": "border-[#afd0c1] bg-[#e7f0ec] text-[#285e49]",
  Corruption: "border-[#e2b7af] bg-[#f6e8e5] text-[#87372b]",
  "River Erosion": "border-[#b5cfda] bg-[#e6eff3] text-[#285f78]",
  "Inadequate Compensation": "border-[#e2b7af] bg-[#f6e8e5] text-[#87372b]",
  "Community Protests": "border-[#e2b7af] bg-[#f6e8e5] text-[#87372b]",
};

function ExpandableImage({
  src,
  alt,
  caption,
}: {
  src: string;
  alt: string;
  caption?: string;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      {/* Thumbnail */}
      <button
        type="button"
        className="group relative block w-full cursor-pointer text-left"
        onClick={() => setExpanded(true)}
        aria-label={`Expand ${alt}`}
      >
        <Image
          src={src}
          alt={alt}
          width={1200}
          height={600}
          className="h-auto w-full rounded-md border border-[var(--line)] transition-colors group-hover:border-[#8ea89a]"
        />
        {/* Expand hint */}
        <span className="absolute inset-0 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100">
          <span className="flex items-center gap-1.5 rounded-md bg-[#17251f]/85 px-3 py-1.5 text-xs font-medium text-white">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
            </svg>
            Click to expand
          </span>
        </span>
        {caption && (
          <span className="mt-1.5 block text-center text-xs text-[#728179]">{caption}</span>
        )}
      </button>

      {/* Fullscreen overlay */}
      {expanded && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4 sm:p-8"
          onClick={() => setExpanded(false)}
          role="dialog"
          aria-modal="true"
          aria-label={alt}
        >
          <button
            type="button"
            className="absolute right-4 top-4 z-10 rounded-md border border-white/25 p-1 text-white/70 hover:text-white"
            onClick={() => setExpanded(false)}
            aria-label="Close expanded image"
          >
            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          <Image
            src={src}
            alt={alt}
            width={2400}
            height={1200}
            className="max-w-full max-h-full object-contain"
            onClick={(e) => e.stopPropagation()}
          />
          {caption && (
            <p className="absolute bottom-4 left-1/2 -translate-x-1/2 text-white/70 text-sm">
              {caption}
            </p>
          )}
        </div>
      )}
    </>
  );
}

function CaseStudyCard({
  study,
  site,
}: {
  study: CaseStudy;
  site: SolarSite | undefined;
}) {
  // Show satellite+LULC overlay map as primary (most informative),
  // plus LULC change detail chart as secondary
  const lulcMaps = site?.images?.lulc_maps;
  const lulcChange = site?.images
    ? `/images/case_studies/${study.id}_lulc_change_detail.png`
    : null;
  const prePost = site?.images?.pre_post;

  return (
    <div className="overflow-hidden rounded-[10px] border border-[var(--line)] bg-white">
      {/* Narrative */}
      <div className="p-6">
        <h3 className="text-lg font-semibold text-[var(--foreground)]">{study.title}</h3>
        <p className="mt-3 text-sm leading-relaxed text-[var(--muted)]">
          {study.narrative}
        </p>
        {site?.conflict_tags && (
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
        )}
      </div>

      {/* Primary figure: Satellite + LULC overlay (click to expand) */}
      <div className="px-6 pb-4">
        {lulcMaps ? (
          <ExpandableImage
            src={lulcMaps}
            alt={`${study.title} — Satellite imagery with LULC classification (2016–2026)`}
            caption="Satellite imagery with Dynamic World land cover classification — click to expand"
          />
        ) : prePost ? (
          <ExpandableImage
            src={prePost}
            alt={`${study.title} — Pre and post construction comparison`}
            caption="Pre/post construction satellite comparison — click to expand"
          />
        ) : null}
      </div>

      {/* Secondary figure: LULC change detail chart */}
      {lulcChange && (
        <div className="px-6 pb-6">
          <ExpandableImage
            src={lulcChange}
            alt={`${study.title} — Land cover change breakdown`}
            caption="Land cover change analysis — click to expand"
          />
        </div>
      )}
    </div>
  );
}

export default function CaseStudies({ sites }: CaseStudiesProps) {
  const siteMap = Object.fromEntries(sites.map((s) => [s.id, s]));

  return (
    <section className="bg-[var(--background)] py-16" id="case-studies">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-semibold text-[var(--foreground)]">Case Studies</h2>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          In-depth satellite analysis of four sites with documented land
          conflicts, spanning 10–200 MW capacity. Click any image to view
          full-size.
        </p>

        <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-2">
          {CASE_STUDIES.map((study) => (
            <CaseStudyCard
              key={study.id}
              study={study}
              site={siteMap[study.id]}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
