"use client";

import Image from "next/image";

const FINDINGS = [
  {
    icon: (
      <svg
        className="w-8 h-8"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    ),
    headline: "39.6%",
    description:
      "of Bangladesh solar sites were built on cropland, based on 10-year satellite land cover analysis using Dynamic World annual composites.",
    color: "text-[#8a681b]",
  },
  {
    icon: (
      <svg
        className="w-8 h-8"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
        />
      </svg>
    ),
    headline: "No surrounding degradation",
    description:
      "Within-site event study analysis shows no detectable cropland loss or tree loss in the surrounding landscape post-construction. The land cover change is confined to within the solar site footprint.",
    color: "text-[#2f6b52]",
  },
  {
    icon: (
      <svg
        className="w-8 h-8"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z"
        />
      </svg>
    ),
    headline: "67% have documented conflicts",
    description:
      "Two-thirds of analyzed Bangladesh solar sites have documented land conflicts, including forced acquisition, three-crop land seizure, farmer livelihood loss, and ecological damage.",
    color: "text-[#a13e30]",
  },
];

export default function SatelliteFindings() {
  return (
    <section className="bg-white py-16" id="findings">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-semibold text-[var(--foreground)]">
          What Satellites Reveal
        </h2>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          Combining 10 years of satellite observations with ground-truth
          conflict reporting paints a detailed picture of solar land use in
          Bangladesh.
        </p>

        {/* Finding summary */}
        <div className="mt-10 border-y border-[var(--line)] md:grid md:grid-cols-3 md:divide-x md:divide-[var(--line)]">
          {FINDINGS.map((finding, i) => (
            <div
              key={i}
              className="border-b border-[var(--line)] py-7 last:border-b-0 md:border-b-0 md:px-7 md:first:pl-0 md:last:pr-0"
            >
              <div className={finding.color}>{finding.icon}</div>
              <p className={`mt-4 text-2xl font-semibold leading-tight ${finding.color}`}>
                {finding.headline}
              </p>
              <p className="mt-2 text-sm leading-relaxed text-[var(--muted)]">
                {finding.description}
              </p>
            </div>
          ))}
        </div>

        {/* Composite figure */}
        <div className="mt-12">
          <div className="overflow-hidden rounded-[8px] border border-[var(--line)]">
            <Image
              src="/images/case_studies/all_sites_pre_post.png"
              alt="Pre- and post-construction satellite imagery and DW LULC maps for all four case study sites"
              width={1600}
              height={800}
              className="w-full h-auto"
            />
          </div>
          <p className="mx-auto mt-3 max-w-3xl text-center text-sm text-[var(--muted)]">
            Pre- and post-construction satellite imagery (Planet 4.77m basemaps)
            with Dynamic World land cover classification for four case study
            sites. The satellite images are January composites; LULC maps show
            dry-season annual composites from the same period. Construction years
            range from 2021 (Manikganj) to 2025 (Moulvibazar).
          </p>
        </div>
      </div>
    </section>
  );
}
