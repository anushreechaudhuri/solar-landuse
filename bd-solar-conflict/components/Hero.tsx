"use client";

import Image from "next/image";
import { SolarSite } from "@/lib/types";

interface HeroProps {
  sites: SolarSite[];
}

export default function Hero({ sites }: HeroProps) {
  const totalSites = sites.length;
  const totalCapacity = sites.reduce((sum, s) => sum + (s.capacity_mw || 0), 0);
  const conflictSites = sites.filter((s) => s.has_conflict).length;
  const conflictPct = totalSites
    ? Math.round((conflictSites / totalSites) * 100)
    : 0;

  const stats = [
    {
      label: "Total Sites",
      value: totalSites.toString(),
      sublabel: "Analyzed",
      valueColor: "text-white",
    },
    {
      label: "Total Capacity",
      value: `${totalCapacity.toFixed(0)} MW`,
      sublabel: "Solar installed",
      valueColor: "text-[#a9d7c4]",
    },
    {
      label: "Sites with Conflict",
      value: `${conflictSites}`,
      sublabel: `${conflictPct}% of total`,
      valueColor: "text-[#efaa9f]",
    },
    {
      label: "Cropland Converted",
      value: "39.6%",
      sublabel: "National average",
      valueColor: "text-[#e8c86f]",
    },
  ];

  return (
    <section className="relative overflow-hidden border-b border-[#315047] bg-[var(--hero-bg)] text-white">
      {/* Hero graphic — positioned top-right, decorative */}
      <div className="pointer-events-none absolute right-0 top-8 w-[280px] select-none opacity-[0.14] sm:w-[340px] lg:w-[420px] lg:opacity-20">
        <Image
          src="/images/hero-graphic.png"
          alt=""
          width={420}
          height={420}
          className="w-full h-auto"
          priority
          aria-hidden="true"
        />
      </div>

      <div className="relative mx-auto max-w-6xl px-4 py-20 sm:px-6 sm:py-28 lg:px-8 lg:py-32">
        {/* Title */}
        <h1 className="max-w-4xl text-balance text-5xl font-semibold leading-[0.98] sm:text-6xl lg:text-7xl">
          Powering Over People
        </h1>
        <p className="mt-5 max-w-3xl text-xl font-medium text-[#d9b85e] sm:text-2xl">
          Solar Expansion and Land Conflicts in Bangladesh
        </p>

        {/* Intro paragraph */}
        <p className="mt-8 max-w-3xl text-lg leading-relaxed text-[var(--hero-muted)]">
          Bangladesh is rapidly expanding solar energy to meet growing demand.
          But satellite evidence reveals that over half of operational solar
          sites have documented land conflicts — from forced acquisition of prime
          cropland to ecological destruction of sensitive wetlands.
        </p>

        {/* Stats bar */}
        <div className="mt-14 grid grid-cols-2 divide-x divide-white/15 border-y border-white/20 lg:grid-cols-4">
          {stats.map((stat) => (
            <div
              key={stat.label}
              className="px-4 py-5 sm:px-6 sm:py-6"
            >
              <p className="text-xs font-semibold uppercase tracking-[0.09em] text-white/65">
                {stat.label}
              </p>
              <p className={`mt-2 text-3xl font-semibold sm:text-4xl ${stat.valueColor}`}>
                {stat.value}
              </p>
              <p className="mt-1 text-sm text-white/55">{stat.sublabel}</p>
            </div>
          ))}
        </div>

        {/* Scroll hint */}
        <div className="mt-14 flex justify-center">
          <a
            href="#map"
            className="flex flex-col items-center gap-2 text-[#a8bdb2] transition-colors hover:text-white"
          >
            <span className="text-sm">Explore the map</span>
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 14l-7 7m0 0l-7-7m7 7V3"
              />
            </svg>
          </a>
        </div>
      </div>
    </section>
  );
}
