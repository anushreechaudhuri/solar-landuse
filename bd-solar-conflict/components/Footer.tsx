import Image from "next/image";

export default function Footer() {
  return (
    <footer className="border-t border-[#315047] bg-[var(--hero-bg)] text-[#a8bdb2]">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Institutional logo */}
        <div className="mb-10 flex flex-wrap items-center justify-center gap-8 border-b border-white/15 pb-8">
          <a
            href="https://www.geog.cam.ac.uk/"
            target="_blank"
            rel="noopener noreferrer"
            className="shrink-0 opacity-80 hover:opacity-100 transition-opacity"
          >
            <Image
              src="/images/cambridge-logo.png"
              alt="University of Cambridge"
              width={180}
              height={38}
              className="h-9 w-auto brightness-200"
            />
          </a>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Credits */}
          <div>
            <h4 className="text-white font-semibold text-sm uppercase tracking-wide mb-3">
              Research
            </h4>
            <p className="text-sm leading-relaxed">
              Research by{" "}
              <a
                href="https://www.geog.cam.ac.uk/people/chaudhuri/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#a9d7c4] underline underline-offset-2 hover:text-white"
              >
                Anushree Chaudhuri
              </a>
              , Department of Geography, University of Cambridge.
            </p>
            <p className="text-sm leading-relaxed mt-2">
              Ground-truthing in collaboration with the{" "}
              <a
                href="https://bigd.bracu.ac.bd/our-work/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#a9d7c4] underline underline-offset-2 hover:text-white"
              >
                BRAC Institute of Governance and Development (BIGD)
              </a>
              .
            </p>
            <a
              href="mailto:anuc@alum.mit.edu"
              className="mt-4 inline-flex items-center gap-2 rounded-md border border-white/20 bg-white/5 px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-white/10"
            >
              <svg
                className="w-3.5 h-3.5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                />
              </svg>
              Contact
            </a>
          </div>

          {/* Data sources */}
          <div>
            <h4 className="text-white font-semibold text-sm uppercase tracking-wide mb-3">
              Data Sources
            </h4>
            <p className="text-sm leading-relaxed">
              <span className="text-[#d7e3dd]">Satellite data:</span> Google
              Dynamic World, Sentinel-2
              <br />
              <span className="text-[#d7e3dd]">Solar database:</span> Global
              Renewables Watch, TransitionZero, Global Energy Monitor
              <br />
              <span className="text-[#d7e3dd]">Conflict data:</span> SREDA,
              field research
            </p>
          </div>

          {/* Methodology */}
          <div>
            <h4 className="text-white font-semibold text-sm uppercase tracking-wide mb-3">
              Methodology
            </h4>
            <p className="text-sm leading-relaxed">
              Land cover change detected using 10-year annual satellite
              composites (2016-2025). Within-site event study design with site
              and year fixed effects. VLM classification via Gemini 2.5 Flash.
            </p>
          </div>
        </div>

        <div className="mt-10 border-t border-white/15 pt-6 text-center text-xs text-[#82978c]">
          <p>
            Satellite imagery and land cover analysis updated through January
            2026. All findings are preliminary and subject to peer review.
          </p>
        </div>
      </div>
    </footer>
  );
}
