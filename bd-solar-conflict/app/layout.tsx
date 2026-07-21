import type { Metadata } from "next";
import { Anek_Bangla, Source_Sans_3 } from "next/font/google";
import "./globals.css";

const sourceSans = Source_Sans_3({
  subsets: ["latin"],
  variable: "--font-source-sans",
  display: "swap",
});

const anekBangla = Anek_Bangla({
  subsets: ["bengali", "latin"],
  variable: "--font-anek-bangla",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Powering Over People — Solar Land Conflicts in Bangladesh",
  description:
    "Interactive investigation of solar energy expansion and land conflicts in Bangladesh. Satellite evidence reveals cropland conversion, forced acquisition, and ecological impacts across 15 solar installations.",
  icons: {
    icon: "/icon.svg",
  },
  openGraph: {
    title: "Powering Over People",
    description:
      "Solar Expansion and Land Conflicts in Bangladesh — An interactive satellite-based investigation",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${sourceSans.variable} ${anekBangla.variable}`}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
