"use client";

import dynamic from "next/dynamic";

const LabelingApp = dynamic(() => import("@/components/LabelingApp"), {
  ssr: false,
  loading: () => (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
        fontFamily: "system-ui",
      }}
    >
      Loading labeling studio...
    </div>
  ),
});

export default function LabelPage() {
  return <LabelingApp />;
}
