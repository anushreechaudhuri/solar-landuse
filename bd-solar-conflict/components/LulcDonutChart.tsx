"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

interface AnnualLulcEntry {
  year: number;
  crops: number;
  trees: number;
  built: number;
  bare: number;
  water: number;
  grass: number;
  shrub: number;
  flooded_veg: number;
  ndvi: number;
}

interface LulcDonutChartProps {
  annualLulc: AnnualLulcEntry[];
  constructionYear: number | null;
}

const LULC_COLORS: Record<string, string> = {
  crops: "#DDCC77",
  trees: "#117733",
  built: "#CC6677",
  bare: "#882255",
  water: "#88CCEE",
  shrub: "#999933",
  grass: "#44AA99",
  flooded_veg: "#332288",
  solar: "#FF6B35",
};

const LULC_LABELS: Record<string, string> = {
  crops: "Cropland",
  trees: "Trees",
  built: "Built",
  bare: "Bare",
  water: "Water",
  shrub: "Shrub",
  grass: "Grass",
  flooded_veg: "Flooded Veg",
  solar: "Solar",
};

const LULC_KEYS = ["crops", "trees", "built", "bare", "water", "shrub", "grass", "flooded_veg"] as const;

interface SliceData {
  name: string;
  value: number;
  fill: string;
}

const RADIAN = Math.PI / 180;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function renderCustomLabel(props: any) {
  const { cx, cy, midAngle, outerRadius, percent, name } = props;
  if (!percent || percent < 0.05) return null;
  const radius = (outerRadius || 80) + 18;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  return (
    <text
      x={x}
      y={y}
      fill="#5b6c63"
      textAnchor={x > cx ? "start" : "end"}
      dominantBaseline="central"
      fontSize={10}
    >
      {name} {(percent * 100).toFixed(0)}%
    </text>
  );
}

function buildSlices(entry: AnnualLulcEntry): SliceData[] {
  return LULC_KEYS
    .map((key) => ({
      name: LULC_LABELS[key] || key,
      value: parseFloat((entry[key] ?? 0).toFixed(1)),
      fill: LULC_COLORS[key] || "#999999",
    }))
    .filter((s) => s.value > 1)
    .sort((a, b) => b.value - a.value);
}

function SingleDonut({
  slices,
  centerLabel,
}: {
  slices: SliceData[];
  centerLabel: string;
}) {
  if (slices.length === 0) return null;
  return (
    <ResponsiveContainer width="100%" height={240}>
      <PieChart>
        <Pie
          data={slices}
          cx="50%"
          cy="50%"
          innerRadius={45}
          outerRadius={78}
          paddingAngle={2}
          dataKey="value"
          label={renderCustomLabel}
        >
          {slices.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.fill} />
          ))}
        </Pie>
        <Tooltip
          formatter={(value, name) => [
            `${Number(value).toFixed(1)}%`,
            String(name),
          ]}
          contentStyle={{
            borderRadius: "6px",
            border: "1px solid #cad5ce",
            fontSize: "12px",
            fontFamily: "var(--font-source-sans)",
          }}
        />
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dominantBaseline="middle"
          fill="#5b6c63"
          fontSize={10}
        >
          {centerLabel}
        </text>
      </PieChart>
    </ResponsiveContainer>
  );
}

export default function LulcDonutChart({
  annualLulc,
  constructionYear,
}: LulcDonutChartProps) {
  if (!annualLulc || annualLulc.length === 0) return null;

  const sorted = [...annualLulc].sort((a, b) => a.year - b.year);

  // Find pre-construction year (last year before construction, or earliest available)
  let preEntry: AnnualLulcEntry | null = null;
  let postEntry: AnnualLulcEntry | null = null;

  if (constructionYear) {
    const preYears = sorted.filter((e) => e.year < constructionYear);
    const postYears = sorted.filter((e) => e.year >= constructionYear);
    preEntry = preYears.length > 0 ? preYears[preYears.length - 1] : null;
    postEntry = postYears.length > 0 ? postYears[postYears.length - 1] : null;
  } else {
    // No construction year — just show earliest and latest
    preEntry = sorted[0];
    postEntry = sorted[sorted.length - 1];
  }

  if (!preEntry && !postEntry) return null;

  const preSlices = preEntry ? buildSlices(preEntry) : [];
  const postSlices = postEntry ? buildSlices(postEntry) : [];

  return (
    <div className="w-full">
      <h4 className="mb-1 text-sm font-semibold text-[#35473e]">
        Land Cover Composition (Buffer Area)
      </h4>
      <p className="mb-4 text-xs text-[#728179]">
        Dynamic World classification of the area surrounding the solar site
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {preEntry && (
          <div>
            <p className="mb-1 text-center text-xs font-medium text-[var(--muted)]">
              Pre-construction ({preEntry.year})
            </p>
            <SingleDonut slices={preSlices} centerLabel={`${preEntry.year}`} />
          </div>
        )}
        {postEntry && (
          <div>
            <p className="mb-1 text-center text-xs font-medium text-[var(--muted)]">
              Post-construction ({postEntry.year})
            </p>
            <SingleDonut slices={postSlices} centerLabel={`${postEntry.year}`} />
          </div>
        )}
      </div>
    </div>
  );
}
