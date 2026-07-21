"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface LulcTimelineChartProps {
  data: Array<{
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
  }>;
  constructionYear: number | null;
}

const LULC_LINES = [
  { key: "crops", name: "Cropland", color: "#DDCC77" },
  { key: "trees", name: "Trees", color: "#117733" },
  { key: "built", name: "Built", color: "#CC6677" },
  { key: "water", name: "Water", color: "#88CCEE" },
  { key: "bare", name: "Bare", color: "#882255" },
];

export default function LulcTimelineChart({
  data,
  constructionYear,
}: LulcTimelineChartProps) {
  return (
    <div className="w-full">
      <h4 className="mb-3 text-sm font-semibold text-[#35473e]">
        Land Cover Timeline (DW annual composites)
      </h4>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#d8e0db" />
          <XAxis
            dataKey="year"
            tick={{ fontSize: 12, fill: "#5b6c63" }}
            domain={[2016, 2025]}
          />
          <YAxis
            tick={{ fontSize: 12, fill: "#5b6c63" }}
            label={{
              value: "Coverage (%)",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 12, fill: "#5b6c63" },
            }}
          />
          <Tooltip
            contentStyle={{
              borderRadius: "6px",
              border: "1px solid #cad5ce",
              fontSize: "13px",
              fontFamily: "var(--font-source-sans)",
            }}
            formatter={(value, name) => [
              `${Number(value).toFixed(1)}%`,
              String(name),
            ]}
          />
          <Legend
            wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }}
          />
          {constructionYear && (
            <ReferenceLine
              x={constructionYear}
              stroke="#b54736"
              strokeDasharray="5 5"
              strokeWidth={2}
              label={{
                value: "Construction",
                position: "top",
                style: { fontSize: 11, fill: "#b54736" },
              }}
            />
          )}
          {LULC_LINES.map((line) => (
            <Line
              key={line.key}
              type="monotone"
              dataKey={line.key}
              name={line.name}
              stroke={line.color}
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
