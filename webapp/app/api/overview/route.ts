import { NextResponse } from "next/server";
import { getOverviewPoints } from "@/lib/db";

export async function GET() {
  try {
    const points = await getOverviewPoints();
    return NextResponse.json({ points });
  } catch (error) {
    console.error("Error fetching overview points:", error);
    return NextResponse.json(
      { error: "Failed to fetch overview" },
      { status: 500 }
    );
  }
}
