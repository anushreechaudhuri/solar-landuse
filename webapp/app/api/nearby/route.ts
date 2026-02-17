import { NextRequest, NextResponse } from "next/server";
import { findNearbyProjects, findNearbyGrwFeatures } from "@/lib/db";

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const lat = parseFloat(params.get("lat") || "");
  const lon = parseFloat(params.get("lon") || "");
  const radiusKm = parseFloat(params.get("radius_km") || "20");
  const type = params.get("type") || "grw";

  if (isNaN(lat) || isNaN(lon)) {
    return NextResponse.json(
      { error: "Missing or invalid lat/lon parameters" },
      { status: 400 }
    );
  }

  try {
    if (type === "gem") {
      const projects = await findNearbyProjects(lat, lon, radiusKm);
      return NextResponse.json({ results: projects });
    } else {
      const features = await findNearbyGrwFeatures(lat, lon, radiusKm);
      return NextResponse.json({ results: features });
    }
  } catch (error) {
    console.error("Error finding nearby:", error);
    return NextResponse.json(
      { error: "Failed to find nearby items" },
      { status: 500 }
    );
  }
}
