import { NextRequest, NextResponse } from "next/server";
import { getGrwFeatures } from "@/lib/db";

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  try {
    const result = await getGrwFeatures({
      page: parseInt(params.get("page") || "1"),
      per_page: parseInt(params.get("per_page") || "50"),
      country: params.get("country") || undefined,
      linked: params.get("linked") || undefined,
      search: params.get("search") || undefined,
      sort: params.get("sort") || undefined,
      order: params.get("order") || undefined,
    });
    return NextResponse.json(result);
  } catch (error) {
    console.error("Error fetching GRW features:", error);
    return NextResponse.json(
      { error: "Failed to fetch GRW features" },
      { status: 500 }
    );
  }
}
