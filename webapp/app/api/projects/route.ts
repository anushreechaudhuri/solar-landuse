import { NextRequest, NextResponse } from "next/server";
import { getProjects } from "@/lib/db";

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  try {
    const result = await getProjects({
      page: parseInt(params.get("page") || "1"),
      per_page: parseInt(params.get("per_page") || "50"),
      country: params.get("country") || undefined,
      status: params.get("status") || undefined,
      confidence: params.get("confidence") || undefined,
      reviewed: params.get("reviewed") || undefined,
      search: params.get("search") || undefined,
      sort: params.get("sort") || undefined,
      order: params.get("order") || undefined,
      tab: params.get("tab") || undefined,
    });
    return NextResponse.json(result);
  } catch (error) {
    console.error("Error fetching projects:", error);
    return NextResponse.json(
      { error: "Failed to fetch projects" },
      { status: 500 }
    );
  }
}
