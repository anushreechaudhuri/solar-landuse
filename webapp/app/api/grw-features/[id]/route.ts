import { NextRequest, NextResponse } from "next/server";
import { getGrwFeature, updateGrwFeature } from "@/lib/db";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const feature = await getGrwFeature(parseInt(id));
    if (!feature) {
      return NextResponse.json({ error: "GRW feature not found" }, { status: 404 });
    }
    return NextResponse.json(feature);
  } catch (error) {
    console.error("Error fetching GRW feature:", error);
    return NextResponse.json(
      { error: "Failed to fetch GRW feature" },
      { status: 500 }
    );
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const body = await request.json();
    await updateGrwFeature(parseInt(id), body);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error updating GRW feature:", error);
    return NextResponse.json(
      { error: "Failed to update GRW feature" },
      { status: 500 }
    );
  }
}
