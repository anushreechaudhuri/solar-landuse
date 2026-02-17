import { NextRequest, NextResponse } from "next/server";
import { createReview } from "@/lib/db";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const body = await request.json();
    const { reviewer_name, action, polygon, notes } = body;

    if (!reviewer_name || !action) {
      return NextResponse.json(
        { error: "reviewer_name and action are required" },
        { status: 400 }
      );
    }

    const validActions = [
      "confirmed",
      "edited_polygon",
      "no_match",
      "drawn_new",
      "feasibility_yes",
      "feasibility_no",
      "feasibility_maybe",
    ];
    if (!validActions.includes(action)) {
      return NextResponse.json(
        { error: `Invalid action. Must be one of: ${validActions.join(", ")}` },
        { status: 400 }
      );
    }

    const review = await createReview(
      id,
      reviewer_name,
      action,
      polygon || null,
      notes || null
    );
    return NextResponse.json(review, { status: 201 });
  } catch (error) {
    console.error("Error creating review:", error);
    return NextResponse.json(
      { error: "Failed to create review" },
      { status: 500 }
    );
  }
}
