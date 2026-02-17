import { NextRequest, NextResponse } from "next/server";
import { mergeGrwToProject, unmergeGrwFromProject } from "@/lib/db";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { grw_feature_id, project_id, performed_by, notes } = body;

    if (!grw_feature_id || !project_id || !performed_by) {
      return NextResponse.json(
        { error: "Missing required fields: grw_feature_id, project_id, performed_by" },
        { status: 400 }
      );
    }

    await mergeGrwToProject(grw_feature_id, project_id, performed_by, notes || null);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error merging:", error);
    return NextResponse.json(
      { error: (error as Error).message || "Failed to merge" },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const body = await request.json();
    const { grw_feature_id, performed_by, notes } = body;

    if (!grw_feature_id || !performed_by) {
      return NextResponse.json(
        { error: "Missing required fields: grw_feature_id, performed_by" },
        { status: 400 }
      );
    }

    await unmergeGrwFromProject(grw_feature_id, performed_by, notes || null);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error unmerging:", error);
    return NextResponse.json(
      { error: (error as Error).message || "Failed to unmerge" },
      { status: 500 }
    );
  }
}
