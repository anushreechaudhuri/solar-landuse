import { NextRequest, NextResponse } from "next/server";
import { saveAnnotation } from "@/lib/db";

export async function POST(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const taskId = parseInt(params.id);
    if (isNaN(taskId)) {
      return NextResponse.json({ error: "Invalid task ID" }, { status: 400 });
    }

    const body = await req.json();
    const { annotator, regions } = body;

    if (!annotator || !Array.isArray(regions)) {
      return NextResponse.json(
        { error: "annotator (string) and regions (array) required" },
        { status: 400 }
      );
    }

    const annotationId = await saveAnnotation(taskId, annotator, regions);

    return NextResponse.json({ id: annotationId, saved: true });
  } catch (error) {
    console.error("Error saving annotation:", error);
    return NextResponse.json(
      { error: "Failed to save annotation" },
      { status: 500 }
    );
  }
}
