import { NextRequest, NextResponse } from "next/server";
import { getLabelingTask } from "@/lib/db";
import { getPresignedUrl } from "@/lib/s3";

export async function GET(
  _req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const id = parseInt(params.id);
    if (isNaN(id)) {
      return NextResponse.json({ error: "Invalid ID" }, { status: 400 });
    }

    const task: any = await getLabelingTask(id);
    if (!task) {
      return NextResponse.json({ error: "Task not found" }, { status: 404 });
    }

    // Generate presigned URL for the image
    const imageUrl = await getPresignedUrl(task.s3_key);

    return NextResponse.json({ ...task, image_url: imageUrl });
  } catch (error) {
    console.error("Error fetching labeling task:", error);
    return NextResponse.json(
      { error: "Failed to fetch labeling task" },
      { status: 500 }
    );
  }
}
