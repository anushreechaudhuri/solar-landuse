import { NextRequest, NextResponse } from "next/server";
import { getLabelingTask } from "@/lib/db";
import { getPresignedUrl } from "@/lib/s3";

export async function POST(req: NextRequest) {
  try {
    const { task_id, points } = await req.json();

    if (!task_id || !points || points.length === 0) {
      return NextResponse.json(
        { error: "task_id and points required" },
        { status: 400 }
      );
    }

    const modalUrl = process.env.MODAL_SAM_URL;
    if (!modalUrl) {
      return NextResponse.json(
        { error: "SAM backend not configured (MODAL_SAM_URL not set)" },
        { status: 503 }
      );
    }

    // Get task to find s3_key
    const task: any = await getLabelingTask(task_id);
    if (!task) {
      return NextResponse.json({ error: "Task not found" }, { status: 404 });
    }

    // Generate presigned URL for the image
    const imageUrl = await getPresignedUrl(task.s3_key);

    // Call Modal SAM endpoint
    const response = await fetch(modalUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_url: imageUrl,
        points,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("SAM backend error:", response.status, errorText);
      return NextResponse.json(
        { error: "SAM prediction failed" },
        { status: 502 }
      );
    }

    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error("SAM API error:", error);
    return NextResponse.json(
      { error: "SAM prediction failed" },
      { status: 500 }
    );
  }
}
