import { NextResponse } from "next/server";
import { getLabelingTasks } from "@/lib/db";

export async function GET() {
  try {
    const tasks = await getLabelingTasks();
    return NextResponse.json({ tasks });
  } catch (error) {
    console.error("Error fetching labeling tasks:", error);
    return NextResponse.json(
      { error: "Failed to fetch labeling tasks" },
      { status: 500 }
    );
  }
}
