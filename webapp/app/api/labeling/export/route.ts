import { NextResponse } from "next/server";
import { getAllAnnotations } from "@/lib/db";

export async function GET() {
  try {
    const annotations = await getAllAnnotations();

    const exportData = annotations.map((a: any) => ({
      task_id: a.task_id,
      site_name: a.site_name,
      buffer_km: a.buffer_km,
      year: a.year,
      month: a.month,
      period: a.period,
      image_filename: a.image_filename,
      image_width: a.image_width,
      image_height: a.image_height,
      bbox: {
        west: a.bbox_west,
        south: a.bbox_south,
        east: a.bbox_east,
        north: a.bbox_north,
      },
      annotator: a.annotator,
      regions: a.regions,
      created_at: a.created_at,
      updated_at: a.updated_at,
    }));

    return NextResponse.json({
      export_format: "polygon_annotations_v1",
      total_annotations: exportData.length,
      annotations: exportData,
    });
  } catch (error) {
    console.error("Error exporting annotations:", error);
    return NextResponse.json(
      { error: "Failed to export annotations" },
      { status: 500 }
    );
  }
}
