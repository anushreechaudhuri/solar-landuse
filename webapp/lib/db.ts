import { createPool, type VercelPool, type QueryResultRow } from "@vercel/postgres";

let _pool: VercelPool | null = null;

function getPool(): VercelPool {
  if (!_pool) {
    _pool = createPool({
      connectionString: process.env.POSTGRES_URL || process.env.DATABASE_URL,
    });
  }
  return _pool;
}

// Tagged template for simple queries: sql`SELECT ...`
function sql(strings: TemplateStringsArray, ...values: Primitive[]) {
  return getPool().sql(strings, ...values);
}

type Primitive = string | number | boolean | undefined | null;

// Also support sql.query(text, values) for dynamic queries
sql.query = <R extends QueryResultRow = QueryResultRow>(text: string, values?: unknown[]) =>
  getPool().query<R>(text, values);

export { sql };

export async function getProjects(params: {
  page?: number;
  per_page?: number;
  country?: string;
  status?: string;
  confidence?: string;
  reviewed?: string;
  search?: string;
  sort?: string;
  order?: string;
  tab?: string;
}) {
  const {
    page = 1,
    per_page = 50,
    country,
    status,
    confidence,
    reviewed,
    search,
    sort = "capacity_mw",
    order = "DESC",
    tab = "active",
  } = params;

  const conditions: string[] = [];
  const values: unknown[] = [];
  let paramIdx = 1;

  // Tab filter: active = operating/construction, proposed = everything else
  if (tab === "active") {
    conditions.push(`status IN ('operating', 'construction')`);
  } else if (tab === "proposed") {
    conditions.push(`status NOT IN ('operating', 'construction')`);
  }

  if (country) {
    conditions.push(`country = $${paramIdx++}`);
    values.push(country);
  }
  if (status) {
    conditions.push(`status = $${paramIdx++}`);
    values.push(status);
  }
  if (confidence) {
    conditions.push(`match_confidence = $${paramIdx++}`);
    values.push(confidence);
  }
  if (reviewed === "true") {
    conditions.push(`id IN (SELECT DISTINCT project_id FROM reviews)`);
  } else if (reviewed === "false") {
    conditions.push(`id NOT IN (SELECT DISTINCT project_id FROM reviews)`);
  }
  if (search) {
    conditions.push(
      `(project_name ILIKE $${paramIdx} OR phase_name ILIKE $${paramIdx} OR id ILIKE $${paramIdx} OR gem_location_id ILIKE $${paramIdx} OR other_ids ILIKE $${paramIdx} OR CAST(capacity_mw AS TEXT) LIKE $${paramIdx} OR CAST(start_year AS TEXT) LIKE $${paramIdx})`
    );
    values.push(`%${search}%`);
    paramIdx++;
  }

  const where = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

  // Validate sort column
  const allowedSorts = [
    "capacity_mw",
    "project_name",
    "country",
    "start_year",
    "match_confidence",
    "status",
  ];
  const sortCol = allowedSorts.includes(sort) ? sort : "capacity_mw";
  const sortOrder = order === "ASC" ? "ASC" : "DESC";

  const offset = (page - 1) * per_page;

  // Count total
  const countQuery = `SELECT COUNT(*) as total FROM projects ${where}`;
  const countResult = await sql.query(countQuery, values);
  const total = parseInt(countResult.rows[0].total);

  // Fetch projects with latest review
  const dataQuery = `
    SELECT p.*,
      r.reviewer_name as latest_reviewer,
      r.action as latest_action,
      r.created_at as latest_review_at,
      (SELECT COUNT(*) FROM reviews WHERE project_id = p.id) as review_count
    FROM projects p
    LEFT JOIN LATERAL (
      SELECT reviewer_name, action, created_at
      FROM reviews
      WHERE project_id = p.id
      ORDER BY created_at DESC
      LIMIT 1
    ) r ON true
    ${where}
    ORDER BY ${sortCol} ${sortOrder} NULLS LAST
    LIMIT $${paramIdx++} OFFSET $${paramIdx++}
  `;
  values.push(per_page, offset);
  const dataResult = await sql.query(dataQuery, values);

  return {
    projects: dataResult.rows.map((row) => ({
      ...row,
      grw_polygons:
        typeof row.grw_polygons === "string"
          ? JSON.parse(row.grw_polygons)
          : row.grw_polygons,
      merged_polygon:
        typeof row.merged_polygon === "string"
          ? JSON.parse(row.merged_polygon)
          : row.merged_polygon,
      latest_review: row.latest_reviewer
        ? {
            reviewer_name: row.latest_reviewer,
            action: row.latest_action,
            created_at: row.latest_review_at,
          }
        : null,
    })),
    total,
    page,
    per_page,
  };
}

export async function getProject(id: string) {
  const result = await sql`
    SELECT * FROM projects WHERE id = ${id}
  `;
  if (result.rows.length === 0) return null;

  const row = result.rows[0];
  const reviews = await sql`
    SELECT * FROM reviews WHERE project_id = ${id} ORDER BY created_at DESC
  `;

  return {
    ...row,
    grw_polygons:
      typeof row.grw_polygons === "string"
        ? JSON.parse(row.grw_polygons)
        : row.grw_polygons,
    merged_polygon:
      typeof row.merged_polygon === "string"
        ? JSON.parse(row.merged_polygon)
        : row.merged_polygon,
    reviews: reviews.rows,
  };
}

export async function updateProjectPolygon(
  id: string,
  polygon: unknown
) {
  await sql`
    UPDATE projects SET merged_polygon = ${JSON.stringify(polygon)}
    WHERE id = ${id}
  `;
}

export async function createReview(
  projectId: string,
  reviewerName: string,
  action: string,
  polygon: unknown | null,
  notes: string | null
) {
  const result = await sql`
    INSERT INTO reviews (project_id, reviewer_name, action, polygon, notes)
    VALUES (${projectId}, ${reviewerName}, ${action},
      ${polygon ? JSON.stringify(polygon) : null}, ${notes})
    RETURNING *
  `;
  return result.rows[0];
}

export async function getStats() {
  const [total, byCountry, byStatus, byConfidence, reviewed, grwUnmatched] =
    await Promise.all([
      sql`SELECT COUNT(*) as count FROM projects`,
      sql`SELECT country, COUNT(*) as count FROM projects GROUP BY country ORDER BY count DESC`,
      sql`SELECT status, COUNT(*) as count FROM projects GROUP BY status ORDER BY count DESC`,
      sql`SELECT match_confidence, COUNT(*) as count FROM projects GROUP BY match_confidence ORDER BY count DESC`,
      sql`SELECT COUNT(DISTINCT project_id) as count FROM reviews`,
      sql`SELECT COUNT(*) as count FROM grw_features WHERE linked_project_id IS NULL`,
    ]);

  return {
    total_projects: parseInt(total.rows[0].count),
    by_country: Object.fromEntries(
      byCountry.rows.map((r) => [r.country, parseInt(r.count)])
    ),
    by_status: Object.fromEntries(
      byStatus.rows.map((r) => [r.status, parseInt(r.count)])
    ),
    by_confidence: Object.fromEntries(
      byConfidence.rows.map((r) => [r.match_confidence || "none", parseInt(r.count)])
    ),
    reviewed: parseInt(reviewed.rows[0].count),
    unreviewed:
      parseInt(total.rows[0].count) - parseInt(reviewed.rows[0].count),
    total_grw_unmatched: parseInt(grwUnmatched.rows[0].count),
  };
}

// --- GRW Features ---

export async function getGrwFeatures(params: {
  page?: number;
  per_page?: number;
  country?: string;
  linked?: string;
  search?: string;
  sort?: string;
  order?: string;
}) {
  const {
    page = 1,
    per_page = 50,
    country,
    linked,
    search,
    sort = "area_m2",
    order = "DESC",
  } = params;

  const conditions: string[] = [];
  const values: unknown[] = [];
  let paramIdx = 1;

  if (country) {
    conditions.push(`country = $${paramIdx++}`);
    values.push(country);
  }
  if (linked === "true") {
    conditions.push(`linked_project_id IS NOT NULL`);
  } else if (linked === "false") {
    conditions.push(`linked_project_id IS NULL`);
  }
  if (search) {
    conditions.push(
      `(CAST(fid AS TEXT) LIKE $${paramIdx} OR user_name ILIKE $${paramIdx} OR country ILIKE $${paramIdx} OR CAST(id AS TEXT) = $${paramIdx})`
    );
    values.push(`%${search}%`);
    paramIdx++;
  }

  const where = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

  const allowedSorts = ["area_m2", "construction_year", "country", "fid", "id"];
  const sortCol = allowedSorts.includes(sort) ? sort : "area_m2";
  const sortOrder = order === "ASC" ? "ASC" : "DESC";
  const offset = (page - 1) * per_page;

  const countQuery = `SELECT COUNT(*) as total FROM grw_features ${where}`;
  const countResult = await sql.query(countQuery, values);
  const total = parseInt(countResult.rows[0].total);

  const dataQuery = `
    SELECT * FROM grw_features
    ${where}
    ORDER BY ${sortCol} ${sortOrder} NULLS LAST
    LIMIT $${paramIdx++} OFFSET $${paramIdx++}
  `;
  values.push(per_page, offset);
  const dataResult = await sql.query(dataQuery, values);

  return {
    features: dataResult.rows.map((row) => ({
      ...row,
      polygon:
        typeof row.polygon === "string" ? JSON.parse(row.polygon) : row.polygon,
    })),
    total,
    page,
    per_page,
  };
}

export async function getGrwFeature(id: number) {
  const result = await sql`SELECT * FROM grw_features WHERE id = ${id}`;
  if (result.rows.length === 0) return null;

  const row = result.rows[0];
  const history = await sql`
    SELECT * FROM merge_history WHERE grw_feature_id = ${id} ORDER BY created_at DESC
  `;

  return {
    ...row,
    polygon:
      typeof row.polygon === "string" ? JSON.parse(row.polygon) : row.polygon,
    merge_history: history.rows,
  };
}

export async function updateGrwFeature(
  id: number,
  updates: { user_name?: string; user_capacity_mw?: number | null; user_status?: string; user_notes?: string }
) {
  const { user_name, user_capacity_mw, user_status, user_notes } = updates;
  await sql`
    UPDATE grw_features
    SET user_name = ${user_name ?? null},
        user_capacity_mw = ${user_capacity_mw ?? null},
        user_status = ${user_status ?? null},
        user_notes = ${user_notes ?? null}
    WHERE id = ${id}
  `;
}

// --- Overview ---

export async function getOverviewPoints() {
  const result = await sql.query(`
    SELECT
      id,
      latitude as lat,
      longitude as lon,
      CASE
        WHEN match_confidence IN ('high', 'medium', 'low') AND merged_polygon IS NOT NULL THEN 'matched'
        ELSE 'gem_only'
      END as type,
      project_name as label
    FROM projects
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    UNION ALL
    SELECT
      CAST(id AS TEXT),
      centroid_lat as lat,
      centroid_lon as lon,
      'grw_only' as type,
      COALESCE(user_name, 'GRW #' || fid) as label
    FROM grw_features
    WHERE linked_project_id IS NULL
  `);
  return result.rows;
}

// --- Merge ---

export async function mergeGrwToProject(
  grwFeatureId: number,
  projectId: string,
  performedBy: string,
  notes: string | null
) {
  // Get GRW feature polygon
  const grwResult = await sql`SELECT polygon FROM grw_features WHERE id = ${grwFeatureId}`;
  if (grwResult.rows.length === 0) throw new Error("GRW feature not found");

  // Get current project polygon for undo snapshot
  const projResult = await sql`SELECT merged_polygon FROM projects WHERE id = ${projectId}`;
  if (projResult.rows.length === 0) throw new Error("Project not found");

  const previousPolygon = projResult.rows[0].merged_polygon;
  const grwPolygon = grwResult.rows[0].polygon;
  const grwPoly = typeof grwPolygon === "string" ? grwPolygon : JSON.stringify(grwPolygon);

  // Update project with GRW polygon
  await sql.query(
    `UPDATE projects SET merged_polygon = $1, match_confidence = 'manual' WHERE id = $2`,
    [grwPoly, projectId]
  );

  // Link GRW feature
  await sql`
    UPDATE grw_features SET linked_project_id = ${projectId}, linked_at = NOW()
    WHERE id = ${grwFeatureId}
  `;

  // Record merge history
  await sql.query(
    `INSERT INTO merge_history (grw_feature_id, project_id, action, performed_by, previous_project_polygon, notes)
     VALUES ($1, $2, 'merge', $3, $4, $5)`,
    [grwFeatureId, projectId, performedBy, previousPolygon ? JSON.stringify(previousPolygon) : null, notes]
  );
}

export async function unmergeGrwFromProject(
  grwFeatureId: number,
  performedBy: string,
  notes: string | null
) {
  // Get the latest merge record to restore previous polygon
  const historyResult = await sql`
    SELECT * FROM merge_history
    WHERE grw_feature_id = ${grwFeatureId} AND action = 'merge'
    ORDER BY created_at DESC LIMIT 1
  `;
  if (historyResult.rows.length === 0) throw new Error("No merge history found");

  const mergeRecord = historyResult.rows[0];
  const previousPolygon = mergeRecord.previous_project_polygon;

  // Restore project polygon
  await sql.query(
    `UPDATE projects SET merged_polygon = $1 WHERE id = $2`,
    [previousPolygon ? JSON.stringify(previousPolygon) : null, mergeRecord.project_id]
  );

  // Unlink GRW feature
  await sql`
    UPDATE grw_features SET linked_project_id = NULL, linked_at = NULL
    WHERE id = ${grwFeatureId}
  `;

  // Record unmerge
  await sql.query(
    `INSERT INTO merge_history (grw_feature_id, project_id, action, performed_by, notes)
     VALUES ($1, $2, 'unmerge', $3, $4)`,
    [grwFeatureId, mergeRecord.project_id, performedBy, notes]
  );
}

export async function findNearbyProjects(lat: number, lon: number, radiusKm: number) {
  // Approximate: 1 degree ≈ 111 km
  const degBuffer = radiusKm / 111.0;
  const result = await sql.query(
    `SELECT id, project_name, capacity_mw, status, latitude, longitude, match_confidence,
            merged_polygon IS NOT NULL as has_polygon
     FROM projects
     WHERE latitude BETWEEN $1 AND $2
       AND longitude BETWEEN $3 AND $4
     ORDER BY ABS(latitude - $5) + ABS(longitude - $6)
     LIMIT 20`,
    [lat - degBuffer, lat + degBuffer, lon - degBuffer, lon + degBuffer, lat, lon]
  );
  return result.rows;
}

export async function findNearbyGrwFeatures(lat: number, lon: number, radiusKm: number) {
  const degBuffer = radiusKm / 111.0;
  const result = await sql.query(
    `SELECT id, fid, country, centroid_lat, centroid_lon, area_m2,
            construction_year, user_name, linked_project_id
     FROM grw_features
     WHERE centroid_lat BETWEEN $1 AND $2
       AND centroid_lon BETWEEN $3 AND $4
       AND linked_project_id IS NULL
     ORDER BY ABS(centroid_lat - $5) + ABS(centroid_lon - $6)
     LIMIT 20`,
    [lat - degBuffer, lat + degBuffer, lon - degBuffer, lon + degBuffer, lat, lon]
  );
  return result.rows;
}

// --- Labeling ---

export async function getLabelingTasks() {
  const result = await sql.query(`
    SELECT t.*,
      (SELECT COUNT(*) FROM labeling_annotations WHERE task_id = t.id) as annotation_count
    FROM labeling_tasks t
    ORDER BY t.site_name, t.buffer_km, t.year, t.period
  `);
  return result.rows.map((row) => ({
    ...row,
    solar_polygon_pixels:
      typeof row.solar_polygon_pixels === "string"
        ? JSON.parse(row.solar_polygon_pixels)
        : row.solar_polygon_pixels,
  }));
}

export async function getLabelingTask(id: number) {
  const result = await sql`
    SELECT * FROM labeling_tasks WHERE id = ${id}
  `;
  if (result.rows.length === 0) return null;

  const row = result.rows[0];
  const annotations = await sql`
    SELECT * FROM labeling_annotations WHERE task_id = ${id} ORDER BY updated_at DESC
  `;

  return {
    ...row,
    solar_polygon_pixels:
      typeof row.solar_polygon_pixels === "string"
        ? JSON.parse(row.solar_polygon_pixels)
        : row.solar_polygon_pixels,
    annotations: annotations.rows.map((a) => ({
      ...a,
      regions: typeof a.regions === "string" ? JSON.parse(a.regions) : a.regions,
    })),
  };
}

export async function saveAnnotation(
  taskId: number,
  annotator: string,
  regions: unknown[]
) {
  // Upsert: one annotation per task per annotator
  const existing = await sql.query(
    `SELECT id FROM labeling_annotations WHERE task_id = $1 AND annotator = $2`,
    [taskId, annotator]
  );

  if (existing.rows.length > 0) {
    await sql.query(
      `UPDATE labeling_annotations SET regions = $1, updated_at = NOW() WHERE id = $2`,
      [JSON.stringify(regions), existing.rows[0].id]
    );
    return existing.rows[0].id;
  } else {
    const result = await sql.query(
      `INSERT INTO labeling_annotations (task_id, annotator, regions)
       VALUES ($1, $2, $3) RETURNING id`,
      [taskId, annotator, JSON.stringify(regions)]
    );
    return result.rows[0].id;
  }
}

export async function getAllAnnotations() {
  const result = await sql.query(`
    SELECT a.*, t.site_name, t.buffer_km, t.year, t.month, t.period,
           t.image_filename, t.image_width, t.image_height,
           t.bbox_west, t.bbox_south, t.bbox_east, t.bbox_north
    FROM labeling_annotations a
    JOIN labeling_tasks t ON a.task_id = t.id
    ORDER BY t.site_name, t.year
  `);
  return result.rows.map((row) => ({
    ...row,
    regions: typeof row.regions === "string" ? JSON.parse(row.regions) : row.regions,
  }));
}
