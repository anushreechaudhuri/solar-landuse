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
  const [total, byCountry, byStatus, byConfidence, reviewed] =
    await Promise.all([
      sql`SELECT COUNT(*) as count FROM projects`,
      sql`SELECT country, COUNT(*) as count FROM projects GROUP BY country ORDER BY count DESC`,
      sql`SELECT status, COUNT(*) as count FROM projects GROUP BY status ORDER BY count DESC`,
      sql`SELECT match_confidence, COUNT(*) as count FROM projects GROUP BY match_confidence ORDER BY count DESC`,
      sql`SELECT COUNT(DISTINCT project_id) as count FROM reviews`,
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
  };
}
