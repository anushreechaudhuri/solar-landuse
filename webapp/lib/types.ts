export interface Project {
  id: string;
  gem_location_id: string | null;
  project_name: string;
  phase_name: string | null;
  country: string;
  state_province: string | null;
  capacity_mw: number;
  capacity_rating: string | null;
  status: string;
  start_year: number | null;
  latitude: number;
  longitude: number;
  location_accuracy: string | null;
  owner: string | null;
  operator: string | null;
  other_ids: string | null;
  wiki_url: string | null;
  grw_polygons: GeoJSONFeature[] | null;
  merged_polygon: GeoJSONFeature | null;
  match_confidence: string | null;
  match_distance_km: number | null;
  grw_construction_date: string | null;
  // Computed fields from reviews
  latest_review?: Review | null;
  review_count?: number;
}

export interface Review {
  id: number;
  project_id: string;
  reviewer_name: string;
  action: string;
  polygon: GeoJSONGeometry | null;
  notes: string | null;
  created_at: string;
}

export interface GeoJSONFeature {
  type: "Feature";
  geometry: GeoJSONGeometry;
  properties: Record<string, unknown>;
}

export interface GeoJSONGeometry {
  type: string;
  coordinates: number[][][] | number[][][][];
}

export interface ProjectsResponse {
  projects: Project[];
  total: number;
  page: number;
  per_page: number;
}

export interface StatsResponse {
  total_projects: number;
  by_country: Record<string, number>;
  by_status: Record<string, number>;
  by_confidence: Record<string, number>;
  reviewed: number;
  unreviewed: number;
}
