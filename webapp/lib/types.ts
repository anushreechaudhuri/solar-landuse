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
  total_grw_unmatched: number;
}

export interface GrwFeature {
  id: number;
  fid: number | null;
  country: string | null;
  centroid_lat: number;
  centroid_lon: number;
  area_m2: number | null;
  construction_year: number | null;
  construction_quarter: number | null;
  landcover: string | null;
  polygon: GeoJSONFeature;
  user_name: string | null;
  user_capacity_mw: number | null;
  user_status: string | null;
  user_notes: string | null;
  linked_project_id: string | null;
  linked_at: string | null;
}

export interface GrwFeaturesResponse {
  features: GrwFeature[];
  total: number;
  page: number;
  per_page: number;
}

export interface OverviewPoint {
  id: string | number;
  lat: number;
  lon: number;
  type: "matched" | "gem_only" | "grw_only";
  label: string;
}

export interface MergeHistoryEntry {
  id: number;
  grw_feature_id: number;
  project_id: string;
  action: string;
  performed_by: string | null;
  notes: string | null;
  created_at: string;
}
