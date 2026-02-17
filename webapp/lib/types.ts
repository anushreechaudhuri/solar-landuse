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

// --- Labeling ---

export interface LabelingTask {
  id: number;
  site_name: string;
  site_display_name: string | null;
  buffer_km: number;
  year: number;
  month: number | null;
  period: string;
  image_filename: string;
  s3_key: string;
  image_width: number;
  image_height: number;
  bbox_west: number | null;
  bbox_south: number | null;
  bbox_east: number | null;
  bbox_north: number | null;
  solar_polygon_pixels: number[][][] | null;
  annotation_count?: number;
  created_at: string;
}

export interface AnnotationRegion {
  id: string;
  class_name: string;
  points: [number, number][];
}

export interface LabelingAnnotation {
  id: number;
  task_id: number;
  annotator: string;
  regions: AnnotationRegion[];
  created_at: string;
  updated_at: string;
}

export const LULC_CLASSES = [
  { name: "cropland", label: "Cropland", color: "#DDCC77" },
  { name: "trees", label: "Trees", color: "#117733" },
  { name: "shrub", label: "Shrub", color: "#999933" },
  { name: "grassland", label: "Grassland", color: "#44AA99" },
  { name: "flooded_veg", label: "Flooded Veg", color: "#332288" },
  { name: "built", label: "Built", color: "#CC6677" },
  { name: "bare", label: "Bare", color: "#882255" },
  { name: "water", label: "Water", color: "#88CCEE" },
  { name: "snow", label: "Snow/Ice", color: "#BBBBBB" },
  { name: "no_data", label: "No Data", color: "#DDDDDD" },
] as const;
