export interface SolarSite {
  id: string;
  name: string;
  capacity_mw: number;
  lat: number | null;
  lon: number | null;
  district: string;
  upazilla: string;
  status: string;
  completion_date: string | null;
  has_conflict: boolean;
  conflict_reasons: string;
  conflict_tags: string[];
  news_links: string[];
  google_maps_link: string | null;
  gem_url: string | null;
  developer: string;
  financing: string;
  polygon: { type: "Polygon"; coordinates: number[][][] } | null;
  matched_site_id: string | null;
  post_lulc: Record<string, number> | null;
  annual_lulc: Array<{
    year: number;
    crops: number;
    trees: number;
    built: number;
    bare: number;
    water: number;
    grass: number;
    shrub: number;
    flooded_veg: number;
    ndvi: number;
  }> | null;
  images: {
    pre_post: string | null;
    image_grid: string | null;
    lulc_maps: string | null;
    lulc_timeseries: string | null;
  };
}
