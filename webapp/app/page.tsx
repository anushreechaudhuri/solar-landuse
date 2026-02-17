"use client";

import { useState, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import SearchBar, { type Filters } from "@/components/SearchBar";
import ProjectList from "@/components/ProjectList";
import ProjectDetail from "@/components/ProjectDetail";
import ReviewPanel from "@/components/ReviewPanel";
import GrwFeatureList from "@/components/GrwFeatureList";
import GrwFeatureDetail from "@/components/GrwFeatureDetail";
import MergeDialog from "@/components/MergeDialog";
import type {
  Project,
  Review,
  GrwFeature,
  MergeHistoryEntry,
  GeoJSONGeometry,
  OverviewPoint,
  StatsResponse,
} from "@/lib/types";

// Dynamic import for map (no SSR — Leaflet needs window)
const MapView = dynamic(() => import("@/components/MapView"), { ssr: false });

const COUNTRIES = ["India", "Bangladesh", "Pakistan", "Nepal", "Sri Lanka", "Bhutan"];

type Tab = "active" | "proposed" | "grw";

export default function Home() {
  const [tab, setTab] = useState<Tab>("active");
  const [projects, setProjects] = useState<Project[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [projectDetail, setProjectDetail] = useState<{ project: Project; reviews: Review[] } | null>(null);
  const [search, setSearch] = useState("");
  const [filters, setFilters] = useState<Filters>({
    country: "",
    confidence: "",
    reviewed: "",
    sort: "capacity_mw",
    order: "DESC",
  });
  const [editMode, setEditMode] = useState<"none" | "edit" | "draw">("none");
  const [pendingPolygon, setPendingPolygon] = useState<GeoJSONGeometry | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);

  // GRW state
  const [grwFeatures, setGrwFeatures] = useState<GrwFeature[]>([]);
  const [grwTotal, setGrwTotal] = useState(0);
  const [grwPage, setGrwPage] = useState(1);
  const [grwLoading, setGrwLoading] = useState(false);
  const [selectedGrwFeature, setSelectedGrwFeature] = useState<GrwFeature | null>(null);
  const [grwFeatureDetail, setGrwFeatureDetail] = useState<(GrwFeature & { merge_history?: MergeHistoryEntry[] }) | null>(null);

  // Overview state
  const [showOverview, setShowOverview] = useState(false);
  const [overviewPoints, setOverviewPoints] = useState<OverviewPoint[]>([]);

  // Merge dialog state
  const [mergeDialog, setMergeDialog] = useState<{
    mode: "grw_to_gem" | "gem_to_grw";
    lat: number;
    lon: number;
    label: string;
    id: string | number;
  } | null>(null);

  // Fetch projects
  const fetchProjects = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(page),
        per_page: "50",
        tab,
        sort: filters.sort,
        order: filters.order,
      });
      if (filters.country) params.set("country", filters.country);
      if (filters.confidence) params.set("confidence", filters.confidence);
      if (filters.reviewed) params.set("reviewed", filters.reviewed);
      if (search) params.set("search", search);

      const res = await fetch(`/api/projects?${params}`);
      const data = await res.json();
      if (data.projects) {
        setProjects(data.projects);
        setTotal(data.total);
      }
    } catch (err) {
      console.error("Failed to fetch projects:", err);
    } finally {
      setLoading(false);
    }
  }, [page, tab, filters, search]);

  // Fetch GRW features
  const fetchGrwFeatures = useCallback(async () => {
    setGrwLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(grwPage),
        per_page: "50",
        linked: "false",
        sort: "area_m2",
        order: "DESC",
      });
      if (filters.country) params.set("country", filters.country);
      if (search) params.set("search", search);

      const res = await fetch(`/api/grw-features?${params}`);
      const data = await res.json();
      if (data.features) {
        setGrwFeatures(data.features);
        setGrwTotal(data.total);
      }
    } catch (err) {
      console.error("Failed to fetch GRW features:", err);
    } finally {
      setGrwLoading(false);
    }
  }, [grwPage, filters.country, search]);

  useEffect(() => {
    if (tab === "grw") {
      fetchGrwFeatures();
    } else {
      fetchProjects();
    }
  }, [tab, fetchProjects, fetchGrwFeatures]);

  // Fetch stats
  useEffect(() => {
    fetch("/api/stats")
      .then((r) => r.json())
      .then((data) => {
        if (data.total_projects !== undefined) setStats(data);
      })
      .catch(console.error);
  }, []);

  // Fetch overview points
  useEffect(() => {
    if (!showOverview) return;
    fetch("/api/overview")
      .then((r) => r.json())
      .then((data) => {
        if (data.points) setOverviewPoints(data.points);
      })
      .catch(console.error);
  }, [showOverview]);

  // Fetch project detail when selected
  const selectProject = useCallback(async (project: Project) => {
    setSelectedProject(project);
    setSelectedGrwFeature(null);
    setGrwFeatureDetail(null);
    setMergeDialog(null);
    setEditMode("none");
    setPendingPolygon(null);
    try {
      const res = await fetch(`/api/projects/${project.id}`);
      const data = await res.json();
      setProjectDetail({ project: data, reviews: data.reviews || [] });
    } catch (err) {
      console.error("Failed to fetch project detail:", err);
      setProjectDetail({ project, reviews: [] });
    }
  }, []);

  // Fetch GRW feature detail when selected
  const selectGrwFeature = useCallback(async (feature: GrwFeature) => {
    setSelectedGrwFeature(feature);
    setSelectedProject(null);
    setProjectDetail(null);
    setMergeDialog(null);
    setEditMode("none");
    setPendingPolygon(null);
    try {
      const res = await fetch(`/api/grw-features/${feature.id}`);
      const data = await res.json();
      setGrwFeatureDetail(data);
    } catch (err) {
      console.error("Failed to fetch GRW feature detail:", err);
      setGrwFeatureDetail(feature);
    }
  }, []);

  const handleSearch = useCallback((q: string) => {
    setSearch(q);
    setPage(1);
    setGrwPage(1);
  }, []);

  const handleFilterChange = useCallback((f: Filters) => {
    setFilters(f);
    setPage(1);
    setGrwPage(1);
  }, []);

  const handlePolygonEdited = useCallback((geometry: GeoJSONGeometry) => {
    setPendingPolygon(geometry);
  }, []);

  const handlePolygonDrawn = useCallback((geometry: GeoJSONGeometry) => {
    setPendingPolygon(geometry);
  }, []);

  const handleReviewSubmitted = useCallback(() => {
    fetchProjects();
    if (selectedProject) {
      selectProject(selectedProject);
    }
  }, [fetchProjects, selectedProject, selectProject]);

  // Handle overview point click
  const handleOverviewPointClick = useCallback(
    (point: OverviewPoint) => {
      if (point.type === "grw_only") {
        // Switch to GRW tab and select
        setTab("grw");
        const grwId = parseInt(String(point.id));
        fetch(`/api/grw-features/${grwId}`)
          .then((r) => r.json())
          .then((data) => {
            if (data.id) {
              setSelectedGrwFeature(data);
              setGrwFeatureDetail(data);
              setSelectedProject(null);
              setProjectDetail(null);
            }
          })
          .catch(console.error);
      } else {
        // Switch to active tab and select
        if (tab === "grw") setTab("active");
        fetch(`/api/projects/${point.id}`)
          .then((r) => r.json())
          .then((data) => {
            if (data.id) {
              setSelectedProject(data);
              setProjectDetail({ project: data, reviews: data.reviews || [] });
              setSelectedGrwFeature(null);
              setGrwFeatureDetail(null);
            }
          })
          .catch(console.error);
      }
    },
    [tab]
  );

  // Handle merge
  const handleMerge = useCallback(
    async (targetId: string | number) => {
      if (!mergeDialog) return;
      const performedBy = localStorage.getItem("reviewer_name") || "anonymous";

      try {
        if (mergeDialog.mode === "grw_to_gem") {
          await fetch("/api/merge", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              grw_feature_id: mergeDialog.id,
              project_id: targetId,
              performed_by: performedBy,
            }),
          });
        } else {
          await fetch("/api/merge", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              grw_feature_id: targetId,
              project_id: mergeDialog.id,
              performed_by: performedBy,
            }),
          });
        }
        setMergeDialog(null);
        // Refresh
        if (tab === "grw") fetchGrwFeatures();
        else fetchProjects();
        if (selectedGrwFeature) selectGrwFeature(selectedGrwFeature);
        if (selectedProject) selectProject(selectedProject);
      } catch (err) {
        console.error("Merge failed:", err);
      }
    },
    [mergeDialog, tab, fetchGrwFeatures, fetchProjects, selectedGrwFeature, selectGrwFeature, selectedProject, selectProject]
  );

  const handleChangeTab = useCallback((newTab: Tab) => {
    setTab(newTab);
    setPage(1);
    setGrwPage(1);
    setSelectedProject(null);
    setProjectDetail(null);
    setSelectedGrwFeature(null);
    setGrwFeatureDetail(null);
    setMergeDialog(null);
    setEditMode("none");
  }, []);

  return (
    <div className="h-screen flex flex-col">
      {/* Top bar */}
      <header className="bg-white border-b border-gray-200 px-4 py-2 flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-lg font-bold text-gray-900">
            Solar Polygon Verification
          </h1>
          <p className="text-xs text-gray-500">
            South Asia &middot;{" "}
            {stats
              ? `${stats.total_projects.toLocaleString()} projects | ${stats.reviewed} reviewed | ${stats.total_grw_unmatched} unmatched GRW`
              : "Loading..."}
          </p>
        </div>
        <div className="flex items-center gap-4">
          {stats && (
            <div className="flex gap-4 text-xs text-gray-500">
              {Object.entries(stats.by_country)
                .slice(0, 4)
                .map(([country, count]) => (
                  <span key={country}>
                    {country}: {count.toLocaleString()}
                  </span>
                ))}
            </div>
          )}
          <button
            onClick={() => setShowOverview(!showOverview)}
            className={`px-3 py-1.5 text-xs font-medium rounded border transition-colors ${
              showOverview
                ? "bg-purple-600 text-white border-purple-600"
                : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
            }`}
          >
            {showOverview ? "Hide Overview" : "Show All Points"}
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <div className="w-96 border-r border-gray-200 bg-white flex flex-col flex-shrink-0">
          {/* Tabs */}
          <div className="flex border-b border-gray-200">
            <button
              onClick={() => handleChangeTab("active")}
              className={`flex-1 px-3 py-2.5 text-sm font-medium ${
                tab === "active"
                  ? "text-blue-600 border-b-2 border-blue-600"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              Active
            </button>
            <button
              onClick={() => handleChangeTab("proposed")}
              className={`flex-1 px-3 py-2.5 text-sm font-medium ${
                tab === "proposed"
                  ? "text-blue-600 border-b-2 border-blue-600"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              Proposed
            </button>
            <button
              onClick={() => handleChangeTab("grw")}
              className={`flex-1 px-3 py-2.5 text-sm font-medium ${
                tab === "grw"
                  ? "text-purple-600 border-b-2 border-purple-600"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              GRW
              {stats && stats.total_grw_unmatched > 0 && (
                <span className="ml-1 text-[10px] bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded-full">
                  {stats.total_grw_unmatched}
                </span>
              )}
            </button>
          </div>

          <SearchBar
            onSearch={handleSearch}
            onFilterChange={handleFilterChange}
            filters={filters}
            countries={COUNTRIES}
          />

          <div className="flex-1 overflow-hidden">
            {tab === "grw" ? (
              <GrwFeatureList
                features={grwFeatures}
                selectedId={selectedGrwFeature?.id || null}
                onSelect={selectGrwFeature}
                loading={grwLoading}
                total={grwTotal}
                page={grwPage}
                perPage={50}
                onPageChange={setGrwPage}
              />
            ) : (
              <ProjectList
                projects={projects}
                selectedId={selectedProject?.id || null}
                onSelect={selectProject}
                loading={loading}
                total={total}
                page={page}
                perPage={50}
                onPageChange={setPage}
              />
            )}
          </div>

          {/* Detail panels */}
          {tab === "grw" && grwFeatureDetail && selectedGrwFeature && (
            <>
              <GrwFeatureDetail
                feature={grwFeatureDetail}
                onClose={() => {
                  setSelectedGrwFeature(null);
                  setGrwFeatureDetail(null);
                  setMergeDialog(null);
                }}
                onUpdated={() => {
                  fetchGrwFeatures();
                  if (selectedGrwFeature) selectGrwFeature(selectedGrwFeature);
                }}
                onMergeClick={() => {
                  setMergeDialog({
                    mode: "grw_to_gem",
                    lat: selectedGrwFeature.centroid_lat,
                    lon: selectedGrwFeature.centroid_lon,
                    label: selectedGrwFeature.user_name || `GRW #${selectedGrwFeature.fid}`,
                    id: selectedGrwFeature.id,
                  });
                }}
              />
              {mergeDialog && (
                <MergeDialog
                  mode={mergeDialog.mode}
                  sourceLat={mergeDialog.lat}
                  sourceLon={mergeDialog.lon}
                  sourceLabel={mergeDialog.label}
                  sourceId={mergeDialog.id}
                  onMerge={handleMerge}
                  onCancel={() => setMergeDialog(null)}
                  onPreview={() => {}}
                />
              )}
            </>
          )}

          {tab !== "grw" && projectDetail && selectedProject && (
            <>
              <ProjectDetail
                project={projectDetail.project}
                reviews={projectDetail.reviews}
                onClose={() => {
                  setSelectedProject(null);
                  setProjectDetail(null);
                  setMergeDialog(null);
                  setEditMode("none");
                }}
                onMergeClick={
                  !selectedProject.merged_polygon
                    ? () => {
                        setMergeDialog({
                          mode: "gem_to_grw",
                          lat: selectedProject.latitude,
                          lon: selectedProject.longitude,
                          label: selectedProject.project_name,
                          id: selectedProject.id,
                        });
                      }
                    : undefined
                }
              />
              {mergeDialog ? (
                <MergeDialog
                  mode={mergeDialog.mode}
                  sourceLat={mergeDialog.lat}
                  sourceLon={mergeDialog.lon}
                  sourceLabel={mergeDialog.label}
                  sourceId={mergeDialog.id}
                  onMerge={handleMerge}
                  onCancel={() => setMergeDialog(null)}
                  onPreview={() => {}}
                />
              ) : (
                <ReviewPanel
                  project={selectedProject}
                  isActiveTab={tab === "active"}
                  editMode={editMode}
                  onEditModeChange={setEditMode}
                  pendingPolygon={pendingPolygon}
                  onReviewSubmitted={handleReviewSubmitted}
                />
              )}
            </>
          )}
        </div>

        {/* Map */}
        <div className="flex-1 relative">
          <MapView
            selectedProject={selectedProject}
            selectedGrwFeature={selectedGrwFeature}
            editMode={editMode}
            onPolygonEdited={handlePolygonEdited}
            onPolygonDrawn={handlePolygonDrawn}
            showOverview={showOverview}
            overviewPoints={overviewPoints}
            onOverviewPointClick={handleOverviewPointClick}
          />
        </div>
      </div>
    </div>
  );
}
