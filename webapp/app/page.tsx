"use client";

import { useState, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import SearchBar, { type Filters } from "@/components/SearchBar";
import ProjectList from "@/components/ProjectList";
import ProjectDetail from "@/components/ProjectDetail";
import ReviewPanel from "@/components/ReviewPanel";
import type { Project, Review, GeoJSONGeometry, ProjectsResponse, StatsResponse } from "@/lib/types";

// Dynamic import for map (no SSR — Leaflet needs window)
const MapView = dynamic(() => import("@/components/MapView"), { ssr: false });

const COUNTRIES = ["India", "Bangladesh", "Pakistan", "Nepal", "Sri Lanka", "Bhutan"];

export default function Home() {
  const [tab, setTab] = useState<"active" | "proposed">("active");
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

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  // Fetch stats
  useEffect(() => {
    fetch("/api/stats")
      .then((r) => r.json())
      .then((data) => {
        if (data.total_projects !== undefined) setStats(data);
      })
      .catch(console.error);
  }, []);

  // Fetch project detail when selected
  const selectProject = useCallback(async (project: Project) => {
    setSelectedProject(project);
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

  const handleSearch = useCallback((q: string) => {
    setSearch(q);
    setPage(1);
  }, []);

  const handleFilterChange = useCallback((f: Filters) => {
    setFilters(f);
    setPage(1);
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
              ? `${stats.total_projects.toLocaleString()} projects | ${stats.reviewed} reviewed`
              : "Loading..."}
          </p>
        </div>
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
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <div className="w-96 border-r border-gray-200 bg-white flex flex-col flex-shrink-0">
          {/* Tabs */}
          <div className="flex border-b border-gray-200">
            <button
              onClick={() => { setTab("active"); setPage(1); }}
              className={`flex-1 px-4 py-2.5 text-sm font-medium ${
                tab === "active"
                  ? "text-blue-600 border-b-2 border-blue-600"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              Active Projects
            </button>
            <button
              onClick={() => { setTab("proposed"); setPage(1); }}
              className={`flex-1 px-4 py-2.5 text-sm font-medium ${
                tab === "proposed"
                  ? "text-blue-600 border-b-2 border-blue-600"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              Proposed/Other
            </button>
          </div>

          <SearchBar
            onSearch={handleSearch}
            onFilterChange={handleFilterChange}
            filters={filters}
            countries={COUNTRIES}
          />

          <div className="flex-1 overflow-hidden">
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
          </div>

          {/* Detail + review panel for selected project */}
          {projectDetail && selectedProject && (
            <>
              <ProjectDetail
                project={projectDetail.project}
                reviews={projectDetail.reviews}
                onClose={() => {
                  setSelectedProject(null);
                  setProjectDetail(null);
                  setEditMode("none");
                }}
              />
              <ReviewPanel
                project={selectedProject}
                isActiveTab={tab === "active"}
                editMode={editMode}
                onEditModeChange={setEditMode}
                pendingPolygon={pendingPolygon}
                onReviewSubmitted={handleReviewSubmitted}
              />
            </>
          )}
        </div>

        {/* Map */}
        <div className="flex-1 relative">
          <MapView
            selectedProject={selectedProject}
            editMode={editMode}
            onPolygonEdited={handlePolygonEdited}
            onPolygonDrawn={handlePolygonDrawn}
          />
        </div>
      </div>
    </div>
  );
}
