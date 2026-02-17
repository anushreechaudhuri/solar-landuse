"use client";

import { useState, useEffect, useRef } from "react";

interface SearchBarProps {
  onSearch: (query: string) => void;
  onFilterChange: (filters: Filters) => void;
  filters: Filters;
  countries: string[];
}

export interface Filters {
  country: string;
  confidence: string;
  reviewed: string;
  sort: string;
  order: string;
}

export default function SearchBar({
  onSearch,
  onFilterChange,
  filters,
  countries,
}: SearchBarProps) {
  const [searchText, setSearchText] = useState("");
  const debounceRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      onSearch(searchText);
    }, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [searchText, onSearch]);

  return (
    <div className="p-3 border-b border-gray-200 space-y-2">
      <input
        type="text"
        placeholder="Search name, ID, capacity..."
        value={searchText}
        onChange={(e) => setSearchText(e.target.value)}
        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
      <div className="flex gap-2 flex-wrap">
        <select
          value={filters.country}
          onChange={(e) => onFilterChange({ ...filters, country: e.target.value })}
          className="text-xs px-2 py-1 border border-gray-300 rounded"
        >
          <option value="">All countries</option>
          {countries.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <select
          value={filters.confidence}
          onChange={(e) => onFilterChange({ ...filters, confidence: e.target.value })}
          className="text-xs px-2 py-1 border border-gray-300 rounded"
        >
          <option value="">All confidence</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
          <option value="none">None</option>
        </select>
        <select
          value={filters.reviewed}
          onChange={(e) => onFilterChange({ ...filters, reviewed: e.target.value })}
          className="text-xs px-2 py-1 border border-gray-300 rounded"
        >
          <option value="">All review status</option>
          <option value="true">Reviewed</option>
          <option value="false">Unreviewed</option>
        </select>
        <select
          value={`${filters.sort}_${filters.order}`}
          onChange={(e) => {
            const [sort, order] = e.target.value.split("_");
            onFilterChange({ ...filters, sort, order });
          }}
          className="text-xs px-2 py-1 border border-gray-300 rounded"
        >
          <option value="capacity_mw_DESC">Capacity (largest)</option>
          <option value="capacity_mw_ASC">Capacity (smallest)</option>
          <option value="project_name_ASC">Name (A-Z)</option>
          <option value="start_year_DESC">Year (newest)</option>
          <option value="start_year_ASC">Year (oldest)</option>
          <option value="match_confidence_DESC">Confidence</option>
        </select>
      </div>
    </div>
  );
}
