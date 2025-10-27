# Satellite Imagery Data Sources

This document describes all downloaded satellite imagery files for the Mongla solar project site in Bangladesh.

## Project Details
- **Site**: Mongla Solar Project, Bangladesh  
- **Coordinates**: 22°34'25.0"N, 89°34'10.4"E (22.573611°N, 89.569444°E)
- **Datasets**: Sentinel-2 MSI and Landsat 8 OLI

---

## mongla_1km_2019.tif

- **Year**: 2019
- **Buffer**: 1km radius
- **Data Source**: `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Surface Reflectance Harmonized)
- **Number of Images**: 93
- **Resolution**: 10m per pixel
- **Bands**: B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)
- **Cloud Filter**: <50%
- **Composite Method**: Median
- **File Size**: 0.43 MB
- **Dimensions**: 217 x 201 pixels
- **Purpose**: High-resolution view for detailed feature labeling

---

## mongla_5km_2019.tif

- **Year**: 2019
- **Buffer**: 5km radius
- **Data Source**: `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Surface Reflectance Harmonized)
- **Number of Images**: 93
- **Resolution**: 10m per pixel
- **Bands**: B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)
- **Cloud Filter**: <50%
- **Composite Method**: Median
- **File Size**: 9.7 MB
- **Dimensions**: 1079 x 1003 pixels
- **Purpose**: Regional context with high detail for land use classification

---

## mongla_1km_2023.tif

- **Year**: 2023
- **Buffer**: 1km radius
- **Data Source**: `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Surface Reflectance Harmonized)
- **Number of Images**: 70
- **Resolution**: 10m per pixel
- **Bands**: B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)
- **Cloud Filter**: <50%
- **Composite Method**: Median
- **File Size**: 0.48 MB
- **Dimensions**: 217 x 201 pixels
- **Purpose**: Recent high-resolution view for post-development comparison

---

## mongla_5km_2023.tif

- **Year**: 2023
- **Buffer**: 5km radius
- **Data Source**: `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Surface Reflectance Harmonized)
- **Number of Images**: 70
- **Composite Method**: Median
- **File Size**: 10.2 MB
- **Dimensions**: 1079 x 1003 pixels
- **Purpose**: Post-development regional context for land use change analysis

---

## Resolution Notes

- **10m resolution**: Maximum available for Sentinel-2 RGB+NIR bands at native resolution
- **Spatial coverage**: 
  - 1km buffer ≈ 3.14 km² area
  - 5km buffer ≈ 78.5 km² area
- **Temporal coverage**: One composite per year representing the median of all clear-sky acquisitions

## Data Processing

All images were processed using Google Earth Engine with:
- Cloud filtering (<50% cloud cover per image)
- Median compositing to reduce cloud contamination
- Native resolution export (10m/pixel for Sentinel-2)
- Coordinate system: EPSG:4326 (WGS84)

## Use Cases

- **1km images**: Best for detailed solar panel detection, individual building classification, narrow road identification
- **5km images**: Best for regional land use patterns, watershed analysis, agricultural vs. urban transitions


