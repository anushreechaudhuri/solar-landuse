# Satellite Imagery Data Sources

Technical documentation for all downloaded satellite imagery files.

## Project Details
- Site: Mongla Solar Project, Bangladesh
- Coordinates: 22.573611°N, 89.569444°E (22°34'25.0"N, 89°34'10.4"E)
- Datasets: Sentinel-2 MSI and Landsat 8 OLI

## File Specifications

### mongla_1km_2019.tif
- Year: 2019
- Buffer: 1km radius
- Data Source: `COPERNICUS/S2_SR_HARMONIZED`
- Number of source images: 93
- Resolution: 10m per pixel
- Bands: B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)
- Cloud filter: <50%
- Composite method: Median
- File size: 0.43 MB
- Dimensions: 217 x 201 pixels
- Coordinate system: EPSG:4326 (WGS84)

### mongla_5km_2019.tif
- Year: 2019
- Buffer: 5km radius
- Data Source: `COPERNICUS/S2_SR_HARMONIZED`
- Number of source images: 93
- Resolution: 10m per pixel
- Bands: B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)
- Cloud filter: <50%
- Composite method: Median
- File size: 9.7 MB
- Dimensions: 1079 x 1003 pixels
- Coordinate system: EPSG:4326 (WGS84)

### mongla_1km_2023.tif
- Year: 2023
- Buffer: 1km radius
- Data Source: `COPERNICUS/S2_SR_HARMONIZED`
- Number of source images: 70
- Resolution: 10m per pixel
- Bands: B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)
- Cloud filter: <50%
- Composite method: Median
- File size: 0.48 MB
- Dimensions: 217 x 201 pixels
- Coordinate system: EPSG:4326 (WGS84)

### mongla_5km_2023.tif
- Year: 2023
- Buffer: 5km radius
- Data Source: `COPERNICUS/S2_SR_HARMONIZED`
- Number of source images: 70
- Resolution: 10m per pixel
- Bands: B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)
- Cloud filter: <50%
- Composite method: Median
- File size: 10.2 MB
- Dimensions: 1079 x 1003 pixels
- Coordinate system: EPSG:4326 (WGS84)

### mongla_2014.tif through mongla_2023.tif
- Years: 2014, 2016-2017, 2019-2023
- Buffer: 10km radius
- Data Source: `COPERNICUS/S2_SR_HARMONIZED` (2015+), `LANDSAT/LC08/C02/T1_L2` (2014)
- Resolution: 20m per pixel (Sentinel-2) or 30m per pixel (Landsat 8)
- Bands: Sentinel-2: B4, B3, B2, B8; Landsat 8: SR_B4, SR_B3, SR_B2, SR_B5
- Cloud filter: <50%
- Composite method: Median
- File sizes: ~8-10 MB
- Dimensions: ~2000 x 2000 pixels
- Coordinate system: EPSG:4326 (WGS84)

## Processing Details

All images processed using Google Earth Engine:
- Cloud filtering: <50% cloud cover per image
- Composite method: Median (reduces cloud contamination)
- Resolution: Native export (10m/pixel for Sentinel-2, 30m/pixel for Landsat 8)
- Coordinate system: EPSG:4326 (WGS84)
- Spatial coverage: 1km buffer ≈ 3.14 km², 5km buffer ≈ 78.5 km², 10km buffer ≈ 314 km²
- Temporal coverage: One composite per year representing median of all clear-sky acquisitions

## Use Cases

1km images: Solar panel detection, individual building classification, narrow road identification
5km images: Regional land use patterns, watershed analysis, agricultural vs. urban transitions
10km images: Broader regional context
