# Importing Images to Label Studio

Images in this directory can be imported directly into Label Studio.

## Method 1: Import from Directory

1. Open Label Studio: http://localhost:8080
2. Create or open your project
3. Go to "Import" tab
4. Select "Import from local files"
5. Navigate to: `data/for_labeling/`
6. Select PNG files to import

## Method 2: Drag and Drop

1. Open Label Studio: http://localhost:8080
2. Create or open your project
3. Go to "Import" tab
4. Drag and drop PNG files from this directory

## Method 3: Use Absolute Path

If Label Studio doesn't see files, use absolute path:

```
/Users/anushreechaudhuri/Documents/Projects/solar-landuse/data/for_labeling/
```

## Available Images

Current images in this directory:
- `mongla_1km_2019.png` - 1km buffer, 2019
- `mongla_1km_2023.png` - 1km buffer, 2023
- `mongla_5km_2019.png` - 5km buffer, 2019
- `mongla_5km_2023.png` - 5km buffer, 2023
- Plus older 10km buffer images from initial download

## Troubleshooting

**Images not showing in Label Studio:**
- Check Label Studio is running: `label-studio start`
- Verify files exist: `ls data/for_labeling/*.png`
- Try using absolute path instead of relative
- Check file permissions (should be readable)

