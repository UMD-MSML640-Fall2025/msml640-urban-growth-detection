# Google Earth Engine Export & Download Instructions

MSML 640 – Urban Growth Detection Project
This guide explains how to export and download the 10 AOI (Before & After) Sentinel-2 composites from Google Earth Engine into the project folder.

## 1. Open the Export Script in Google Earth Engine

Go to: https://code.earthengine.google.com

Open the script file:
src/gee/export_10_aois_before_after.js

Make sure the 10 AOI polygons (aoi1…aoi10) are imported or drawn in the GEE editor.

## 2. Run the Export Script

Click Run at the top of the GEE Code Editor.

Go to the Tasks tab (top-right).

You should see 20 export tasks automatically created:

10 BEFORE composites

10 AFTER composites

For each task:

Click RUN

Confirm export settings in the pop-up

Click RUN again

Notes:

Exports go to Google Drive under GEE_UrbanGrowth_10AOI

Each file is a cloud-masked, median Sentinel-2 composite (10 m resolution)

Typical export time: 1–5 minutes per AOI

## 3. Download the TIFF Files From Google Drive

Once all tasks finish:

Open Google Drive → GEE_UrbanGrowth_10AOI

Download all 20 files:

*_before.tif

*_after.tif

## 4. Place the TIFF Files in the Project Folder

Place files as follows:

MSML640-URBAN-GROWTH/
│
├── data/
│   ├── gee_exports/
│   │   ├── raw_before/
│   │   │   ├── Austin_Pflugerville_before.tif
│   │   │   ├── Dallas_Frisco_before.tif
│   │   │   └── ...
│   │   ├── raw_after/
│   │   │   ├── Austin_Pflugerville_after.tif
│   │   │   ├── Dallas_Frisco_after.tif
│   │   │   └── ...
│   │   ├── region_shapes/
│   │   └── metadata/


raw_before/ contains all BEFORE TIFFs

raw_after/ contains all AFTER TIFFs

Do NOT modify these raw files — all processing will read from here.

## 5. (Optional) Export AOI Shapes as GeoJSON

For reproducibility:

In GEE, right-click each AOI geometry

Choose Export Table to Drive

Select GeoJSON

Place the exported shapes here:

data/gee_exports/region_shapes/


These are used for masking, cropping, and map visualization.

## 6. After Download – Next Steps

Once the TIFFs are in raw_before/ and raw_after/, you can proceed to:

src/preprocessing/ → NDVI/NDBI computation

src/change_detection/ → Difference maps

src/models/ → Tabular ML model

src/inference/ → Predictions

Your dataset is now ready for the full analysis pipeline.

## Troubleshooting
Issue	Solution
Task shows Failed: incompatible bands	Use the updated SCL-based cloud mask code
“Task not submitted”	You must manually press RUN
TIFF looks black	Apply RGB stretch (bands 4/3/2, min 0.02, max 0.30)
Missing files	Ensure all 20 tasks finished successfully