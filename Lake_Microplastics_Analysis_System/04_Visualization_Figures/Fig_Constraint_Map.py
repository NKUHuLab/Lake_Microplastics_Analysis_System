# -*- coding: utf-8 -*-
#
# SCRIPT to plot custom data from an Excel file onto a global map of lakes.
# This script first merges data from an Excel column with a base lake Shapefile
# and then generates a high-quality map and a separate legend file.
#
# VERSION 7.0:
#   1. Reverted the colormap to the user's originally requested color list.
#

import os
import warnings
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

# Ignore common warnings from geopandas and matplotlib for a cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# --- 1. CONFIGURATION SECTION ---
# ==============================================================================

# --- Core Input File Paths ---
# Define the root directory for your project
PROJECT_ROOT = r"E:\lake-MP-W"

# Path to the user-provided Excel data file
EXCEL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "yueshu", "CL.xlsx")

# Paths to the base geographic files
BASE_GEOGRAPHY_SHP_PATH = os.path.join(PROJECT_ROOT, "data", "base_shp", "predicted_lakes_2022.shp")
BORDERS_SHAPEFILE_PATH = os.path.join(PROJECT_ROOT, "data", "base_shp", "world map china line.shp")


# --- Dynamic Output File Paths ---
# Automatically derive output names from the input Excel file's name ("CL")
INPUT_BASENAME = os.path.splitext(os.path.basename(EXCEL_DATA_PATH))[0]

# Create a dedicated output directory for this analysis to keep files organized
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_maps", INPUT_BASENAME)

# Define full paths for all generated output files
GENERATED_SHP_PATH = os.path.join(OUTPUT_DIR, f"{INPUT_BASENAME}_spatially_joined.shp")
MAP_OUTPUT_PNG_PATH = os.path.join(OUTPUT_DIR, f"{INPUT_BASENAME}_map.png")
MAP_LEGEND_PDF_PATH = os.path.join(OUTPUT_DIR, f"{INPUT_BASENAME}_legend.pdf")


# --- Analysis and Plotting Parameters ---
# The specific column from the Excel file that you want to visualize
TARGET_COLUMN = 'P'

# --- **MODIFIED** Custom Colormap Definition (Version 7.0) ---
# Reverting to the user's originally specified color list.
CUSTOM_COLOR_LIST = [
    "#A6CEE3",
    "#FEF0D9",
    "#F1AB3D",
    "#DA3839"
]

# Map Appearance Settings
BORDER_COLOR = 'lightgrey'


# Matplotlib plotting parameters
PLT_RC_PARAMS = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42
}
plt.rcParams.update(PLT_RC_PARAMS)


# ==============================================================================
# --- 2. DATA MERGING FUNCTION (Unchanged) ---
# ==============================================================================

def create_data_shp_spatial():
    """
    Loads lake polygons and Excel data, then performs a SPATIAL JOIN based on
    lon/lat coordinates to accurately merge the datasets. Uses a 'left' join
    to ensure all original lake polygons are preserved.
    """
    print(f"--- Task 1: Merging data via Spatial Join ---")

    # 1. Load Data
    try:
        print(f"-> Reading base lake polygons from: {BASE_GEOGRAPHY_SHP_PATH}")
        gdf_lakes = gpd.read_file(BASE_GEOGRAPHY_SHP_PATH)

        print(f"-> Reading Excel data from: {EXCEL_DATA_PATH}")
        df_excel = pd.read_excel(EXCEL_DATA_PATH)
    except FileNotFoundError as e:
        print(f"\n[ERROR] A required input file was not found. Please check paths.\n  File: {e.filename}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Failed to load data: {e}")
        return False

    # 2. Validate Coordinate Data in Excel file
    if not all(col in df_excel.columns for col in ['lon', 'lat']):
        print(f"\n[ERROR] The Excel file '{os.path.basename(EXCEL_DATA_PATH)}' is missing required 'lon' or 'lat' columns.")
        return False
    df_excel.dropna(subset=['lon', 'lat'], inplace=True)

    # 3. Create a GeoDataFrame from the Excel data points
    print("-> Converting lon/lat data into geographic points...")
    gdf_points = gpd.GeoDataFrame(
        df_excel,
        geometry=gpd.points_from_xy(df_excel.lon, df_excel.lat)
    )

    # 4. Ensure Coordinate Reference Systems (CRS) match before joining
    if gdf_lakes.crs is None:
        print("[WARNING] Lake shapefile has no CRS defined. Assuming WGS84 (EPSG:4326).")
        gdf_lakes.set_crs(epsg=4326, inplace=True)
    gdf_points.set_crs(gdf_lakes.crs, inplace=True)
    print(f"-> Both layers set to CRS: {gdf_lakes.crs.name}")

    # 5. Perform the Spatial Join
    print("-> Performing 'left' spatial join to map data points to lake polygons...")
    merged_gdf = gpd.sjoin(gdf_lakes, gdf_points, how="left", predicate="contains")

    # 6. Handle cases where one lake contains multiple data points
    if merged_gdf.index.duplicated().any():
        print("-> Aggregating results for lakes with multiple data points (using mean)...")
        matched_indices = merged_gdf.dropna(subset=['index_right']).index.unique()
        matched_part = merged_gdf.loc[matched_indices]
        unmatched_part = merged_gdf[~merged_gdf.index.isin(matched_indices)].drop_duplicates(subset=['geometry'])

        aggregated_data = matched_part.groupby(matched_part.index).mean(numeric_only=True)

        original_matched_geometries = gdf_lakes.loc[aggregated_data.index].geometry
        aggregated_gdf = gpd.GeoDataFrame(aggregated_data, geometry=original_matched_geometries, crs=gdf_lakes.crs)

        original_cols = gdf_lakes.columns.drop('geometry')
        for col in original_cols:
            if col not in aggregated_gdf.columns:
                 aggregated_gdf[col] = gdf_lakes.loc[aggregated_gdf.index, col]

        merged_gdf = pd.concat([aggregated_gdf, unmatched_part.loc[:, gdf_lakes.columns]], ignore_index=True)


    if merged_gdf.empty:
        print("\n[WARNING] The spatial join resulted in no matches. No output will be created.")
        return False

    print(f"-> Spatial join successful. The final file will contain all {len(gdf_lakes)} original lakes.")

    # 7. Save the new spatially-joined Shapefile
    print(f"-> Saving merged Shapefile to: {GENERATED_SHP_PATH}")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        merged_gdf.drop(columns=['index_right'], inplace=True, errors='ignore')
        merged_gdf.to_file(GENERATED_SHP_PATH, driver='ESRI Shapefile', encoding='utf-8')
    except Exception as e:
        print(f"\n[ERROR] Could not save the new Shapefile: {e}")
        return False

    print("--- Data merging and Shapefile creation successful. ---\n")
    return True


# ==============================================================================
# --- 3. MAP PLOTTING FUNCTION (Unchanged) ---
# ==============================================================================

def plot_custom_map():
    """
    Generates the global map using the new shapefile.
    """
    print(f"--- Task 2: Plotting Global Map for '{TARGET_COLUMN}' ---")

    # 1. Load Data
    try:
        print(f"-> Loading generated Shapefile: {GENERATED_SHP_PATH}")
        gdf_lakes = gpd.read_file(GENERATED_SHP_PATH)
    except Exception as e:
        print(f"\n[ERROR] Failed to load the spatially-joined shapefile for plotting: {e}")
        return

    # 2. Prepare Data for Plotting with a FIXED 0-10 SCALE
    print("-> Applying fixed color scale from 0 to 10.")
    # The new color list is used here to create the colormap object
    custom_cmap = LinearSegmentedColormap.from_list("custom_gradient_map", CUSTOM_COLOR_LIST)
    norm = Normalize(vmin=0, vmax=10)

    # This process now handles polygons with data (P value) and without data (NaN)
    gdf_lakes['plot_value'] = pd.to_numeric(gdf_lakes[TARGET_COLUMN], errors='coerce')
    gdf_lakes['plot_value'].fillna(-9999, inplace=True) # Lakes without data now have plot_value = -9999

    # 3. Project Geographies
    print("-> Applying Robinson projection for a global view...")
    robinson_proj = "+proj=robin +datum=WGS84"
    gdf_lakes_proj = gdf_lakes.to_crs(robinson_proj)
    world_proj = gpd.read_file(BORDERS_SHAPEFILE_PATH).to_crs(robinson_proj)
    world_proj['geometry'] = world_proj.geometry.buffer(0)

    name_col = next((col for col in ['NAME', 'name', 'ADMIN', 'SOVEREIGNT'] if col in world_proj.columns), 'NAME')
    world_filtered = world_proj[world_proj[name_col] != 'Antarctica']
    world_for_borders = world_filtered[~world_filtered[name_col].str.contains("russia", case=False, na=False)]

    # 4. Plot the Map Layers
    print("-> Rendering map layers...")
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_facecolor('white')

    world_filtered.plot(ax=ax, color='white', edgecolor='none', zorder=1)

    # This selection now correctly finds all lakes that had no matching data point
    gdf_lakes_no_data = gdf_lakes_proj[gdf_lakes_proj['plot_value'] == -9999]
    # Unmatched lakes are plotted with a neutral grey
    if not gdf_lakes_no_data.empty:
        gdf_lakes_no_data.plot(color="#E7E7E7", linewidth=0, edgecolor='none', ax=ax, zorder=2)

    # This selection finds all lakes that had data and will be colored by the colormap
    gdf_lakes_with_data = gdf_lakes_proj[gdf_lakes_proj['plot_value'] != -9999]
    if not gdf_lakes_with_data.empty:
        gdf_lakes_with_data.plot(column='plot_value', cmap=custom_cmap, norm=norm,
                                 linewidth=0, edgecolor='none', ax=ax, legend=False, zorder=3)

    world_for_borders.plot(ax=ax, color='none', edgecolor=BORDER_COLOR, linewidth=0.5, zorder=4)
    ax.set_axis_off()

    # 5. Save Map Image
    print(f"-> Saving final map image to: {MAP_OUTPUT_PNG_PATH}")
    plt.savefig(MAP_OUTPUT_PNG_PATH, dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 6. Generate Legend
    print("-> Generating standalone legend file with 0-10 scale...")
    fig_legend = plt.figure(figsize=(2.5, 7))
    ax_legend = fig_legend.add_axes([0.05, 0.05, 0.2, 0.9])

    dummy_mappable = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap)

    cbar = fig_legend.colorbar(dummy_mappable, cax=ax_legend, orientation='vertical')
    cbar.set_label(f"{TARGET_COLUMN} Value", fontsize=14, weight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    print(f"-> Saving map legend to: {MAP_LEGEND_PDF_PATH}")
    fig_legend.savefig(MAP_LEGEND_PDF_PATH, format='pdf', bbox_inches='tight')
    plt.close(fig_legend)

    print("--- Map plotting finished successfully. ---\n")


# ==============================================================================
# --- 4. MAIN EXECUTION BLOCK ---
# ==============================================================================

if __name__ == '__main__':
    print("======================================================")
    print("--- Starting Custom Global Lake Map Generation (v7.0) ---")
    print(f"    Input Excel: {os.path.basename(EXCEL_DATA_PATH)}")
    print(f"    Target Column: '{TARGET_COLUMN}'")
    print("    Mapping Method: Left Spatial Join via lon/lat")
    print("    Color Scale: Fixed 0 to 10")
    print("======================================================\n")

    success = create_data_shp_spatial()

    if success:
        plot_custom_map()
    else:
        print("[PROCESS HALTED] The script stopped due to an error during data preparation.")

    print("--- Script finished. ---")