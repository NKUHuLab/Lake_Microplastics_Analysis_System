# -*- coding: utf-8 -*-
#
# Combined script to generate a global map of microplastic reduction potential in lakes.
# VERSION 5: Uses an enhanced, multi-color gradient for clearer visualization.
#
# This script loads processed data, joins it with geographic lake boundaries,
# transforms the data using a natural log scale, and plots the result on a world map.
#

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import warnings
import os

# --- Configuration Section ---

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set global plotting parameters for font and vector graphics consistency
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# --- Core File Paths ---
# Define the root directory for your project
PROJECT_ROOT = r"E:\lake-MP-W"

# 1. **INPUT**: Path to your data file with 'lon', 'lat', and 'change' columns.
REDUCTION_POTENTIAL_CSV = os.path.join(PROJECT_ROOT, "data", "opt", "data", "processed_output", "Ychange.csv")

# 2. **INPUT**: Path to the base Shapefile containing the lake/watershed polygons to be colored.
BASE_LAKES_SHP = os.path.join(PROJECT_ROOT, "data", "base_shp", "predicted_lakes_2022.shp")

# 3. **INPUT**: Path to the world map borders Shapefile.
WORLD_BORDERS_SHP = os.path.join(PROJECT_ROOT, "data", "base_shp", "world map china line.shp")

# 4. **OUTPUT**: Directory where the final map and legend will be saved.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "draw", "09_Reduction_Potential_Map")


# --- Helper Function ---

def ensure_directories_exist():
    """Checks if the output directory exists and creates it if it doesn't."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")


# --- Main Plotting Function ---

def create_reduction_potential_map():
    """
    Main function to load data, perform spatial join, and generate the final
    global map of microplastic reduction potential.
    """
    print("--- Starting: Microplastic Reduction Potential Map Generation (v5) ---")

    # 1. Load Data Files
    print("Step 1: Loading data files...")
    try:
        data_df = pd.read_csv(REDUCTION_POTENTIAL_CSV)
        lakes_geo_df = gpd.read_file(BASE_LAKES_SHP)
        world_gdf = gpd.read_file(WORLD_BORDERS_SHP)
    except Exception as e:
        print(f"FATAL ERROR: Failed to load a required file. Please check paths. Error: {e}")
        return

    # 2. Validate and Prepare Input Data
    if 'change' not in data_df.columns or 'lon' not in data_df.columns or 'lat' not in data_df.columns:
        print("FATAL ERROR: The input CSV file is missing required columns: 'change', 'lon', 'lat'.")
        return

    data_df['change'] = pd.to_numeric(data_df['change'], errors='coerce')
    data_df.dropna(subset=['change'], inplace=True)
    data_df = data_df[data_df['change'] <= 0].copy()

    if data_df.empty:
        print("WARNING: No data points with negative or zero 'change' values found. Cannot create map.")
        return
    print(f"Found {len(data_df)} data points with valid reduction potential values.")

    # 3. Perform Spatial Join
    print("Step 2: Performing spatial join to link data to lake polygons...")
    points_gdf = gpd.GeoDataFrame(
        data_df, geometry=gpd.points_from_xy(data_df.lon, data_df.lat)
    )
    if lakes_geo_df.crs:
        points_gdf.set_crs(lakes_geo_df.crs, inplace=True)
    else:
        print("WARNING: Base lakes Shapefile has no CRS. Assuming WGS84 (EPSG:4326).")
        lakes_geo_df.set_crs(epsg=4326, inplace=True)
        points_gdf.set_crs(epsg=4326, inplace=True)

    merged_gdf = gpd.sjoin(lakes_geo_df, points_gdf, how="left", predicate="contains")

    if merged_gdf.index.duplicated().any():
        print("Aggregating results for lakes with multiple data points...")
        original_geometries = lakes_geo_df.loc[merged_gdf.index.unique()].geometry
        original_crs = lakes_geo_df.crs
        merged_gdf = merged_gdf.groupby(merged_gdf.index).mean(numeric_only=True)
        merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry=original_geometries, crs=original_crs)

    if 'change' not in merged_gdf.columns:
        print("FATAL ERROR: 'change' column lost after spatial join. Cannot proceed.")
        return

    # 4. Apply Natural Logarithmic (ln) Transformation
    print("Step 3: Applying natural log (ln) transformation to 'change' data...")
    change_numeric = pd.to_numeric(merged_gdf['change'], errors='coerce')
    merged_gdf['plot_value'] = np.log(change_numeric.abs() + 1)

    # 5. **MODIFIED**: Setup Enhanced Colormap and Normalization
    print("Step 4: Setting up enhanced, multi-color gradient...")
    # New color sequence: Gray -> Pale Yellow -> Pale Green -> Green -> Blue -> Dark Blue
    colors = [
        "#E7E7E7",  # 灰色 (起始/无数据)
        "#FFFFE0",  # 淡黄色
        "#afe1af",  # 原始的淡绿色
        "#4CAF50",  # 一个更鲜明的绿色
        "#6aa4bb",  # 原始的淡蓝色
        "#4885c1",  # 原始的蓝色
        "#08306B"  # 一个更深的蓝色 (结尾)
    ]
    custom_cmap = LinearSegmentedColormap.from_list("reduction_potential_cmap", colors)

    valid_plot_values = merged_gdf['plot_value'].dropna()
    vmin = 0
    vmax = valid_plot_values.quantile(0.99) if not valid_plot_values.empty else 1
    if vmax == 0:
        vmax = 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    merged_gdf['plot_value'].fillna(-9999, inplace=True)

    # 6. Project Geographies and Plot Map
    print("Step 5: Projecting geometries and plotting the map...")
    robinson_proj = "+proj=robin +datum=WGS84"
    lakes_proj = merged_gdf.to_crs(robinson_proj)
    world_proj = world_gdf.to_crs(robinson_proj)
    world_proj['geometry'] = world_proj.geometry.buffer(0)

    name_col = next((col for col in ['NAME', 'name', 'ADMIN'] if col in world_proj.columns), 'NAME')

    world_fill = world_proj[world_proj[name_col] != 'Antarctica']
    world_borders_plot = world_fill[world_fill[name_col] != 'Russia']

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_facecolor('white')

    # Plot layers in order (bottom to top)
    world_fill.plot(ax=ax, color='white', edgecolor='none', zorder=1)

    lakes_no_data = lakes_proj[lakes_proj['plot_value'] == -9999]
    lakes_no_data.plot(color="#E7E7E7", linewidth=0, edgecolor='none', ax=ax, zorder=2)

    lakes_with_data = lakes_proj[lakes_proj['plot_value'] != -9999]
    lakes_with_data.plot(column='plot_value', cmap=custom_cmap, norm=norm,
                         linewidth=0, edgecolor='none', ax=ax, legend=False, zorder=3)

    world_borders_plot.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5, zorder=4)

    ax.set_axis_off()

    map_png_path = os.path.join(OUTPUT_DIR, "global_reduction_potential_map_ln_scale_enhanced_color.png")
    plt.savefig(map_png_path, dpi=1000, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    print(f"--> Successfully saved map to: {map_png_path}")

    # 7. Generate and Save a Separate, High-Quality Legend
    print("Step 6: Generating and saving a separate legend file...")
    fig_legend = plt.figure(figsize=(2.5, 6))
    ax_legend = fig_legend.add_axes([0.2, 0.1, 0.2, 0.8])

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax_dummy = fig_legend.add_axes([0, 0, 0, 0], visible=False)
    im = ax_dummy.imshow(gradient, aspect='auto', cmap=custom_cmap, norm=norm)

    cbar = fig_legend.colorbar(im, cax=ax_legend, orientation='vertical')
    cbar.set_label("Reduction Potential\n$\\ln(|Change| + 1)$", fontsize=12, weight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=10)

    legend_pdf_path = os.path.join(OUTPUT_DIR, "map_legend_reduction_potential_ln_scale_enhanced_color.pdf")
    fig_legend.savefig(legend_pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig_legend)
    print(f"--> Successfully saved editable legend to: {legend_pdf_path}")
    print("\n--- Map generation finished successfully. ---")


# --- Execution Block ---

if __name__ == '__main__':
    ensure_directories_exist()
    create_reduction_potential_map()