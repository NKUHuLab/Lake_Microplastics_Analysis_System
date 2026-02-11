# -*- coding: utf-8 -*-
# code/06_plot_map.py (FINAL - With a Perfect, Well-Labeled Legend)

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pandas as pd
import numpy as np
import warnings
import os
from shapely.geometry import Polygon
import config

warnings.filterwarnings("ignore")
plt.rcParams.update(config.PLT_RC_PARAMS)


def plot_global_abundance_map():
    """
    Plots the final global abundance map with a continuous color scale and a
    complete, well-labeled legend, while ensuring a clean, artifact-free image.
    """
    print("--- Step 6: Starting Final Plot with a Complete, Labeled Legend ---")

    # 1. Load Data
    try:
        gdf_lakes = gpd.read_file(config.PREDICTED_SHP_PATH)
        world_raw = gpd.read_file(config.BORDERS_SHAPEFILE_PATH)
    except Exception as e:
        print(f"Error: Failed to load geographic data. Error: {e}")
        return

    if gdf_lakes.crs is None:
        gdf_lakes.set_crs(epsg=4326, inplace=True)

    # 2. Setup CONTINUOUS Colormap and handle NA values
    colors = ["#466A77", "#5B7352", "#F1AB3D", "#DA3839"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)

    gdf_lakes['plot_value'] = pd.to_numeric(gdf_lakes['prediction'], errors='coerce')

    vmin, vmax = gdf_lakes['plot_value'].dropna().quantile([0.01, 0.99])
    norm = Normalize(vmin=vmin, vmax=vmax)

    gdf_lakes['plot_value'].fillna(-9999, inplace=True)

    # 3. Project Geographies & Fix Geometry
    print("Applying Robinson projection and fixing geometries...")
    robinson_proj = "+proj=robin +datum=WGS84"
    gdf_lakes_proj = gdf_lakes.to_crs(robinson_proj)
    world_proj = world_raw.to_crs(robinson_proj)

    world_proj['geometry'] = world_proj.geometry.buffer(0)

    name_col = next((col for col in ['NAME', 'name', 'ADMIN', 'SOVEREIGNT'] if col in world_proj.columns), 'NAME')
    world_filtered = world_proj[world_proj[name_col] != 'Antarctica']
    world_for_borders = world_filtered[~world_filtered[name_col].str.contains("russia", case=False, na=False)]

    # 4. Plot the Map Layers in the Correct Order
    print("Plotting map layers...")
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_facecolor('white')

    world_filtered.plot(ax=ax, color='white', edgecolor='none', zorder=1)
    gdf_lakes_na = gdf_lakes_proj[gdf_lakes_proj['plot_value'] == -9999]
    gdf_lakes_na.plot(color="#E7E7E7", linewidth=0, edgecolor='none', ax=ax, zorder=2)
    gdf_lakes_with_data = gdf_lakes_proj[gdf_lakes_proj['plot_value'] != -9999]
    gdf_lakes_with_data.plot(column='plot_value', cmap=custom_cmap, norm=norm,
                             linewidth=0, edgecolor='none', ax=ax, legend=False, zorder=3)
    world_for_borders.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5, zorder=4)
    ax.set_axis_off()

    # 5. Save Files
    map_png_path = os.path.join(config.GLOBAL_MAP_DIR, "global_abundance_map_pure.png")
    plt.savefig(map_png_path, dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Pure global map saved to: {map_png_path}")

    # 6. Generate a PERFECT Legend [MODIFIED FOR GUARANTEED TICKS]
    print("Generating a complete, well-labeled legend...")

    # Create a new figure specifically for the legend
    fig_legend = plt.figure(figsize=(2.5, 7))

    # Add an axis for the colorbar, defining its position and size
    # [left, bottom, width, height] in fractions of figure width
    ax_legend = fig_legend.add_axes([0.05, 0.05, 0.2, 0.9])

    # Create a dummy image that uses our colormap and normalization.
    # This is the key to forcing matplotlib to draw the ticks.
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax_dummy = fig_legend.add_axes([0, 0, 0, 0], visible=False)  # A hidden axis
    im = ax_dummy.imshow(gradient, aspect='auto', cmap=custom_cmap, norm=norm)

    # Create the colorbar from the dummy image
    cbar = fig_legend.colorbar(im, cax=ax_legend, orientation='vertical')

    # Set the label for the colorbar, making it bold
    cbar.set_label(f"Predicted {config.TARGET_VARIABLE}", fontsize=14, weight='bold', labelpad=15)

    # Set font properties for the tick labels
    cbar.ax.tick_params(labelsize=12)

    # Save the final legend as a PDF
    legend_pdf_path = os.path.join(config.GLOBAL_MAP_DIR, "map_legend.pdf")
    fig_legend.savefig(legend_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig_legend)
    print(f"Editable and fully labeled legend saved to: {legend_pdf_path}")
    print("--- Map plotting finished. ---\n")


if __name__ == '__main__':
    config.ensure_directories_exist()
    plot_global_abundance_map()