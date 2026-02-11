# -*- coding: utf-8 -*-
# code/08_shap_clustering_and_map.py (FINAL - Mirrored from 06_plot_map.py logic)

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, Polygon
from collections import Counter
import matplotlib.patches as mpatches
import os
import config
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update(config.PLT_RC_PARAMS)


def map_feature_clusters():
    """
    Generates the final cluster map with a clean PNG output and a separate,
    beautifully formatted PDF legend, mirroring the robust logic from the
    abundance map script.
    """
    print("--- Step 8: Starting Final Plot of Cluster Map with Perfect Legend ---")

    # 1. Load Data
    try:
        df_clusters = pd.read_csv(config.SHAP_CLUSTER_DATA_PATH)
        gdf_lakes = gpd.read_file(config.PREDICTED_SHP_PATH)
        world = gpd.read_file(config.BORDERS_SHAPEFILE_PATH)
    except Exception as e:
        print(f"Error: Failed to load a required file. Error: {e}")
        return

    # 2. Prepare Data
    if gdf_lakes.crs is None: gdf_lakes.set_crs(epsg=4326, inplace=True)
    df_clusters_valid = df_clusters.dropna(subset=['lon', 'lat', 'feature_cluster'])
    geometry = [Point(lon, lat) for lon, lat in zip(df_clusters_valid['lon'], df_clusters_valid['lat'])]
    gdf_points = gpd.GeoDataFrame(df_clusters_valid, geometry=geometry, crs=gdf_lakes.crs)

    # 3. Spatial Join and Cluster Assignment
    gdf_joined = gpd.sjoin(gdf_lakes, gdf_points, how="inner", predicate="contains")
    if gdf_joined.empty:
        print("Error: Spatial join failed.")
        return

    def get_dominant_cluster(series):
        return Counter(series.dropna()).most_common(1)[0][0] if not series.empty else None

    dominant_clusters = gdf_joined.groupby(level=0)['feature_cluster'].apply(get_dominant_cluster)
    gdf_lakes['dominant_cluster'] = dominant_clusters
    gdf_lakes['dominant_cluster'].fillna(-9999, inplace=True)

    gdf_water_clustered = gdf_lakes[gdf_lakes['dominant_cluster'] != -9999]
    gdf_water_clustered['dominant_cluster'] = gdf_water_clustered['dominant_cluster'].astype(int)

    # 4. Project and Fix Geometry
    robinson_proj_str = "+proj=robin +datum=WGS84"
    gdf_lakes_proj = gdf_lakes.to_crs(robinson_proj_str)
    world_proj = world.to_crs(robinson_proj_str)
    world_proj['geometry'] = world_proj.geometry.buffer(0)

    name_col = next((col for col in ['NAME', 'name', 'ADMIN', 'SOVEREIGNT'] if col in world_proj.columns), 'NAME')
    world_filtered = world_proj[world_proj[name_col] != 'Antarctica']
    world_for_borders = world_filtered[~world_filtered[name_col].str.contains("russia", case=False, na=False)]

    # 5. Setup Colors
    unique_clusters = sorted(gdf_water_clustered['dominant_cluster'].unique())
    provided_colors = ["#4885c1", "#c96734", "#6aa4bb", "#e4cb3a", "#8c5374", "#aab381", "#ffc000", "#ae3a4e",
                       "#afe1af", "#cad675", "#8dc2b5", "#6a6c9b"]
    cluster_color_map = {cluster: provided_colors[i % len(provided_colors)] for i, cluster in
                         enumerate(unique_clusters)}
    gdf_lakes_proj['plot_color'] = gdf_lakes_proj['dominant_cluster'].map(cluster_color_map)

    # 6. Plot the PURE PNG Map
    print("Plotting pure PNG map...")
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_facecolor('white')

    # Layer 1: Base landmass (countries) in white
    world_filtered.plot(ax=ax, color='white', edgecolor='none', zorder=1)

    # Layer 2: Lakes WITHOUT data (NA values), colored in light grey
    gdf_lakes_na = gdf_lakes_proj[gdf_lakes_proj['dominant_cluster'] == -9999]
    gdf_lakes_na.plot(color="#E7E7E7", linewidth=0, edgecolor='none', ax=ax, zorder=2)

    # Layer 3: Lakes WITH data, colored by cluster
    gdf_lakes_with_data = gdf_lakes_proj[gdf_lakes_proj['dominant_cluster'] != -9999]
    gdf_lakes_with_data.plot(color=gdf_lakes_with_data['plot_color'], ax=ax, linewidth=0, edgecolor='none', zorder=3)

    # Optional: If you want borders on the PNG, uncomment the next line
    # world_for_borders.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.5, zorder=4)

    ax.set_axis_off()

    # Save the pure PNG file
    map_png_path = os.path.join(config.SHAP_CLUSTER_MAP_DIR, "shap_cluster_map_pure.png")
    plt.savefig(map_png_path, dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Pure global map saved to: {map_png_path}")

    # 7. Generate a PERFECT Legend PDF
    print("Generating a complete, well-labeled legend...")

    # Create the legend handles (color patches and labels)
    handles = [mpatches.Patch(color=color, label=f'Cluster {int(cluster)}') for cluster, color in
               sorted(cluster_color_map.items())]

    # Create a new figure just for the legend
    # Adjust figsize width based on the number of legend items
    fig_legend, ax_legend = plt.subplots(figsize=(max(4, len(handles) * 1.8), 2))

    # Turn off the axis for a clean look
    ax_legend.set_axis_off()

    # Create the legend object itself
    fig_legend.legend(
        handles=handles,
        loc='center',  # Center the legend in the figure
        frameon=False,  # No frame around the legend
        ncol=len(handles),  # Arrange all items in a single row
        handletextpad=0.5,  # Space between patch and text
        columnspacing=1.5,  # Space between columns
        prop={'size': 12}  # Font size
    )

    # Save the final legend as a PDF
    legend_pdf_path = os.path.join(config.SHAP_CLUSTER_MAP_DIR, 'cluster_map_legend.pdf')
    # Use tight_layout or bbox_inches='tight' to ensure it's not clipped
    plt.savefig(legend_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig_legend)
    print(f"Editable and fully labeled legend saved to: {legend_pdf_path}")
    print("--- Feature cluster map plotting finished. ---\n")


if __name__ == '__main__':
    config.ensure_directories_exist()
    map_feature_clusters()