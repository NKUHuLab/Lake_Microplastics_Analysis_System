# -*- coding: utf-8 -*-
# code/11_geographic_clustering.py

import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import MultiPoint
from code import config
import warnings
import os

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')
plt.rcParams.update(config.PLT_RC_PARAMS)


def geo_clustering_analysis():
    """
    Performs a focused geospatial clustering and creates a final, beautified,
    and performance-optimized map that is easy to open and edit.
    """
    print("--- Step 11: Starting Final Geographic Clustering Analysis ---")

    # 1. Load Full Dataset
    try:
        df_full = pd.read_csv(config.PREDICTION_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Prediction CSV file not found at '{config.PREDICTION_CSV_PATH}'.")
        return

    df_full.rename(columns={'prediction': 'ln_abundance'}, inplace=True)
    df_full.dropna(subset=['lon', 'lat', 'ln_abundance'], inplace=True)

    # 2. Filter data: Keep only top 5% for clustering
    high_pollution_threshold = df_full['ln_abundance'].quantile(0.95)
    print(f"Isolating high-pollution lakes with ln_abundance >= {high_pollution_threshold:.2f} (Top 5%).")

    df_low_pollution = df_full[df_full['ln_abundance'] < high_pollution_threshold].copy()
    df_for_clustering = df_full[df_full['ln_abundance'] >= high_pollution_threshold].copy()

    if len(df_for_clustering) < 100:
        print("Error: Not enough high-pollution data points to perform meaningful clustering.")
        return

    # 3. Find 10,000 representative centers from the high-pollution subset
    n_centers = 10000
    if len(df_for_clustering) < n_centers:
        print(
            f"Warning: Number of high-pollution lakes ({len(df_for_clustering)}) is less than requested centers ({n_centers}). Using all high-pollution lakes as centers.")
        df_centers = df_for_clustering[['lon', 'lat', 'ln_abundance']].copy()
    else:
        print(f"Finding {n_centers} representative centers from high-pollution lakes using K-Means...")
        features_for_centers = df_for_clustering[['lon', 'lat', 'ln_abundance']].values
        scaler_centers = StandardScaler()
        features_scaled_centers = scaler_centers.fit_transform(features_for_centers)

        kmeans_centers = KMeans(n_clusters=n_centers, random_state=config.RANDOM_STATE, n_init='auto')
        kmeans_centers.fit(features_scaled_centers)

        representative_centers = scaler_centers.inverse_transform(kmeans_centers.cluster_centers_)
        df_centers = pd.DataFrame(representative_centers, columns=['lon', 'lat', 'ln_abundance'])

    # 4. Perform Final Clustering on the Representative Centers
    n_final_clusters = 20
    print(f"Performing final K-Means clustering on {len(df_centers)} centers to find {n_final_clusters} clusters...")

    scaler_final = StandardScaler()
    features_scaled_final = scaler_final.fit_transform(df_centers)
    features_scaled_final[:, 0] *= 4.0  # Prioritize geography
    features_scaled_final[:, 1] *= 4.0
    features_scaled_final[:, 2] *= 1.0

    kmeans_final = KMeans(n_clusters=n_final_clusters, random_state=config.RANDOM_STATE, n_init='auto')
    df_centers['cluster'] = kmeans_final.fit_predict(features_scaled_final)
    print(f"  Clustering complete. Found {n_final_clusters} distinct clusters from the representative centers.")

    # 5. Calculate and Save Cluster Statistics
    print("\n--- High-Pollution Cluster Statistics ---")
    stats_df = df_centers.groupby('cluster')['ln_abundance'].agg(['count', 'mean']).reset_index()
    stats_df.rename(columns={'count': 'Num Rep. Points', 'mean': 'Mean ln', 'cluster': 'Cluster ID'}, inplace=True)
    stats_df = stats_df.sort_values(by='Mean ln', ascending=False)

    print(stats_df)
    stats_df.to_csv(os.path.join(config.GEO_CLUSTERING_DIR, 'cluster_summary.csv'), index=False)
    print("\nCluster statistics saved to 'cluster_summary.csv'")

    # 6. Identify High-Risk Categories for Visualization
    num_risk_zones = max(1, n_final_clusters // 2)
    high_risk_clusters = stats_df.nlargest(num_risk_zones, 'Mean ln')['Cluster ID'].tolist()
    print(f"  Will draw risk zones for the top {len(high_risk_clusters)} most polluted clusters: {high_risk_clusters}")

    hotspot_threshold_995 = df_full['ln_abundance'].quantile(0.995)
    df_centers['is_hotspot'] = df_centers['ln_abundance'] >= hotspot_threshold_995
    print(f"  Individual hotspots defined as Top 0.5% (ln >= {hotspot_threshold_995:.2f}).")

    # 7. Enhanced Visualization
    print("  Plotting final global clustering map (with rasterization for performance)...")
    robinson_proj = "+proj=robin +datum=WGS84"
    world = gpd.read_file(config.BORDERS_SHAPEFILE_PATH).to_crs(robinson_proj)
    world_no_russia_border = world[world['NAME'] != 'Russia']

    fig, ax = plt.subplots(1, 1, figsize=(22, 12))
    ax.set_facecolor('#aadaff')

    # --- Plotting Layers ---
    # Layer 1: Vector Basemap (will remain editable)
    world.plot(ax=ax, color='#e0e0e0', edgecolor='none', zorder=1)
    world_no_russia_border.plot(ax=ax, color='none', edgecolor='white', linewidth=0.5, zorder=2)

    # Convert data to GeoDataFrames
    gdf_low_pollution = gpd.GeoDataFrame(df_low_pollution,
                                         geometry=gpd.points_from_xy(df_low_pollution.lon, df_low_pollution.lat),
                                         crs="EPSG:4326").to_crs(robinson_proj)
    gdf_centers = gpd.GeoDataFrame(df_centers, geometry=gpd.points_from_xy(df_centers.lon, df_centers.lat),
                                   crs="EPSG:4326").to_crs(robinson_proj)

    custom_colors = plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors
    color_map = {cid: custom_colors[i % len(custom_colors)] for i, cid in enumerate(stats_df['Cluster ID'])}

    # Layer 2: RASTERIZED Scatter Points (for small file size)
    # The `rasterized=True` flag converts these points into a single image inside the PDF
    ax.scatter(gdf_low_pollution.geometry.x, gdf_low_pollution.geometry.y, s=5, color='silver', alpha=0.3, zorder=3,
               rasterized=True)
    for cid, color in color_map.items():
        gdf_centers[gdf_centers['cluster'] == cid].plot(ax=ax, color=color, markersize=35, alpha=0.6, edgecolor='none',
                                                        zorder=4, rasterized=True)

    # Layer 3: VECTOR Risk Zones and Hotspots (will remain editable on top of the image)
    for cid in high_risk_clusters:
        core_hotspots = gdf_centers[(gdf_centers['cluster'] == cid) & (gdf_centers['is_hotspot'])]
        if len(core_hotspots) > 2:
            hull = MultiPoint(core_hotspots.geometry.to_list()).convex_hull
            gpd.GeoSeries([hull], crs=gdf_centers.crs).plot(ax=ax, facecolor='red', alpha=0.15, edgecolor='red',
                                                            linewidth=2, zorder=5)

    gdf_centers[gdf_centers['is_hotspot']].plot(ax=ax, marker='o', markersize=50, facecolors='none', edgecolors='gold',
                                                linewidth=1.2, zorder=6)

    # Final Legend Creation (also a vector object)
    legend_elements = [
        Patch(facecolor='silver', label='Low Pollution Lakes (Bottom 95%)'),
        Line2D([0], [0], marker='o', color='w', label=f'Severe Hotspot (Top 0.5%)', markerfacecolor='none',
               markeredgecolor='gold', markersize=12, linewidth=1.2),
        Patch(facecolor='red', alpha=0.3, label='Severe Pollution Zone (Core)')
    ]
    for _, row in stats_df.iterrows():
        cid = int(row['Cluster ID'])
        label = f"Cluster {cid} (Points: {row['Num Rep. Points']})"
        patch = Patch(facecolor=color_map.get(cid, 'black'), alpha=0.7, label=label)
        legend_elements.append(patch)

    leg = ax.legend(handles=legend_elements, loc='lower left', fontsize=9, title="Legend", frameon=True, ncol=3)

    # Color the legend text for the high-risk clusters
    for text in leg.get_texts():
        if "Cluster" in text.get_text():
            try:
                cluster_id_from_label = int(text.get_text().split(" ")[1])
                if cluster_id_from_label in high_risk_clusters:
                    text.set_color('red')
            except (ValueError, IndexError):
                continue

    ax.set_axis_off()
    ax.set_title('Global Geographic Clustering of High-Pollution Lakes', fontsize=20, pad=10)

    # Save outputs
    for fmt in ['pdf', 'png']:
        # The dpi here controls the resolution of the rasterized point layers in the PDF
        plt.savefig(os.path.join(config.GEO_CLUSTERING_DIR, f"global_geo_clustering_map.{fmt}"), dpi=300,
                    bbox_inches='tight')
    print(f"  Geographic clustering map has been saved.")
    plt.close(fig)
    print("--- Geographic clustering analysis finished. ---\n")


if __name__ == '__main__':
    geo_clustering_analysis()