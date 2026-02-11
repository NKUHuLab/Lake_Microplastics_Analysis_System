# -*- coding: utf-8 -*-
# code/04_pdp_analysis.py (Final Enhanced Version)

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
from mpl_toolkits.mplot3d import Axes3D
import warnings
from itertools import combinations
import config

# Suppress warnings and apply global plot styles
warnings.filterwarnings("ignore")
plt.rcParams.update(config.PLT_RC_PARAMS)


def get_all_feature_pairs(features):
    """Generates all unique pairs of features for 2D PDP analysis."""
    return list(combinations(features, 2))


def plot_3d_pdp_surface(model, X_cluster_data, feature_pair, output_path, cluster_id, grid_resolution=150):
    """
    Generates and saves a high-resolution, beautifully rendered 3D PDP surface plot
    with advanced lighting and shading effects.
    """
    feature_1, feature_2 = feature_pair
    print(f"  - Generating High-Res 3D PDP for '{feature_1}' vs '{feature_2}'...")

    # --- 1. Define Custom Colormap from user-provided colors ---
    custom_colors = [
        "#6aa4bb", "#8dc2b5", "#aab381", "#cad675", "#e4cb3a",
        "#ffc000", "#e4931a", "#c96734", "#ae3a4e", "#8c5374"
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_pdp_color", custom_colors)

    # --- 2. Create High-Resolution Grid of Values ---
    # --- 2. Create High-Resolution Grid of Values ---
    f1_min, f1_max = np.percentile(X_cluster_data[feature_1], [2, 98])
    f2_min, f2_max = np.percentile(X_cluster_data[feature_2], [2, 98])
    if f1_min == f1_max:
        f1_min, f1_max = X_cluster_data[feature_1].min(), X_cluster_data[feature_1].max()
    if f2_min == f2_max:
        f2_min, f2_max = X_cluster_data[feature_2].min(), X_cluster_data[feature_2].max()

    f1_vals = np.linspace(f1_min, f1_max, num=grid_resolution)
    f2_vals = np.linspace(f2_min, f2_max, num=grid_resolution)
    xx, yy = np.meshgrid(f1_vals, f2_vals)

    # --- 3. Prepare Data for Prediction ---
    grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=feature_pair)
    other_feature_means = X_cluster_data.drop(columns=list(feature_pair)).mean()
    full_grid_df = pd.concat(
        [grid_df] + [pd.DataFrame([other_feature_means] * len(grid_df), columns=other_feature_means.index)], axis=1)
    full_grid_df = full_grid_df[model.feature_names_in_]

    # --- 4. Get Predictions and Reshape ---
    Z = model.predict(full_grid_df).reshape(grid_resolution, grid_resolution)

    # --- 5. Advanced Plotting & Beautification ---
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # --- 5a. Create Light Source for Shading ---
    # Create a light source object, coming from the northwest (-135 degrees azimuth).
    ls = LightSource(azdeg=315, altdeg=45)
    # Shade the surface data using the light source and colormap.
    rgb = ls.shade(Z, cmap=custom_cmap, vert_exag=0.1, blend_mode='soft')

    # --- 5b. Plot the Main Surface with Lighting ---
    # The `facecolors` argument applies the pre-rendered RGB colors.
    surface = ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, facecolors=rgb,
                              linewidth=0, antialiased=True, alpha=0.95)

    # --- 5c. Add a Contour Plot on the "Floor" for Depth ---
    ax.contourf(xx, yy, Z, zdir='z', offset=ax.get_zlim()[0], cmap=custom_cmap, alpha=0.6)

    # --- 5d. Aesthetic Refinements ---
    # Set a professional viewing angle.
    ax.view_init(elev=30, azim=-135)

    # Set background panes to pure white for a clean look.
    # **FIX**: Using 'xaxis' instead of 'w_xaxis' for modern matplotlib versions.
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Use a subtle grid for reference.
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, color='grey', alpha=0.5)

    # --- 5e. Labels, Title, and Colorbar ---
    # Use `labelpad` to prevent labels from overlapping the ticks.
    ax.set_xlabel(f'\n{feature_1}', fontsize=14, linespacing=2.5, labelpad=15)
    ax.set_ylabel(f'\n{feature_2}', fontsize=14, linespacing=2.5, labelpad=15)
    ax.set_zlabel(f'Predicted {config.TARGET_VARIABLE}', fontsize=14, labelpad=15)
    ax.set_title(f'High-Resolution 3D PDP for Cluster {cluster_id}\n({feature_1} & {feature_2})', fontsize=18, pad=25)

    # Add a colorbar that accurately represents the un-shaded data values.
    # Create a mappable object for the colorbar.
    m = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=Z.min(), vmax=Z.max()))
    m.set_array([])
    fig.colorbar(m, shrink=0.6, aspect=15, ax=ax, label=f'Predicted {config.TARGET_VARIABLE} Value', pad=0.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, format='pdf', dpi=300)
    plt.close(fig)


def pdp_analysis_by_cluster():
    """
    Generates high-resolution 3D PDPs for each feature-based cluster
    for every possible pair of features.
    """
    print("--- Step 4: Starting Per-Cluster High-Resolution 3D PDP Analysis ---")

    try:
        model = joblib.load(config.MODEL_PATH)
        clustered_data = pd.read_csv(config.SHAP_CLUSTER_DATA_PATH)
    except Exception as e:
        print(f"Error: Could not load required files. Error: {e}")
        return

    feature_pairs = get_all_feature_pairs(config.MODEL_FEATURES)
    print(f"Total feature pairs to analyze: {len(feature_pairs)}")

    unique_clusters = sorted(clustered_data['feature_cluster'].dropna().unique())
    print(f"Found {len(unique_clusters)} valid clusters. Generating PDPs for each...")

    for cluster_id in unique_clusters:
        print(f"\nProcessing Cluster {int(cluster_id)}...")
        cluster_pdp_dir = os.path.join(config.PDP_DIR, f"Cluster_{int(cluster_id)}")
        os.makedirs(cluster_pdp_dir, exist_ok=True)

        cluster_subset = clustered_data[clustered_data['feature_cluster'] == cluster_id]
        if len(cluster_subset) < 20:
            print(f"  Skipping Cluster {int(cluster_id)} due to insufficient data ({len(cluster_subset)} samples).")
            continue

        X_cluster = cluster_subset[model.feature_names_in_]

        for pair in feature_pairs:
            filename = f"pdp_3D_HiRes_cluster_{int(cluster_id)}_{pair[0]}_vs_{pair[1]}.pdf"
            output_path = os.path.join(cluster_pdp_dir, filename)
            plot_3d_pdp_surface(model, X_cluster, pair, output_path, int(cluster_id))

    print("\n--- Per-Cluster High-Resolution 3D PDP analysis finished. ---")


if __name__ == '__main__':
    config.ensure_directories_exist()
    pdp_analysis_by_cluster()