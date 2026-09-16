# -*- coding: utf-8 -*-
# code/10_uncertainty_analysis.py

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm
import config

# Apply global plot settings
plt.rcParams.update(config.PLT_RC_PARAMS)


def analyze_uncertainty():
    """
    Calculates descriptive dispersion among trees on the log-response scale.
    This is not a sampling confidence interval or a per-lake significance test.
    """
    print("--- Step 10: Starting Uncertainty Analysis ---")

    # 1. Load Model, Feature Data, and the BASE Geography Shapefile
    try:
        model = joblib.load(config.MODEL_PATH)
        features_df = pd.read_csv(config.PREDICT_DATA_PATH)
        # Load the original, complete set of lake polygons
        base_geo_df = gpd.read_file(config.BASE_GEOGRAPHY_SHP_PATH)
    except Exception as e:
        print(f"Error: Failed to load a required file. Error: {e}")
        return

    # Ensure the base map has a CRS, assuming WGS84 if missing.
    if base_geo_df.crs is None:
        print("Warning: Base Shapefile has no CRS. Assuming WGS84 (EPSG:4326).")
        base_geo_df.set_crs(epsg=4326, inplace=True)

    # 2. Prepare feature data for prediction
    try:
        model_feature_order = model.feature_names_in_
        X_predict = features_df[model_feature_order].copy()
        if X_predict.isna().any().any():
            if not hasattr(model, "training_feature_medians_"):
                raise ValueError("Missing training-set imputation statistics; refit with the training script before predicting incomplete inputs.")
            X_predict = X_predict.fillna(model.training_feature_medians_)
    except (KeyError, AttributeError) as e:
        print(f"Error aligning data columns with the model's expected features. Error: {e}")
        return

    # 3. Calculate Uncertainty using the model's estimators (trees)
    print("  Getting predictions from each decision tree in the forest...")
    predictions_per_tree = [tree.predict(X_predict) for tree in tqdm(model.estimators_, desc="Trees")]
    predictions_matrix = np.array(predictions_per_tree)

    print("  Calculating descriptive tree dispersion...")
    mean_predictions = predictions_matrix.mean(axis=0)
    std_predictions = predictions_matrix.std(axis=0)
    uncertainty = std_predictions / (np.abs(mean_predictions) + 1e-9)


    # 4. Create a DataFrame with the results
    results_df = features_df[['lon', 'lat']].copy()
    results_df['Uncertainty'] = uncertainty
    results_df.to_csv(config.UNCERTAINTY_CSV_PATH, index=False)
    print(f"Uncertainty results saved to CSV: {config.UNCERTAINTY_CSV_PATH}")

    # 5. Spatially join the results back to the BASE lake map
    print("  Spatially joining uncertainty results to the base global map...")
    points_gdf = gpd.GeoDataFrame(
        results_df, geometry=gpd.points_from_xy(results_df.lon, results_df.lat), crs=base_geo_df.crs
    )

    # Use a left join to keep ALL base polygons
    gdf_uncertainty = gpd.sjoin(base_geo_df, points_gdf, how="left", predicate="contains")

    # If multiple points are in one polygon, aggregate them
    if gdf_uncertainty.index.duplicated().any():
        print("  Aggregating results for lakes with multiple points...")
        # Get the original geometry, which is lost during groupby
        geometries = base_geo_df.loc[gdf_uncertainty.index.unique()].geometry
        # Aggregate descriptive tree-dispersion ratios only
        agg_cols = ['Uncertainty']
        # Group by the original polygon index and average the results
        aggregated_data = gdf_uncertainty.groupby(gdf_uncertainty.index)[agg_cols].mean()
        # Re-create the GeoDataFrame
        gdf_uncertainty = gpd.GeoDataFrame(aggregated_data, geometry=geometries)

    # 6. Plot maps
    plot_uncertainty_map(gdf_uncertainty, 'Uncertainty')

    print("--- Uncertainty analysis finished. ---\n")


def plot_uncertainty_map(gdf, column_name):
    """
    Plots a global map for the specified column, showing "No Data" for un-analyzed areas.
    """
    print(f"  Plotting global map for: {column_name}...")

    # Project data to Robinson
    robinson_proj = "+proj=robin +datum=WGS84"
    gdf_proj = gdf.to_crs(robinson_proj)
    world = gpd.read_file(config.BORDERS_SHAPEFILE_PATH).to_crs(robinson_proj)

    gdf_proj['plot_value'] = pd.to_numeric(gdf_proj[column_name], errors='coerce')

    cmap = 'magma'
    vmin, vmax = np.nanquantile(gdf_proj['plot_value'], [0.01, 0.99]) if not gdf_proj['plot_value'].dropna().empty else (0, 1)

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_facecolor('lightgray')
    world.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5)

    # **New Logic**: Plot "No Data" areas first in a neutral color
    gdf_proj[gdf_proj['plot_value'].isna()].plot(color='#d3d3d3', ax=ax, linewidth=0)

    # Plot areas with data on top
    gdf_proj.dropna(subset=['plot_value']).plot(
        column='plot_value', cmap=cmap, linewidth=0, ax=ax,
        legend=True, legend_kwds={'label': 'Tree dispersion / absolute mean log prediction', 'shrink': 0.5},
        vmin=vmin, vmax=vmax
    )

    ax.set_axis_off()
    ax.set_title('Global Map of Descriptive Tree Dispersion')

    # Save outputs
    for fmt in ['pdf', 'png']:
        output_path = os.path.join(config.UNCERTAINTY_DIR, f"global_{column_name}_map.{fmt}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Map for {column_name} has been saved.")
    plt.close(fig)


if __name__ == '__main__':
    analyze_uncertainty()