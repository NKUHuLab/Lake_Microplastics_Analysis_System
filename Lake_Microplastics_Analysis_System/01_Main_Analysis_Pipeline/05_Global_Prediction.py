# -*- coding: utf-8 -*-
# code/05_predict_global.py

import pandas as pd
import geopandas as gpd
import numpy as np
import joblib
import warnings
import os
import config

warnings.filterwarnings("ignore")


def predict_and_create_shp():
    """
    Loads model and data, runs predictions on all features, saves the complete
    tabular results, and then creates a spatially-joined Shapefile.
    """
    print("--- Step 5: Starting Global Prediction and Data Export ---")

    # 1. Load Model and Data
    try:
        model = joblib.load(config.MODEL_PATH)
        features_df = pd.read_csv(config.PREDICT_DATA_PATH)
        geo_df = gpd.read_file(config.BASE_GEOGRAPHY_SHP_PATH)
    except Exception as e:
        print(f"Error: Failed to load a required file. Error: {e}")
        return

    # 2. Data Cleaning and Validation
    features_df.columns = features_df.columns.str.strip()
    if 'lon' not in features_df.columns or 'lat' not in features_df.columns:
        print("Error: The feature file is missing 'lon' or 'lat' columns.")
        return

    # 3. Data Preparation and Alignment for Prediction
    try:
        model_feature_order = model.feature_names_in_
        X_predict = features_df[model_feature_order].copy()
    except (KeyError, AttributeError) as e:
        print(f"Error aligning feature columns. Error: {e}")
        return
    X_predict.fillna(X_predict.median(), inplace=True)

    # 4. Make Predictions on the ENTIRE feature set
    print(f"Making predictions on all {len(features_df)} data points...")
    features_df['prediction'] = model.predict(X_predict)

    # 5. **New**: Save Complete Tabular Data (CSV and Excel) FIRST
    print("Saving complete tabular data (CSV and Excel)...")
    try:
        # The `features_df` now contains all original data plus the 'prediction' column
        features_df.to_csv(config.PREDICTION_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"  -> Successfully saved {config.PREDICTION_CSV_PATH}")

        features_df.to_excel(config.PREDICTION_XLSX_PATH, index=False)
        print(f"  -> Successfully saved {config.PREDICTION_XLSX_PATH}")

    except Exception as e:
        print(f"Error saving tabular data files: {e}")

    # 6. Perform Spatial Join to Create the Geographic File
    print("\nStarting spatial join to create Shapefile...")
    # Drop any points that don't have valid coordinates before creating geometries
    features_df.dropna(subset=['lon', 'lat'], inplace=True)
    points_gdf = gpd.GeoDataFrame(
        features_df, geometry=gpd.points_from_xy(features_df.lon, features_df.lat)
    )
    if geo_df.crs:
        points_gdf.set_crs(geo_df.crs, inplace=True)

    # Use an inner join, so only points that fall within a lake polygon are kept
    merged_gdf = gpd.sjoin(geo_df, points_gdf, how="left", predicate="contains")

    # Aggregate results for lakes with multiple points
    if merged_gdf.index.duplicated().any():
        print("Aggregating results for lakes with multiple points...")
        original_geometries = geo_df.loc[merged_gdf.index.unique()].geometry
        original_crs = geo_df.crs
        merged_gdf = merged_gdf.groupby(merged_gdf.index).mean(numeric_only=True)
        merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry=original_geometries, crs=original_crs)

    if merged_gdf.empty:
        print("Warning: Spatial join resulted in an empty GeoDataFrame. No output Shapefile will be created.")
        return

    print(f"Spatial join successful. {len(merged_gdf)} lakes matched to be saved in Shapefile.")

    # 7. Save Geographic Data (Shapefile)
    print(f"Saving new Shapefile to: {config.PREDICTED_SHP_PATH}")
    try:
        merged_gdf.to_file(config.PREDICTED_SHP_PATH, driver='ESRI Shapefile', encoding='utf-8')
    except Exception as e:
        print(f"Error saving the final Shapefile: {e}")
        return

    print("--- Global prediction and data export finished. ---\n")


if __name__ == '__main__':
    predict_and_create_shp()