# -*- coding: utf-8 -*-
# code/config.py

import os
from itertools import combinations

# --- Project Root ---
PROJECT_ROOT = r"E:\lake-MP-W"

# --- Core Input File Paths ---
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train", "train_data.csv")
PREDICT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "feature", "feature_2022.csv")
BASE_GEOGRAPHY_SHP_PATH = os.path.join(PROJECT_ROOT, "data", "base_shp", "predicted_lakes_2022.shp")
BORDERS_SHAPEFILE_PATH = os.path.join(PROJECT_ROOT, "data", "base_shp", "world map china line.shp")

# --- Model and Data Definitions ---
TARGET_VARIABLE = 'ln'
MODEL_FEATURES = [
    'Lake_area', 'Shore_dev', 'Vol_total', 'Res_time',
    'Total_POP_SERVED', 'Average_DF', 'Primary_Waste_Discharge',
    'Secondary_Waste_Discharge', 'Advanced_Waste_Discharge',
    'RSE_paved', 'RSE_gravel', 'RSE_other', 'prec', 'emis_tyre_TSP_HEG',
    'emis_brake_TSP_HEG', 'PM2_5', 'PM10', 'Mismanaged', 'Total_Plast',
    'fish_gdp_sqkm', 'Cultivated_land', 'Artificial_surface'
]
FEATURE_GROUPS = {
    'Atmospheric Input': ['prec', 'emis_tyre_TSP_HEG', 'emis_brake_TSP_HEG', 'PM2_5', 'PM10'],
    'Hydrological Input': ['Total_POP_SERVED', 'Average_DF', 'Primary_Waste_Discharge', 'Secondary_Waste_Discharge', 'Advanced_Waste_Discharge'],
    'Watershed Non-Point Pollution': ['Mismanaged', 'Total_Plast', 'fish_gdp_sqkm', 'Cultivated_land', 'Artificial_surface']
}

# --- Output Directories and File Paths ---
DRAW_DIR = os.path.join(PROJECT_ROOT, "draw")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
GENERATED_SHP_DIR = os.path.join(PROJECT_ROOT, "data", "generated_shp")

# Module 00
MODEL_COMPARISON_DIR = os.path.join(DRAW_DIR, "00_Model_Comparison")
MODEL_COMPARISON_EXCEL_PATH = os.path.join(MODEL_COMPARISON_DIR, "model_performance_comparison.xlsx")

# Module 01
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
EVAL_DIR = os.path.join(DRAW_DIR, "01_Model_Evaluation")

# Module 02
FEATURE_IMPORTANCE_DIR = os.path.join(DRAW_DIR, "02_Feature_Importance")
FEATURE_IMPORTANCE_PATH = os.path.join(FEATURE_IMPORTANCE_DIR, "feature_importance.pdf")

# Module 03
SHAP_DIR = os.path.join(DRAW_DIR, "03_SHAP_Analysis")
SHAP_CLUSTER_DATA_PATH = os.path.join(SHAP_DIR, 'data_with_feature_clusters.csv')

# Module 04
PDP_DIR = os.path.join(DRAW_DIR, "04_PDP_Analysis")

# Module 05
PREDICTED_SHP_PATH = os.path.join(GENERATED_SHP_DIR, "predicted_lakes_2022.shp")
# **New Paths** for tabular output
GLOBAL_PREDICTION_DIR = os.path.join(DRAW_DIR, "05_Global_Prediction_Data")
PREDICTION_CSV_PATH = os.path.join(GLOBAL_PREDICTION_DIR, "global_predictions.csv")
PREDICTION_XLSX_PATH = os.path.join(GLOBAL_PREDICTION_DIR, "global_predictions.xlsx")


# Module 06
GLOBAL_MAP_DIR = os.path.join(DRAW_DIR, "06_Global_Map")
MAP_OUTPUT_PATH = os.path.join(GLOBAL_MAP_DIR, "global_abundance_map.png")

# Module 07
FEATURE_GROUP_CONTRIBUTION_DIR = os.path.join(DRAW_DIR, "07_Feature_Group_Contribution")
FEATURE_GROUP_RADAR_PATH = os.path.join(FEATURE_GROUP_CONTRIBUTION_DIR, 'Feature_Contribution_Radar.pdf')

# Module 08
SHAP_CLUSTER_MAP_DIR = os.path.join(DRAW_DIR, "08_SHAP_Cluster_Map")
SHAP_CLUSTER_MAP_PATH = os.path.join(SHAP_CLUSTER_MAP_DIR, 'global_shap_cluster_map.png')

# Module 10
UNCERTAINTY_DIR = os.path.join(DRAW_DIR, "10_Uncertainty_Analysis")
UNCERTAINTY_CSV_PATH = os.path.join(UNCERTAINTY_DIR, "global_prediction_uncertainty.csv")
UNCERTAINTY_MAP_PATH = os.path.join(UNCERTAINTY_DIR, "global_Uncertainty_map.pdf")

# Module 11
GEO_CLUSTERING_DIR = os.path.join(DRAW_DIR, "11_Geographic_Clustering")
GEO_CLUSTERING_MAP_PATH = os.path.join(GEO_CLUSTERING_DIR, "global_geo_clustering_map.pdf")

# Module 12
CAUSAL_DAG_DIR = os.path.join(DRAW_DIR, "12_Causal_DAG")
CAUSAL_DAG_PATH = os.path.join(CAUSAL_DAG_DIR, "causal_dag.pdf")


# --- Analysis and Plotting Parameters ---
TEST_SIZE = 0.2
CV_FOLDS = 10
RANDOM_STATE = 42
KMEANS_MAX_CLUSTERS = 15
SCORING_METRIC = 'r2'
PLT_RC_PARAMS = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42
}

# --- Helper Functions ---
def get_pdp_feature_pairs():
    reps = {
        'Atmospheric Input': 'PM2_5',
        'Hydrological Input': 'Total_POP_SERVED',
        'Watershed Non-Point Pollution': 'Mismanaged'
    }
    hydro_features = ['Total_POP_SERVED', 'Average_DF', 'Primary_Waste_Discharge', 'Secondary_Waste_Discharge']
    pairs = list(combinations(reps.values(), 2)) + list(combinations(hydro_features, 2))
    return list(set(pairs))

def ensure_directories_exist():
    dirs = [
        MODEL_DIR, GENERATED_SHP_DIR, DRAW_DIR, MODEL_COMPARISON_DIR,
        EVAL_DIR, FEATURE_IMPORTANCE_DIR, SHAP_DIR, PDP_DIR,
        GLOBAL_PREDICTION_DIR, # **Added new directory**
        GLOBAL_MAP_DIR,
        FEATURE_GROUP_CONTRIBUTION_DIR, SHAP_CLUSTER_MAP_DIR, UNCERTAINTY_DIR,
        GEO_CLUSTERING_DIR, CAUSAL_DAG_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("--- All output directories have been checked and are ready. ---")