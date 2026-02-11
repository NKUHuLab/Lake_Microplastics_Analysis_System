# -*- coding: utf-8 -*-
# code/03_shap_analysis.py (Modified to Keep ALL Clusters)

import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from tqdm import tqdm
import config

warnings.filterwarnings("ignore")
plt.rcParams.update(config.PLT_RC_PARAMS)


def feature_based_clustering_and_shap():
    """
    Performs K-Means clustering, saves SHAP values, and prepares
    data for external visualization, using paths from the config file.
    This version RETAINS ALL CLUSTERS regardless of their size.
    """
    print("--- Step 3: Starting SHAP Analysis and Feature-Based Clustering ---")

    # 1. Load Model and Data
    try:
        model = joblib.load(config.MODEL_PATH)
        data = pd.read_csv(config.PREDICT_DATA_PATH)
    except Exception as e:
        print(f"Error loading required files: {e}")
        return

    X = data[config.MODEL_FEATURES].copy().fillna(data[config.MODEL_FEATURES].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Automatically Determine Optimal Number of Clusters
    print("Determining optimal number of clusters using the Elbow method...")
    sse = []
    k_range = range(2, config.KMEANS_MAX_CLUSTERS + 1)
    for k in tqdm(k_range, desc="Testing k values"):
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init='auto')
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(k_range, sse, curve='convex', direction='decreasing')
    n_clusters = kl.elbow or 4  # If KneeLocator fails, it defaults to 4
    print(f"✅ Optimal number of clusters found: k={n_clusters}")

    # --- MODIFICATION START ---
    # The section that filtered small clusters has been removed.
    print("Performing K-Means clustering and keeping ALL clusters.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init='auto')

    # We now directly assign the cluster labels to the final column.
    data['feature_cluster'] = kmeans.fit_predict(X_scaled)

    print("Cluster sizes:")
    print(data['feature_cluster'].value_counts())
    print("✅ All generated clusters will be used in the analysis.")
    # --- MODIFICATION END ---

    # Save the elbow plot for review
    elbow_path = os.path.join(config.SHAP_DIR, "feature_clustering_elbow_method.pdf")
    kl.plot_knee()
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Errors)")
    plt.title(f"Elbow Method for Optimal k (Found k={n_clusters})")
    plt.savefig(elbow_path, format='pdf', bbox_inches='tight')
    plt.close()

    # Save data with the complete cluster labels
    data.to_csv(config.SHAP_CLUSTER_DATA_PATH, index=False)
    print(f"Data with ALL cluster labels saved to: {config.SHAP_CLUSTER_DATA_PATH}")

    # 4. Calculate and Save SHAP values
    print("Calculating SHAP values for the full dataset...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df['cluster'] = data['feature_cluster']

    config.SHAP_VALUES_PATH = os.path.join(config.SHAP_DIR, 'shap_values_with_clusters.csv')
    shap_df.to_csv(config.SHAP_VALUES_PATH, index=False)
    print(f"SHAP values and cluster labels saved to: {config.SHAP_VALUES_PATH}")

    print("\n--- SHAP analysis and clustering finished. ---")
    print("Next steps:")
    print("1. Run 'shap_聚类主图绘制.py' to generate the innovative heatmap.")
    print("2. Run '08_shap_clustering_and_map.py' to see all clusters on the map.")


if __name__ == '__main__':
    config.ensure_directories_exist()
    feature_based_clustering_and_shap()