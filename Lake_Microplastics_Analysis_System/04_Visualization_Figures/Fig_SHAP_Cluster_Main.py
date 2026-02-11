# -*- coding: utf-8 -*-
# code/shap_聚类主图绘制.py (FINAL - Innovative Sunburst Plot)

import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import warnings
import config

# Suppress warnings
warnings.filterwarnings("ignore")


def plot_shap_sunburst_figure():
    """
    Creates an innovative and beautiful SHAP summary figure using a Sunburst plot.
    This chart visualizes cluster sizes and the most impactful features for each
    cluster in a single, hierarchical, and aesthetically pleasing diagram.
    """
    print("--- Starting Innovative Sunburst SHAP Cluster Visualization ---")

    # 1. Load Data directly from CSV files
    try:
        shap_values_csv_path = os.path.join(config.SHAP_DIR, 'shap_values_with_clusters.csv')
        shap_data = pd.read_csv(shap_values_csv_path)
        print("CSV data loaded successfully.")
    except Exception as e:
        print(f"Error loading required CSV files: {e}")
        return

    # 2. Prepare data for the Sunburst plot
    valid_clusters = sorted(shap_data['cluster'].dropna().unique())
    if not valid_clusters:
        print("No valid clusters found in the data.")
        return
    print(f"Processing {len(valid_clusters)} clusters...")

    plot_data = {
        "ids": ["Root"],
        "labels": ["All Clusters"],
        "parents": [""],
        "values": [len(shap_data)]
    }

    # Use a high-contrast, beautiful color palette
    color_palette = ["#4885c1", "#c96734", "#6aa4bb", "#e4cb3a", "#8c5374", "#aab381"]

    for i, cluster_id in enumerate(valid_clusters):
        cluster_mask = (shap_data['cluster'] == cluster_id)
        cluster_data = shap_data[cluster_mask]

        # --- Add Cluster Level to the Plot Data ---
        cluster_label = f"Cluster {int(cluster_id)}"
        plot_data["ids"].append(cluster_label)
        plot_data["labels"].append(cluster_label)
        plot_data["parents"].append("Root")
        plot_data["values"].append(len(cluster_data))  # Size of sector based on sample count

        # --- Calculate and Add Feature Level ---
        # Calculate mean absolute SHAP values for this cluster
        mean_abs_shap = cluster_data[config.MODEL_FEATURES].abs().mean().nlargest(4)

        for feature, importance in mean_abs_shap.items():
            feature_id = f"{cluster_label}-{feature}"
            plot_data["ids"].append(feature_id)
            plot_data["labels"].append(feature)
            plot_data["parents"].append(cluster_label)
            plot_data["values"].append(importance)  # Size of sector based on importance

    # 3. Create the Sunburst Figure
    print("Generating the Sunburst plot...")
    fig = go.Figure(go.Sunburst(
        ids=plot_data["ids"],
        labels=plot_data["labels"],
        parents=plot_data["parents"],
        values=plot_data["values"],
        branchvalues="total",  # Values of inner sectors are sum of outer sectors
        hoverinfo="label+percent parent+value",
        insidetextorientation='radial',
        marker=dict(
            colors=[color_palette[int(p.split(' ')[-1]) % len(color_palette)] if 'Cluster' in p else '#cccccc' for p in
                    plot_data["parents"]],
            line=dict(color='#ffffff', width=2)
        )
    ))

    # 4. Customize Layout and Style
    fig.update_layout(
        title={
            'text': "<b>Top Feature Impacts Across SHAP-Derived Clusters</b>",
            'y': 0.96,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        margin=dict(t=80, l=0, r=0, b=0),
        sunburstcolorway=color_palette,
    )

    # 5. Save the final figure as a high-quality, editable PDF
    output_path = os.path.join(config.SHAP_DIR, "shap_cluster_sunburst_summary.pdf")
    fig.write_image(output_path, width=1200, height=1200, scale=2)

    print(f"\nInnovative Sunburst visualization saved to: {output_path}")
    print("--- Visualization script finished successfully. ---")


if __name__ == '__main__':
    config.ensure_directories_exist()
    plot_shap_sunburst_figure()