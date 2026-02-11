# 15_compositional_analysis.py
# Final corrected version focusing on correctness and clarity.

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.path import Path
from sklearn.preprocessing import normalize
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Import the setup functions from your data preparation script
from data_preparation import setup_environment, save_plot


def create_segmented_colormap():
    """Creates a high-contrast, segmented colormap."""
    cdict = {
        'red': ((0.0, 0.0, 0.0), (0.49, 0.56, 0.56), (0.5, 0.98, 0.98), (0.75, 0.97, 0.97), (1.0, 0.98, 0.0)),
        'green': ((0.0, 0.47, 0.47), (0.49, 0.74, 0.74), (0.5, 0.78, 0.78), (0.75, 0.60, 0.60), (1.0, 0.25, 0.0)),
        'blue': ((0.0, 0.71, 0.71), (0.49, 0.43, 0.43), (0.5, 0.31, 0.31), (0.75, 0.12, 0.12), (1.0, 0.27, 0.0))
    }
    return LinearSegmentedColormap('custom_segmented', cdict)


def project_point(p, inverted=False):
    """Projects a single ternary point (0-100) to 2D (x,y) coordinates."""
    p_right, p_left, p_top = p[0], p[1], p[2]
    x = p_right + p_top * 0.5
    y = p_top * (np.sqrt(3) / 2.)
    return np.array([x, -y if inverted else y])


def boundary_constrained_jitter(xy_coords, boundary_path, strength=3.0):
    """
    Applies random jitter to points but ensures they remain within the boundary path.
    """
    jittered_coords = np.copy(xy_coords)
    for i in range(len(xy_coords)):
        # Generate a random displacement
        dx = np.random.uniform(-strength, strength)
        dy = np.random.uniform(-strength, strength)
        new_point = xy_coords[i] + np.array([dx, dy])
        # Only accept the move if the new point is inside the triangle
        if boundary_path.contains_point(new_point):
            jittered_coords[i] = new_point

    return jittered_coords


def plot_pro_style_ternary(ax, proportions, residuals, components, custom_cmap, norm, median_benchmark=None,
                           inverted=False):
    """Plots a clean, professional ternary plot with boundary-constrained jitter."""
    scale = 100
    y_lim, y_dir = ((-100, 10), -1) if inverted else ((-10, 100), 1)
    ax.set_xlim(-10, 110)
    ax.set_ylim(y_lim)

    # --- 1. Clean Background Triangle ---
    corners = np.array([project_point(p, inverted) for p in [[100, 0, 0], [0, 100, 0], [0, 0, 100]]])
    triangle = plt.Polygon(corners, facecolor='none', ec='black', lw=2.0, zorder=1)
    ax.add_patch(triangle)
    triangle_path = Path(corners)

    # --- 2. Data-Driven Benchmark (Single Point) ---
    if median_benchmark is not None:
        label, point_data = list(median_benchmark.items())[0]
        # Project the median point
        median_xy = project_point(point_data, inverted)
        # Plot as a large, distinct star marker
        ax.scatter(median_xy[0], median_xy[1], s=500, c='red', marker='*',
                   edgecolors='white', linewidth=1, alpha=0.8, zorder=20, label=label)

    # --- 3. Labels ---
    font_labels = {'family': 'Arial', 'size': 14, 'weight': 'bold'}
    c1, c2, c3 = corners[0], corners[1], corners[2]
    ax.text(c3[0], c3[1] + 5 * y_dir, components[2].replace('_', ' ').title(), font_labels, ha='center',
            va='center' if inverted else 'bottom')
    ax.text(c2[0] - 5, c2[1] - 5 * y_dir, components[1].replace('_', ' ').title(), font_labels, ha='right',
            va='top' if inverted else 'center')
    ax.text(c1[0] + 5, c1[1] - 5 * y_dir, components[0].replace('_', ' ').title(), font_labels, ha='left',
            va='top' if inverted else 'center')

    # --- 4. Plotting Data Points (with new Jitter method) ---
    points_scaled = proportions * scale
    xy_coords = np.array([project_point(p, inverted) for p in points_scaled])

    # Apply new boundary-constrained jitter
    xy_jittered = boundary_constrained_jitter(xy_coords, triangle_path, strength=5.0)  # Increased strength

    ax.scatter(xy_jittered[:, 0], xy_jittered[:, 1], c=residuals, cmap=custom_cmap, norm=norm,
               s=70, alpha=0.7, edgecolors='k', linewidth=0.5, zorder=10)

    ax.set_aspect('equal')
    ax.axis('off')


def analyze_composition(df, components, reference_component):
    print(f"\n--- Analyzing Components: {' '.join(components)} ---")
    comp_df = df[components].copy() + 1e-9
    comp_proportions = pd.DataFrame(normalize(comp_df, axis=1, norm='l1'), columns=components)
    other_components = [c for c in components if c != reference_component]
    X_ratios = pd.DataFrame()
    for comp in other_components:
        col_name = f"log_ratio_{comp}_vs_{reference_component}"
        X_ratios[col_name] = np.log(comp_proportions[comp] / comp_proportions[reference_component])
    model = sm.OLS(df['ln'], sm.add_constant(X_ratios)).fit()
    return model.resid, comp_proportions.values


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    print("--- Running Compositional Analysis Script (15) ---")
    df, output_dir = setup_environment()

    if df is not None:
        custom_cmap = create_segmented_colormap()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
        fig.subplots_adjust(wspace=0.1)

        # --- Data-Driven Benchmark Calculation ---
        high_pollution_lakes = df[df['ln'] > df['ln'].quantile(0.80)]
        # For Wastewater
        waste_components = ['Primary_Waste_Discharge', 'Secondary_Waste_Discharge', 'Advanced_Waste_Discharge']
        median_comp_waste = high_pollution_lakes[waste_components].median()
        median_comp_waste_norm = (median_comp_waste / median_comp_waste.sum() * 100)
        benchmark_waste = {f"Median of Top 20% Polluted Lakes": median_comp_waste_norm.values}

        # For Roads
        road_components = ['RSE_paved', 'RSE_gravel', 'RSE_other']
        median_comp_road = high_pollution_lakes[road_components].median()
        median_comp_road_norm = (median_comp_road / median_comp_road.sum() * 100)
        benchmark_road = {f"Median of Top 20% Polluted Lakes": median_comp_road_norm.values}

        # --- Analysis and Plotting ---
        res1, points1 = analyze_composition(df, waste_components, 'Advanced_Waste_Discharge')
        res2, points2 = analyze_composition(df, road_components, 'RSE_other')
        all_residuals = np.concatenate([res1, res2])
        norm = TwoSlopeNorm(vcenter=0, vmin=all_residuals.min(), vmax=all_residuals.max())

        # Upright Plot
        ax1.set_title('Wastewater Treatment Type Effect', fontsize=18, pad=20)
        plot_pro_style_ternary(ax1, points1, res1, waste_components, custom_cmap, norm,
                               median_benchmark=benchmark_waste, inverted=False)
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=12, frameon=False)

        # Inverted Plot
        ax2.set_title('Road Surface Type Effect', fontsize=18, pad=20)
        plot_pro_style_ternary(ax2, points2, res2, road_components, custom_cmap, norm, median_benchmark=benchmark_road,
                               inverted=True)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fontsize=12, frameon=False)

        # --- Shared Colorbar ---
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.9, 0.25, 0.015, 0.5])
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Model Residual (ln(Abundance))', fontsize=12)

        save_plot(fig, "15_publication_ternary_final_v2", output_dir)
        print("\nScript finished successfully.")