# 16_independent_variable_analysis.py
# Final professional version with advanced visualization in English.

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

# Import the setup functions from your data preparation script
from data_preparation import setup_environment, save_plot


def plot_lollipop_professional(effects, output_dir):

    # --- 1. Data Preparation ---
    data = pd.DataFrame({'effect': effects}).sort_values(by='effect')
    data['abs_effect'] = abs(data['effect'])

    # --- 2. Define Visual Mappings ---
    # Mapping 1: Head Size -> Effect Magnitude (Absolute Value)
    min_size, max_size = 60, 500
    data['size'] = np.interp(data['abs_effect'],
                             (data['abs_effect'].min(), data['abs_effect'].max()),
                             (min_size, max_size))

    # Mapping 2: Head Color -> Effect Magnitude (Absolute Value)
    cmap = plt.get_cmap('cividis')  # A perceptually uniform colormap
    norm = plt.Normalize(vmin=data['abs_effect'].min(), vmax=data['abs_effect'].max())

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(14, 12))

    # Vertical line at x=0 for clear separation of positive/negative effects
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=1.2, zorder=0)

    # Draw thicker, black "stems"
    ax.hlines(y=data.index, xmin=0, xmax=data['effect'], color='black', linewidth=2.5, zorder=1)

    # Draw gradient-colored, variable-sized "heads"
    scatter = ax.scatter(data['effect'], data.index,
                         s=data['size'], c=data['abs_effect'], cmap=cmap, norm=norm,
                         alpha=0.9, zorder=2, edgecolors='white', linewidth=1)

    # --- 4. Labeling: Place text next to each point ---
    for i, row in data.iterrows():
        # Place text to the right for positive effects, left for negative
        ha = 'left' if row['effect'] >= 0 else 'right'
        offset = 0.008 if row['effect'] >= 0 else -0.008
        ax.text(row['effect'] + offset, i, i,
                ha=ha, va='center', fontsize=12, fontdict={'family': 'Arial'})

    # --- 5. Aesthetic Final Touches ---
    ax.spines[['left', 'top', 'right']].set_visible(False)  # Remove plot frame
    ax.tick_params(axis='y', length=0, labelsize=0)  # Hide y-axis ticks and labels
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlabel('Effect Coefficient (Standardized)', fontsize=14, fontdict={'family': 'Arial'})
    ax.set_title('Standardized Variable Effects on ln(Abundance)', fontsize=22, pad=20, fontdict={'family': 'Arial'})

    # --- 6. Legends ---
    # 6a. Vertical Colorbar Legend for Color
    # Position: [left, bottom, width, height] in figure-relative coordinates
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Effect Magnitude (Absolute Coefficient)', size=12, fontdict={'family': 'Arial'})

    # 6b. Custom Visual Legend for Size
    # We will plot "dummy" points in a clear area to create the legend
    legend_handles = []
    # Create 4 representative sizes for the legend
    legend_effects = np.linspace(data['abs_effect'].min(), data['abs_effect'].max(), 4)
    legend_sizes = np.interp(legend_effects,
                             (data['abs_effect'].min(), data['abs_effect'].max()),
                             (min_size, max_size))

    for i, size in enumerate(legend_sizes):
        legend_handles.append(ax.scatter([], [], s=size, color='#64748B',  # A neutral grey
                                         label=f'{legend_effects[i]:.2f}'))

    legend = ax.legend(handles=legend_handles,
                       title='Effect Magnitude\n(by Size)',
                       loc='upper left',
                       bbox_to_anchor=(0.0, 0.95),  # Position legend inside the plot
                       frameon=False,
                       labelspacing=2.0,  # Increase spacing between legend items
                       scatterpoints=1)

    plt.setp(legend.get_title(), fontsize=14, fontfamily='Arial')
    plt.setp(legend.get_texts(), fontsize=12, fontfamily='Arial')

    plt.tight_layout(rect=[0, 0, 0.88, 1])  # Adjust layout to make room for the colorbar
    save_plot(fig, "16_ridge_effects_lollipop_professional", output_dir)


if __name__ == '__main__':
    # --- Global Font Settings ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    print("--- Running Independent Variable Analysis Script (v.Pro) ---")
    df, output_dir = setup_environment()

    if df is not None:
        # --- 1. Define all predictors to be included in the model ---
        all_predictors = [
            'Primary_Waste_Discharge', 'Secondary_Waste_Discharge', 'Advanced_Waste_Discharge',
            'RSE_paved', 'RSE_gravel', 'RSE_other',
            'Mismanaged', 'fish_gdp_sqkm', 'Cultivated_land', 'Artificial_surface'
        ]
        X_df = df[all_predictors].copy()
        Y_df = df['ln']

        # --- 2. Preprocessing and Standardization ---
        log_transform_cols = ['Mismanaged', 'fish_gdp_sqkm']
        for col in log_transform_cols:
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=0)
                X_df[col] = np.log1p(X_df[col])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=all_predictors)

        # --- 3. Modeling with Ridge Regression ---
        print("\n--- Fitting RidgeCV Regression Model (includes all variables) ---")
        # RidgeCV automatically finds the best regularization strength (alpha)
        ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 100)).fit(X_scaled_df, Y_df)

        print(f"Optimal alpha found by RidgeCV: {ridge_model.alpha_:.4f}")

        # Extract effect coefficients
        effects = pd.Series(ridge_model.coef_, index=all_predictors)
        print("\nStandardized Effect Coefficients:")
        print(effects.sort_values(ascending=False))

        # --- 4. Visualization ---
        plot_lollipop_professional(effects, output_dir)

        print("\nScript finished successfully.")