# -*- coding: utf-8 -*-
#
# VERSION 9.1: Fixes the Venn diagram legend to show level-specific background colors.
#
# This script creates a final map and generates two correct legend versions:
#   1. A comprehensive "equation-style" legend.
#   2. A conceptual "Venn diagram" legend with corrected, level-specific colors.
#

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import warnings
import os

# --- Configuration Section ---
warnings.filterwarnings("ignore")
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42
})

# --- Core File Paths ---
PROJECT_ROOT = r"E:\lake-MP-W"
CATEGORIZED_LAKES_SHP = os.path.join(PROJECT_ROOT, "data", "shp", "all8_categorized_summary.shp")
LOOKUP_CSV = os.path.join(PROJECT_ROOT, "data", "shp", "category_summary_lookup.csv")
WORLD_BORDERS_SHP = os.path.join(PROJECT_ROOT, "data", "base_shp", "world map china line.shp")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "draw", "12_Final_Outputs")


# --- Color Logic and Helper Functions ---
def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#');
    return tuple(int(hex_code[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb_tuple):
    return '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb_tuple)


def blend_colors_hex(hex_colors):
    if not hex_colors: return "#FFFFFF"
    rgb_colors = [hex_to_rgb(c) for c in hex_colors]
    avg_rgb = [sum(components) / len(components) for components in zip(*rgb_colors)]
    return rgb_to_hex(tuple(avg_rgb))


def generate_color_map(lookup_df):
    """Generates the final color dictionary based on the monochromatic scales and blending rules."""
    print("Generating discrete color mapping...")
    color_ramps = {
        'AWD': ['#aed6f1', '#5dade2', '#2e86c1', '#1b4f72'],
        'FISH': ['#abebc6', '#58d68d', '#1e8449', '#145a32'],
        'CULT': ['#fdebd0', '#f5b041', '#af601a', '#6e2c00']
    }
    color_dict = {"No_Impact": "#E7E7E7"}
    for _, row in lookup_df.iterrows():
        cat_name = row['Category_Name']
        if cat_name == "No_Impact": continue
        parts = cat_name.split('_');
        factors_to_blend = []
        all_level_1 = all(p.endswith('1') for p in parts)
        for part in parts:
            factor_type = part[:-1];
            level = int(part[-1])
            if not all_level_1 and level == 1: continue
            if factor_type in color_ramps and level > 0 and level <= len(color_ramps[factor_type]):
                factors_to_blend.append(color_ramps[factor_type][level - 1])
        if len(factors_to_blend) <= 1:
            highest_level_part = max(parts, key=lambda p: int(p[-1]))
            factor_type = highest_level_part[:-1];
            level = int(highest_level_part[-1])
            if factor_type in color_ramps and level <= len(color_ramps[factor_type]):
                color_dict[cat_name] = color_ramps[factor_type][level - 1]
        else:
            color_dict[cat_name] = blend_colors_hex(factors_to_blend)
    return color_dict, color_ramps


# --- Legend Generation Function 1: All-Equation Style ---
def create_all_equation_legend(color_ramps, color_dict, lookup_df, output_path):
    """Creates a comprehensive legend where every category is shown as a visual equation."""
    print("Generating Legend Version 1: Comprehensive 'Equation-Style'...")
    lookup_df['num_factors'] = lookup_df['Category_Name'].apply(lambda x: len(x.split('_')) if x != 'No_Impact' else 0)
    sorted_lookup = lookup_df.sort_values(by=['num_factors', 'Category_Name']).reset_index(drop=True)
    no_impact_row = sorted_lookup[sorted_lookup['Category_Name'] == 'No_Impact']
    standard_categories_df = sorted_lookup[sorted_lookup['Category_Name'] != 'No_Impact']

    num_cols = 2
    num_rows = (len(standard_categories_df) + num_cols) // num_cols
    fig_height = 1.5 + num_rows * 0.45
    fig, ax = plt.subplots(figsize=(12, fig_height));
    ax.set_facecolor('white')
    fig.text(0.5, 0.98, "Legend: Category Color Derivation", ha='center', va='top', fontsize=18, weight='bold')

    y_start, x_start = 0.93, 0.05
    line_height = (y_start - 0.05) / num_rows if num_rows > 0 else 0.1
    col_width = 0.48;
    box_size = 0.025;
    fs = 11
    factor_full_names = {'AWD': 'Industrial Waste', 'FISH': 'Fishery GDP', 'CULT': 'Cultivated Land'}

    for i, (_, row) in enumerate(standard_categories_df.iterrows()):
        col_index = i // num_rows;
        row_index = i % num_rows
        y_pos = y_start - row_index * line_height;
        x_pos = x_start + col_index * col_width
        cat_name = row['Category_Name'];
        color = color_dict.get(cat_name);
        parts = cat_name.split('_')
        current_x = x_pos
        if len(parts) == 1:
            factor_type = cat_name[:-1];
            level = int(cat_name[-1])
            ax.text(current_x, y_pos, f"{factor_full_names[factor_type]} Lvl {level} =", va='center', ha='left',
                    fontsize=fs)
            current_x += 0.22
            ax.add_patch(
                Rectangle((current_x, y_pos - box_size / 2), box_size, box_size, facecolor=color, edgecolor='k',
                          lw=0.5))
            ax.text(current_x + box_size + 0.01, y_pos, cat_name, va='center', ha='left', fontsize=fs, weight='bold')
        else:
            for j, part in enumerate(parts):
                part_factor = part[:-1];
                part_level = int(part[-1])
                part_color = color_ramps[part_factor][part_level - 1]
                ax.add_patch(Rectangle((current_x, y_pos - box_size / 2), box_size, box_size, facecolor=part_color,
                                       edgecolor='k', lw=0.5))
                current_x += box_size
                if j < len(parts) - 1: ax.text(current_x + 0.005, y_pos, "+", va='center', ha='center',
                                               fontsize=fs + 1); current_x += 0.025
            ax.text(current_x + 0.005, y_pos, "=", va='center', ha='center', fontsize=fs + 1);
            current_x += 0.025
            ax.add_patch(
                Rectangle((current_x, y_pos - box_size / 2), box_size, box_size, facecolor=color, edgecolor='k',
                          lw=0.5))
            ax.text(current_x + box_size + 0.01, y_pos, cat_name, va='center', ha='left', fontsize=fs, weight='bold')

    if not no_impact_row.empty:
        i = len(standard_categories_df)
        col_index = i // num_rows;
        row_index = i % num_rows
        y_pos = y_start - row_index * line_height;
        x_pos = x_start + col_index * col_width
        ax.add_patch(Rectangle((x_pos, y_pos - box_size / 2), box_size, box_size, facecolor=color_dict['No_Impact'],
                               edgecolor='k', lw=0.5))
        ax.text(x_pos + box_size + 0.01, y_pos, "No_Impact", va='center', ha='left', fontsize=fs)

    ax.set_axis_off();
    ax.set_xlim(0, 1);
    ax.set_ylim(0, 1)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"--> Successfully saved Equation Legend to: {output_path}")


# --- Legend Generation Function 2: Venn Diagram Style (CORRECTED) ---
def create_venn_legend(color_ramps, color_dict, lookup_df, output_path):
    """Creates a schematic Venn diagram legend with level-specific background colors."""
    print("Generating Legend Version 2: Venn Diagrams (Corrected)...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor='white')
    axes = axes.flatten()
    fig.suptitle("Legend: Intersection of Dominant Factors by Impact Level", fontsize=20, weight='bold')

    lookup_df['level'] = lookup_df['Category_Name'].apply(
        lambda x: int(x[-1]) if x != 'No_Impact' and '_' not in x else (
            int(x.split('_')[0][-1]) if x != 'No_Impact' else 0))

    for level in range(1, 5):
        ax = axes[level - 1];
        ax.set_title(f"Level {level} Impacts", fontsize=14, weight='bold')
        ax.set_aspect('equal');
        ax.set_axis_off();
        ax.set_xlim(0, 10);
        ax.set_ylim(0, 10)

        # --- FIX IS HERE: Dynamically select circle colors based on the current level ---
        awd_bg_color = color_ramps['AWD'][min(level - 1, len(color_ramps['AWD']) - 1)]
        fish_bg_color = color_ramps['FISH'][min(level - 1, len(color_ramps['FISH']) - 1)]
        cult_bg_color = color_ramps['CULT'][min(level - 1, len(color_ramps['CULT']) - 1)]

        venn_props = {
            'AWD': {'center': (4, 6.5), 'radius': 3, 'color': awd_bg_color},
            'FISH': {'center': (6, 6.5), 'radius': 3, 'color': fish_bg_color},
            'CULT': {'center': (5, 3.5), 'radius': 3, 'color': cult_bg_color}
        }
        for props in venn_props.values():
            ax.add_patch(Circle(props['center'], props['radius'], color=props['color'], alpha=0.5, zorder=1))

        ax.text(1.5, 8.5, "AWD", fontsize=12, weight='bold', ha='center')
        ax.text(8.5, 8.5, "FISH", fontsize=12, weight='bold', ha='center')
        ax.text(5, 1, "CULT", fontsize=12, weight='bold', ha='center')

        pos = {
            'AWD': (2.5, 5.5), 'FISH': (7.5, 5.5), 'CULT': (5, 2.5),
            'AWD_FISH': (5, 7.8), 'AWD_CULT': (3.5, 4), 'FISH_CULT': (6.5, 4),
            'AWD_FISH_CULT': (5, 5.2)}
        level_df = lookup_df[lookup_df['level'] == level]
        for _, row in level_df.iterrows():
            cat_name = row['Category_Name'];
            color = color_dict.get(cat_name)
            # A simplified key generation for positioning
            key = '_'.join(sorted([p[:-level] for p in cat_name.split('_')]))
            if key in pos:
                x, y = pos[key];
                rect_size = 0.9
                ax.add_patch(Rectangle((x - rect_size / 2, y - rect_size / 2), rect_size, rect_size, facecolor=color,
                                       edgecolor='k', lw=0.5, zorder=2))
                ax.text(x, y + rect_size * 0.7, cat_name, ha='center', va='center', fontsize=8, zorder=3,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))

    ax_no_impact = fig.add_axes([0.4, 0.05, 0.2, 0.05]);
    ax_no_impact.set_axis_off()
    ax_no_impact.add_patch(Rectangle((0.1, 0.1), 0.2, 0.8, facecolor=color_dict['No_Impact'], edgecolor='k', lw=0.5))
    ax_no_impact.text(0.35, 0.5, "No Impact / No Data", ha='left', va='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"--> Successfully saved Venn Diagram Legend to: {output_path}")


# --- Main Plotting Function ---
def main():
    """Main function to run the entire workflow."""
    print("--- Starting: Unified Map & Legend Generation (v9.1) ---")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print("\nStep 1: Loading data files...")
    try:
        lakes_gdf = gpd.read_file(CATEGORIZED_LAKES_SHP)
        lookup_df = pd.read_csv(LOOKUP_CSV)
        world_gdf = gpd.read_file(WORLD_BORDERS_SHP)
    except Exception as e:
        print(f"FATAL ERROR: {e}");
        return

    print("\nStep 2: Applying discrete color map...")
    category_colors, color_ramps = generate_color_map(lookup_df)
    lakes_gdf['plot_color'] = lakes_gdf['cat_name'].map(category_colors).fillna("#E7E7E7")

    print("\nStep 3: Plotting and saving the main map...")
    if lakes_gdf.crs is None:
        print("WARNING: Input shapefile has no CRS. Assuming WGS 84 (EPSG:4326).")
        lakes_gdf.set_crs(epsg=4326, inplace=True)

    robinson_proj = "+proj=robin +datum=WGS84"
    lakes_proj = lakes_gdf.to_crs(robinson_proj)
    world_proj = world_gdf.to_crs(robinson_proj)

    name_col = next((col for col in ['NAME', 'name', 'ADMIN', 'SOVEREIGNT'] if col in world_proj.columns), 'NAME')
    world_fill = world_proj[world_proj[name_col] != 'Antarctica']
    world_borders_plot = world_fill[world_fill[name_col] != 'Russia']

    fig, ax = plt.subplots(1, 1, figsize=(20, 10));
    ax.set_facecolor('white')
    world_fill.plot(ax=ax, color='white', edgecolor='none', zorder=1)
    lakes_proj.plot(color=lakes_gdf['plot_color'], linewidth=0, edgecolor='none', ax=ax, zorder=2)
    world_borders_plot.plot(ax=ax, color='none', edgecolor='grey', linewidth=0.25, zorder=3)
    ax.set_axis_off()

    map_png_path = os.path.join(OUTPUT_DIR, "global_categorized_map_final.png")
    plt.savefig(map_png_path, dpi=1000, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    print(f"--> Successfully saved main map to: {map_png_path}")

    # --- Step 4: Generate BOTH legend versions ---
    print("\nStep 4: Generating both legend versions...")

    legend1_path = os.path.join(OUTPUT_DIR, "map_legend_V1_equation.pdf")
    create_all_equation_legend(color_ramps, category_colors, lookup_df, legend1_path)

    legend2_path = os.path.join(OUTPUT_DIR, "map_legend_V2_venn.pdf")
    create_venn_legend(color_ramps, category_colors, lookup_df, legend2_path)

    print("\n--- All tasks finished successfully. ---")


# --- Execution Block ---
if __name__ == '__main__':
    main()