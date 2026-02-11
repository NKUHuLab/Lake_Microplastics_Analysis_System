import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
from matplotlib.backends.backend_pdf import PdfPages

# --- 1. Global Setup ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# --- File Paths ---
input_file_path = "E:\\lake-MP-W\\data\\opt\\data\\processed_output\\Xchange.csv"
geopackage_path = "E:\\lake-MP-W\\draw\\11_Geographic_Clustering\\cluster_hulls_WGS84.gpkg"
output_dir = "E:\\lake-MP-W\\draw"
output_file_path = os.path.join(output_dir, "最终版报告_最终定制Page3.pdf")

# --- [MODIFICATION] New directory for detailed CSV outputs ---
csv_output_dir = os.path.join(output_dir, "output_data_details")
os.makedirs(csv_output_dir, exist_ok=True)

# Coordinate column names
LONGITUDE_COL = 'lon'
LATITUDE_COL = 'lat'

# --- 2. Risk Zone Definitions & Colors ---
risk_zone_mapping = {
    16: 'Risk Zone a', 0: 'Risk Zone b', 12: 'Risk Zone b', 14: 'Risk Zone b', 7: 'Risk Zone c',
    11: 'Risk Zone d', 17: 'Risk Zone d', 4: 'Risk Zone e', 13: 'Risk Zone f', 18: 'Risk Zone f', 9: 'Risk Zone g'
}
risk_zone_color_map = {
    "Risk Zone a": "#ff6347", "Risk Zone b": "#4682b4", "Risk Zone c": "#32cd32",
    "Risk Zone d": "#ffa500", "Risk Zone e": "#9370db", "Risk Zone f": "#00ced1",
    "Risk Zone g": "#ff1493", "Outside Risk Zones": "#778899"
}

# --- 3. Data Loading and Merging ---
# ... (This section remains unchanged, so it's omitted for brevity) ...
try:
    main_df = pd.read_csv(input_file_path)
    risk_zones_gdf = gpd.read_file(geopackage_path)
    points_gdf = gpd.GeoDataFrame(main_df, geometry=gpd.points_from_xy(main_df[LONGITUDE_COL], main_df[LATITUDE_COL]),
                                  crs="EPSG:4326")
    merged_gdf = gpd.sjoin(points_gdf, risk_zones_gdf[['cluster_id', 'geometry']], how="left", predicate='within')
    merged_gdf['Risk Zone'] = merged_gdf['cluster_id'].map(risk_zone_mapping).fillna("Outside Risk Zones")
    merged_gdf.drop_duplicates(subset=main_df.columns, keep='first', inplace=True)
    plot_df = pd.DataFrame(merged_gdf.drop(columns='geometry'))
except Exception as e:
    print(f"Error during data loading: {e}. Using placeholder data.")
    main_df = pd.DataFrame({
        'Advanced_Waste_Discharge': np.random.lognormal(3, 2, 2000), 'fish_gdp_sqkm': np.random.uniform(-100, -1, 2000),
        'Cultivated_land': np.random.uniform(-100, -1, 2000), 'change': np.random.randn(2000) * 100,
        LATITUDE_COL: np.random.uniform(-90, 90, 2000), LONGITUDE_COL: np.random.uniform(-180, 180, 2000)
    })
    plot_df = main_df.copy()
    plot_df['Risk Zone'] = "Outside Risk Zones"

# --- 4. Data Pre-processing ---
# ... (This section remains unchanged) ...
plot_df['abs_change'] = plot_df['change'].abs()
q99 = plot_df['abs_change'].quantile(0.998)
plot_df['abs_change_clipped'] = plot_df['abs_change'].clip(upper=q99)


# --- 5. Advanced Plotting Functions (Modified to return data) ---

def get_binned_data(data, x_col, y_col, use_log, config):
    # ... (This function remains unchanged as it already returns the data) ...
    key_columns = [x_col, y_col, 'Risk Zone'];
    plot_data = data.dropna(subset=key_columns).copy()
    plot_data['x_transformed'] = np.log1p(np.abs(plot_data[x_col])) if use_log else np.abs(plot_data[x_col])
    if plot_data.empty or not np.isfinite(plot_data['x_transformed']).all(): return None
    bin_count = 50;
    min_val, max_val = plot_data['x_transformed'].min(), plot_data['x_transformed'].max()
    bins = np.linspace(min_val, max_val, bin_count + 1)
    plot_data['x_bin'] = pd.cut(plot_data['x_transformed'], bins=bins, right=False, labels=bins[:-1],
                                include_lowest=True)
    all_agg_data = {}
    smoothing_window = 8
    for category in sorted(plot_df['Risk Zone'].unique()):
        subset = plot_data[plot_data['Risk Zone'] == category]
        if subset.empty: continue
        agg = subset.groupby('x_bin', observed=False).agg(median=(y_col, 'median'),
                                                          p25=(y_col, lambda x: x.quantile(0.25)),
                                                          p75=(y_col, lambda x: x.quantile(0.75)),
                                                          count=(y_col, 'size')).reset_index().dropna(subset=['median'])
        agg = agg[agg['count'] >= 5]
        if agg.empty: continue
        agg['median_smoothed'] = agg['median'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        agg['p25_smoothed'] = agg['p25'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        agg['p75_smoothed'] = agg['p75'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        agg['bin_center'] = agg['x_bin'].astype(float) + (bins[1] - bins[0]) / 2 if len(bins) > 1 else 0
        all_agg_data[category] = agg
    return all_agg_data


def create_main_plot(ax, data, x_col, y_col, xlabel, use_log, config):
    all_agg_data = get_binned_data(data, x_col, y_col, use_log, config)
    if not all_agg_data: return all_agg_data, None
    # ... (Plotting logic is the same) ...
    highlight_zones = ['Risk Zone b']
    if config['name'] == 'Cultivated Land': highlight_zones = ['Risk Zone a']
    for category, agg in all_agg_data.items():
        is_highlight = category in highlight_zones
        line_alpha = 1.0 if is_highlight else 0.6;
        fill_alpha = 0.2 if is_highlight else 0.1
        ax.plot(agg['bin_center'], agg['median_smoothed'], label=category, color=risk_zone_color_map.get(category),
                linestyle='-', lw=3.0 if is_highlight else 1.5, alpha=line_alpha, zorder=10 if is_highlight else 5)
        ax.fill_between(agg['bin_center'], agg['p25_smoothed'], agg['p75_smoothed'],
                        color=risk_zone_color_map.get(category), alpha=fill_alpha)

    golden_interval_data = None
    if highlight_zones and highlight_zones[0] in all_agg_data:
        hz_data = all_agg_data[highlight_zones[0]]
        if not hz_data.empty and not hz_data['median_smoothed'].isnull().all():
            peak_idx = hz_data['median_smoothed'].idxmax()
            peak_x = hz_data.loc[peak_idx, 'bin_center']
            peak_y = hz_data.loc[peak_idx, 'median_smoothed']
            interval_width = 1.0
            ax.axvspan(peak_x - interval_width / 2, peak_x + interval_width / 2, color='gold', alpha=0.3, zorder=1)
            ax.annotate('Golden Interval', xy=(peak_x, peak_y), xytext=(peak_x, peak_y * 1.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8), fontsize=12, ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="gold", alpha=0.8))
            golden_interval_data = {'Factor': config['name'], 'Golden_Interval_Center': peak_x,
                                    'Golden_Interval_Peak_Effect': peak_y, 'Interval_Width': interval_width}

    ax.set_xlabel(xlabel, fontsize=12);
    ax.set_ylim(bottom=0)
    return all_agg_data, golden_interval_data


def create_marginal_effect_plot(ax, data, x_col, y_col, xlabel, use_log, config):
    all_agg_data = get_binned_data(data, x_col, y_col, use_log, config)
    if not all_agg_data: return all_agg_data, None
    # ... (Plotting logic is the same) ...
    if config['name'] in ['Fishery Value', 'Advanced Waste Discharge']:
        highlight_zones = ['Risk Zone b']
    else:
        highlight_zones = []

    for category, agg in all_agg_data.items():
        if len(agg) < 2: continue
        slope = np.gradient(agg['median_smoothed'], agg['bin_center'])
        if config['name'] == 'Advanced Waste Discharge':
            agg['slope_to_plot'] = slope
        elif config['name'] == 'Fishery Value':
            agg['slope_to_plot'] = pd.Series(slope).rolling(window=3, center=True, min_periods=1).mean()
        else:
            agg['slope_to_plot'] = pd.Series(slope).rolling(window=5, center=True, min_periods=1).mean()
        is_highlight = category in highlight_zones
        ax.plot(agg['bin_center'], agg['slope_to_plot'], label=category, color=risk_zone_color_map.get(category),
                linestyle='-', lw=2.5 if is_highlight else 1.0, alpha=1.0 if is_highlight else 0.6)

    interval_data = None
    if highlight_zones and highlight_zones[0] in all_agg_data:
        hz_data = all_agg_data[highlight_zones[0]].copy();
        hz_data.reset_index(drop=True, inplace=True)
        if 'slope_to_plot' in hz_data.columns and not hz_data.empty:
            interval_data = {'Factor': config['name']}
            if not hz_data['slope_to_plot'].isnull().all() and hz_data['slope_to_plot'].max() > 0:
                idxmax_pos = hz_data['slope_to_plot'].idxmax();
                start_pos = idxmax_pos
                while start_pos > 0 and hz_data.loc[start_pos - 1, 'slope_to_plot'] > 0: start_pos -= 1
                end_pos = idxmax_pos
                while end_pos < len(hz_data) - 1 and hz_data.loc[end_pos + 1, 'slope_to_plot'] > 0: end_pos += 1
                x_start = hz_data.loc[start_pos, 'bin_center'];
                x_end = hz_data.loc[end_pos, 'bin_center']
                ax.axvspan(x_start, x_end, color='green', alpha=0.15, zorder=1)
                ax.text((x_start + x_end) / 2, ax.get_ylim()[1] * 0.9, 'Primary Increase Region', ha='center', va='top',
                        color='green', weight='bold')
                interval_data.update({'Increase_Region_Start': x_start, 'Increase_Region_End': x_end})
            if not hz_data['slope_to_plot'].isnull().all() and hz_data['slope_to_plot'].min() < 0 and config[
                'name'] != 'Advanced Waste Discharge':
                idxmin_pos = hz_data['slope_to_plot'].idxmin();
                start_pos = idxmin_pos
                while start_pos > 0 and hz_data.loc[start_pos - 1, 'slope_to_plot'] < 0: start_pos -= 1
                end_pos = idxmin_pos
                while end_pos < len(hz_data) - 1 and hz_data.loc[end_pos + 1, 'slope_to_plot'] < 0: end_pos += 1
                x_start = hz_data.loc[start_pos, 'bin_center'];
                x_end = hz_data.loc[end_pos, 'bin_center']
                ax.axvspan(x_start, x_end, color='red', alpha=0.15, zorder=1)
                ax.text((x_start + x_end) / 2, ax.get_ylim()[0] * 0.9, 'Primary Decrease Region', ha='center',
                        va='bottom', color='red', weight='bold')
                interval_data.update({'Decrease_Region_Start': x_start, 'Decrease_Region_End': x_end})

    ax.axhline(0, color='grey', linestyle='--', lw=1);
    ax.set_xlabel(xlabel, fontsize=12);
    ax.set_ylabel("Marginal Effect (Slope)", fontsize=12);
    ax.set_title(f"Sensitivity to {config['name']}", fontsize=14)
    return all_agg_data, interval_data


def create_summary_barplot(fig, data, factors):
    # ... (This function remains unchanged, but we'll capture its data source) ...
    summary_list = []
    for factor in factors:
        factor_name = factor['name'];
        subset = data[data[factor['x_col']].notna() & (data[factor['x_col']] != 0)]
        if not subset.empty:
            summary = subset.groupby('Risk Zone')['abs_change_clipped'].median().reset_index()
            summary['Factor'] = factor_name;
            summary.rename(columns={'abs_change_clipped': 'Median Effect'}, inplace=True)
            summary_list.append(summary)
    if not summary_list:
        ax = fig.add_subplot(111);
        ax.text(0.5, 0.5, "Error: No non-zero data found.", ha='center');
        return None
    summary_df = pd.concat(summary_list, ignore_index=True)
    ax = fig.add_subplot(111)
    zone_order = sorted(data['Risk Zone'].unique(), key=lambda x: (x.startswith("Outside"), x))
    sns.barplot(data=summary_df, x='Risk Zone', y='Median Effect', hue='Factor', ax=ax, palette='viridis',
                order=zone_order)
    ax.set_title("Overall Impact Comparison of Factors Across All Zones", fontsize=16, pad=20);
    ax.set_ylabel("Median Optimization Effect (MP Reduction)", fontsize=12);
    ax.set_xlabel("Zone", fontsize=12)
    ax.tick_params(axis='x', rotation=45);
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    if not summary_df.empty: ax.set_ylim(top=summary_df['Median Effect'].max() * 1.1)
    return summary_df


# --- 6. Main Execution ---
plot_configs = [
    {'name': 'Fishery Value', 'x_col': 'fish_gdp_sqkm', 'x_label': 'log(abs(Fishery Value) + 1)', 'use_log': True},
    {'name': 'Advanced Waste Discharge', 'x_col': 'Advanced_Waste_Discharge',
     'x_label': 'log(abs(Waste Discharge) + 1)', 'use_log': True},
    {'name': 'Cultivated Land', 'x_col': 'Cultivated_land', 'x_label': 'abs(Cultivated Land)', 'use_log': False}
]

annotations_summary_list = []

print(f"\n--- Generating final customized analysis PDF report at: {output_file_path} ---")
with PdfPages(output_file_path) as pdf:
    # --- PAGE 1 ---
    print("Generating Page 1 and exporting its data...")
    sns.set_style("whitegrid")
    fig1, axes1 = plt.subplots(3, 1, figsize=(8, 18), sharey=True)
    for i, config in enumerate(plot_configs):
        plot_data, golden_data = create_main_plot(axes1[i], plot_df.copy(), config['x_col'], 'abs_change_clipped',
                                                  config['x_label'], config['use_log'], config)
        if plot_data:
            # Combine dict of dataframes into a single dataframe and save
            long_df = pd.concat(plot_data.values(), keys=plot_data.keys(),
                                names=['Risk Zone', 'original_index']).reset_index()
            long_df.to_csv(os.path.join(csv_output_dir, f"page1_data_{config['name'].replace(' ', '_')}.csv"),
                           index=False)
        if golden_data:
            annotations_summary_list.append(golden_data)
    # ... (legend and layout code is the same) ...
    fig1.text(-0.02, 0.5, "Optimization Effect (MP Reduction)", va='center', rotation='vertical', fontsize=16)
    handles, labels = axes1[-1].get_legend_handles_labels();
    by_label = dict(zip(labels, handles))
    sorted_labels = sorted(by_label.keys(), key=lambda x: (x.startswith("Outside"), x));
    sorted_handles = [by_label[k] for k in sorted_labels]
    fig1.legend(sorted_handles, sorted_labels, title="Risk Zone", loc='center left', bbox_to_anchor=(1.02, 0.5),
                fontsize=12, title_fontsize=14)
    fig1.suptitle("Page 1: Smoothed Impact of Drivers on Optimization Effect", fontsize=20, y=0.96);
    fig1.tight_layout(rect=[0.03, 0, 0.85, 0.95]);
    pdf.savefig(fig1, bbox_inches='tight');
    plt.close(fig1)

    # --- PAGE 2 ---
    print("Generating Page 2 and exporting its data...")
    fig2 = plt.figure(figsize=(20, 7))
    summary_df_for_barplot = create_summary_barplot(fig2, plot_df.copy(), plot_configs)
    if summary_df_for_barplot is not None:
        summary_df_for_barplot.to_csv(os.path.join(csv_output_dir, "page2_data_barchart_summary.csv"), index=False)
    fig2.suptitle("Page 2: Summary of Factor Effectiveness Comparison", fontsize=20, y=0.98);
    fig2.tight_layout();
    pdf.savefig(fig2);
    plt.close(fig2)

    # --- PAGE 3 ---
    print("Generating Page 3 and exporting its data...")
    fig3, axes3 = plt.subplots(3, 1, figsize=(8, 18))
    for i, config in enumerate(plot_configs):
        plot_data, interval_data = create_marginal_effect_plot(axes3[i], plot_df.copy(), config['x_col'],
                                                               'abs_change_clipped', config['x_label'],
                                                               config['use_log'], config)
        if plot_data:
            long_df = pd.concat(plot_data.values(), keys=plot_data.keys(),
                                names=['Risk Zone', 'original_index']).reset_index()
            long_df.to_csv(os.path.join(csv_output_dir, f"page3_data_{config['name'].replace(' ', '_')}.csv"),
                           index=False)
        if interval_data:
            annotations_summary_list.append(interval_data)
    # ... (legend and layout code is the same) ...
    fig3.text(-0.02, 0.5, "Marginal Effect (Slope of Effect Curve)", va='center', rotation='vertical', fontsize=16);
    handles, labels = axes3[-1].get_legend_handles_labels();
    by_label = dict(zip(labels, handles));
    sorted_labels = sorted(by_label.keys());
    sorted_handles = [by_label[k] for k in sorted_labels]
    fig3.legend(sorted_handles, sorted_labels, title="Risk Zone", loc='center left', bbox_to_anchor=(1.02, 0.5),
                fontsize=12, title_fontsize=14)
    fig3.suptitle("Page 3: Customized Sensitivity Analysis", fontsize=20, y=0.96);
    fig3.tight_layout(rect=[0.03, 0, 0.85, 0.95]);
    pdf.savefig(fig3, bbox_inches='tight');
    plt.close(fig3)

# --- Save annotations summary ---
if annotations_summary_list:
    annotations_df = pd.DataFrame(annotations_summary_list)
    annotations_df.to_csv(os.path.join(csv_output_dir, "annotations_summary.csv"), index=False)
    print(f"\n✅ Annotation summary data saved to: {os.path.join(csv_output_dir, 'annotations_summary.csv')}")

print(f"\n✅ Success! The final customized analysis report has been saved to: {output_file_path}")
print(f"✅ All detailed data files have been saved in: {csv_output_dir}")