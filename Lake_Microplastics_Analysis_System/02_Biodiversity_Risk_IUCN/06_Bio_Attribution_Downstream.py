import os
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

# --- 1. 全局路径和参数配置 ---
PROJECT_ROOT_ANI = r"E:\lake-MP-ani"
PROJECT_ROOT_W = r"E:\lake-MP-W"
PLOT_OUTPUT_DIR = r"E:\lake-MP-W\draw\IUCN_Advanced"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

DATA_SUMMARY_DIR = os.path.join(PROJECT_ROOT_ANI, "draw", "Biological_Attribution_Analysis_SJoin_Fast",
                                "data_summaries")
INTERMEDIATE_SJOIN_GPKG = os.path.join(DATA_SUMMARY_DIR, "intermediate_lake_species_intersections.gpkg")
POLLUTION_HULLS_GPKG = os.path.join(PROJECT_ROOT_W, "draw", "11_Geographic_Clustering", "cluster_hulls_WGS84.gpkg")

THREAT_WEIGHTS = {'Critically Endangered': 50, 'Endangered': 25, 'Vulnerable': 10, 'Near Threatened': 5}
PARTITION_MAP = {16: 'a', 0: 'b', 12: 'b', 14: 'b', 7: 'c', 11: 'd', 17: 'd', 4: 'e', 13: 'f', 18: 'f', 9: 'g'}
ORIGIN_MAP = {1: 'Native', 2: 'Reintroduced', 3: 'Introduced'}
POLYGON_ID_COLUMN = 'hylak_id'

# --- 2. 全局绘图设置 ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid", font="Arial")
warnings.filterwarnings("ignore")


# --- [THE MASTERPIECE: A CUSTOM NORMALIZATION ENGINE] ---
class PiecewiseLogNorm(colors.Normalize):
    """
    A custom, continuous normalization engine.
    It makes the visual length of specified intervals equal.
    """
    def __init__(self, breakpoints, vmin=None, vmax=None, clip=False):
        self.breakpoints = np.array(breakpoints)
        # Define the desired equidistant points in normalized [0, 1] space
        self.norm_points = np.linspace(0, 1, len(self.breakpoints))
        super().__init__(vmin=self.breakpoints[0], vmax=self.breakpoints[-1], clip=clip)

    def __call__(self, value, clip=None):
        # We perform a piecewise linear interpolation on the LOG of the values.
        # This creates a smooth, continuous scale bent to your exact specifications.
        log_value = np.log10(np.ma.asarray(value))
        log_breakpoints = np.log10(self.breakpoints)
        return np.ma.masked_invalid(np.interp(log_value, log_breakpoints, self.norm_points))


# --- [END OF CUSTOM ENGINE] ---


def robust_bool_converter(series):
    map_dict = {'True': True, 'False': False, 'true': True, 'false': False,
                'TRUE': True, 'FALSE': False, 1: True, 0: False,
                1.0: True, 0.0: False, True: True, False: False}
    return series.map(map_dict).fillna(False)


def run_habitat_census(gdf, output_dir):
    print("\n--- [DELIVERABLE] Performing Polygon Habitat Census... ---")
    try:
        gdf['habitat_str'] = "F_" + gdf['freshwater'].astype(str) + "_M_" + gdf['marine'].astype(str) + "_T_" + gdf[
            'terrestria'].astype(str)
        habitat_counts = gdf.groupby('habitat_str')[POLYGON_ID_COLUMN].nunique().reset_index()
        habitat_counts.rename(columns={POLYGON_ID_COLUMN: 'unique_polygon_count'}, inplace=True)
        habitat_counts = habitat_counts.sort_values(by='unique_polygon_count', ascending=False)
        output_path = os.path.join(output_dir, "habitat_polygon_counts.csv")
        habitat_counts.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"   - [成功] 已将栖息地类型普查结果导出至: {output_path}")
    except Exception as e:
        print(f"   - [错误] 栖息地普查失败: {e}")


# --- [V39 THE FINAL, PERFECTED VISUALIZATION FUNCTION] ---
def generate_final_bubble_plot(long_df, output_dir):
    """
    V39: The definitive version with the true custom continuous scale and intelligent label spacing.
    """
    print(f"\n--- [绘图核心] 生成最终美化版图表... ---")
    try:
        long_df['Capped_Unique_Species'] = long_df['Unique_Species'].clip(upper=1000)
        long_df['size_log'] = np.log1p(long_df['Capped_Unique_Species'])
        dim_order = ['Lifecycle Group', 'Habitat Dependency', 'Origin Label', 'Risk Zone']
        long_df['Analysis_Dimension'] = pd.Categorical(long_df['Analysis_Dimension'], categories=dim_order,
                                                       ordered=True)
        long_df = long_df.sort_values(['Analysis_Dimension', 'Category'])
        long_df['x_category'] = long_df['Analysis_Dimension'].astype(str) + ":\n" + long_df['Category'].astype(str)
        fig, ax = plt.subplots(figsize=(48, 16))

        # --- [BESPOKE VISUALIZATION IMPLEMENTED] ---
        max_risk_value = long_df['Average_Risk_per_Species'].max()
        if pd.isna(max_risk_value) or max_risk_value <= 100: max_risk_value = 1000

        # 1. Generate the breakpoints for our TRUE custom scale: 1, 100, 1000, ...
        max_power = int(np.ceil(np.log10(max_risk_value)))
        breakpoints = [1]
        if max_power >= 2:
            breakpoints.extend([10 ** p for p in range(2, max_power + 1)])

        # 2. Instantiate our custom normalization engine with these breakpoints.
        custom_norm = PiecewiseLogNorm(breakpoints=breakpoints)

        scatter = ax.scatter(
            x=long_df['x_category'], y=long_df['class'], s=long_df['size_log'] * 320,
            c=long_df['Average_Risk_per_Species'], cmap='RdYlBu_r', norm=custom_norm,
            alpha=0.8, linewidths=0.5, edgecolors='black')

        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.tick_params(axis='y', rotation=45, labelsize=16)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7);
        ax.grid(False, axis='x')
        ax.spines['top'].set_visible(False);
        ax.spines['right'].set_visible(False)
        all_x_cats = list(long_df['x_category'].unique())
        dimension_boundaries = long_df.drop_duplicates('x_category')['Analysis_Dimension'].ne(
            long_df.drop_duplicates('x_category')['Analysis_Dimension'].shift())
        boundary_indices = dimension_boundaries[dimension_boundaries].index
        for idx in boundary_indices[1:]:
            x_pos = all_x_cats.index(long_df.loc[idx, 'x_category'])
            ax.axvline(x=x_pos - 0.5, color='black', linestyle='--', linewidth=2)
        for dim in dim_order:
            subset = long_df[long_df['Analysis_Dimension'] == dim]
            if not subset.empty and len(subset['x_category'].unique()) > 0:
                positions = [all_x_cats.index(cat) for cat in subset['x_category'].unique() if cat in all_x_cats]
                if positions:
                    label_pos = (min(positions) + max(positions)) / 2.0
                    ax.text(label_pos, ax.get_ylim()[1] * 1.01, dim, ha='center', va='bottom', fontsize=20,
                            weight='bold')

        cbar = fig.colorbar(scatter, ax=ax, pad=0.01, aspect=40)
        cbar.set_label('Average Risk per Species (Custom Continuous Scale)', rotation=270, labelpad=25, fontsize=18)

        # 3. Intelligent Ticks: Use the breakpoints for the scale, but only display a subset to prevent overlap.
        ticks_to_display = breakpoints
        if len(breakpoints) > 5:  # If there are many breakpoints, don't show all labels
            ticks_to_display = breakpoints[::2]  # Show every second one
            if breakpoints[-1] not in ticks_to_display:
                ticks_to_display.append(breakpoints[-1])  # Always show the max value at the end

        cbar.set_ticks(ticks_to_display)
        cbar.ax.set_yticklabels([f'{tick:,.0f}' for tick in ticks_to_display])
        # --- [END OF VISUALIZATION] ---

        legend_species_counts = [1, 10, 100, 1000]
        legend_sizes_s = [np.log1p(count) * 120 for count in legend_species_counts]
        legend_handles = [plt.scatter([], [], s=s_val, color='grey', edgecolor='black', alpha=0.7) for s_val in
                          legend_sizes_s]
        legend_labels = [str(count) for count in legend_species_counts]
        size_legend = ax.legend(legend_handles, legend_labels, title='Unique Species Count\n(capped at 1000)',
                                loc='upper left', bbox_to_anchor=(1.015, 1), fontsize=16, title_fontsize=18)
        title = f'Unified Risk Analysis (Top {long_df["class"].nunique()} Classes by Record Count)'
        fig.suptitle(title, fontsize=30, weight='bold')
        ax.set_ylabel("Taxonomic Class", fontsize=20)
        plt.tight_layout(rect=[0.08, 0.2, 0.93, 0.92])
        output_path = os.path.join(output_dir, "Advanced_Plot_Unified_BubbleChart_Final_Polished.pdf")
        plt.savefig(output_path, format='pdf')
        plt.close('all')
        print(f"   - [成功] 已保存最终美化版图表至: {output_path}")
    except Exception as e:
        print(f"   - [错误] 绘制图表失败: {e}")


# --- 5. 主分析流程 ---
def run_final_analysis_v38():
    print("--- [初始化] 最终版高级分析流程 (V38 - 定制化连续色阶) ---")
    print(f"\n--- [步骤 1/5] 加载完整核心数据... ---")
    try:
        gdf = gpd.read_file(INTERMEDIATE_SJOIN_GPKG, engine='pyogrio')
        gdf_hulls = gpd.read_file(POLLUTION_HULLS_GPKG, engine='pyogrio')
    except Exception as e:
        print(f"[严重错误] 加载核心文件失败: {e}");
        return

    print(f"\n--- [步骤 2/5] 核心逻辑: 对所有记录进行栖息地分类... ---")
    print("   - [关键修复] 应用鲁棒的布尔值转换，防止数据类型错误...")
    for col in ['freshwater', 'marine', 'terrestria']:
        gdf[col] = robust_bool_converter(gdf[col])
    run_habitat_census(gdf.copy(), PLOT_OUTPUT_DIR)
    conditions = [
        (gdf['freshwater']) & (gdf['marine']) & (gdf['terrestria']),
        (gdf['freshwater']) & (gdf['marine']) & (~gdf['terrestria']),
        (gdf['freshwater']) & (~gdf['marine']) & (gdf['terrestria']),
        (~gdf['freshwater']) & (gdf['marine']) & (gdf['terrestria']),
        (gdf['freshwater']) & (~gdf['marine']) & (~gdf['terrestria']),
        (~gdf['freshwater']) & (gdf['marine']) & (~gdf['terrestria']),
        (~gdf['freshwater']) & (~gdf['marine']) & (gdf['terrestria']),
    ]
    choices = ['FMT-Generalist', 'Freshwater&Marine', 'Freshwater&Terrestrial', 'Marine&Terrestrial', 'Freshwater-Only',
               'Marine-Only', 'Terrestrial-Only']
    gdf['habitat_dependency'] = np.select(conditions, choices, default='Undefined')
    print(f"   - [成功] 已对所有 {len(gdf)} 条记录完成栖息地分类。")
    print(f"   - [勘探] 在完整数据中发现的类型: {gdf['habitat_dependency'].unique().tolist()}")

    print(f"\n--- [步骤 3/5] 数据筛选与丰富化... ---")
    top_classes = gdf['class'].value_counts().nlargest(30).index
    print(f"   - 筛选逻辑: 现在将分析限制在记录数最多的前 {len(top_classes)} 个生物大类。")
    gdf_filtered = gdf[gdf['class'].isin(top_classes)].copy()
    gdf_filtered['lifecycle_group'] = np.where(gdf_filtered['seasonal'] == 2, 'Breeding Area', 'Other Areas')
    gdf_filtered['origin_label'] = gdf_filtered['origin'].map(ORIGIN_MAP).fillna('Unknown')
    gdf_filtered = gpd.sjoin(gdf_filtered.to_crs(gdf_hulls.crs), gdf_hulls[['cluster_id', 'geometry']], how='left',
                             predicate='intersects')
    gdf_filtered['risk_zone'] = gdf_filtered['cluster_id'].map(PARTITION_MAP).fillna('Outside Zones')
    gdf_filtered['Threat_Score'] = gdf_filtered['redlist_category'].map(THREAT_WEIGHTS).fillna(0)
    print("   - [INSIGHT] 应用逻辑修复：负数的IRS_baseline值在此次计算中将被视为零风险。")
    gdf_filtered['IRS_baseline_risk'] = gdf_filtered['IRS_baseline'].clip(lower=0)
    gdf_filtered['Weighted_Threat_Risk'] = gdf_filtered['IRS_baseline_risk'].fillna(0) * gdf_filtered['Threat_Score']
    df_plot = gdf_filtered[~gdf_filtered['habitat_dependency'].isin(['Undefined', 'Terrestrial-Only'])].copy()

    print(f"\n--- [步骤 4/5] 按类别聚合数据... ---")
    dims_to_plot = ['lifecycle_group', 'habitat_dependency', 'origin_label', 'risk_zone']
    all_dims_df_list = []
    for dim in dims_to_plot:
        agg_df = df_plot.groupby(['class', dim]).agg(Total_Risk=('Weighted_Threat_Risk', 'sum'),
                                                     Unique_Species=('SCI_NAME', 'nunique')).reset_index()
        agg_df = agg_df[agg_df['Unique_Species'] > 0]
        agg_df['Average_Risk_per_Species'] = agg_df['Total_Risk'] / agg_df['Unique_Species']
        agg_df['Analysis_Dimension'] = dim.replace('_', ' ').title()
        agg_df.rename(columns={dim: 'Category'}, inplace=True)
        all_dims_df_list.append(agg_df)
    if not all_dims_df_list:
        print("[严重错误] 聚合后没有生成任何可供绘图的数据。请检查筛选条件。")
        return
    long_df = pd.concat(all_dims_df_list, ignore_index=True)
    long_df.dropna(subset=['Category'], inplace=True)
    long_df['Average_Risk_per_Species'].fillna(0, inplace=True)
    class_order = df_plot.groupby('class')['Weighted_Threat_Risk'].mean().sort_values(ascending=True).index
    long_df['class'] = pd.Categorical(long_df['class'], categories=class_order, ordered=True)
    long_df = long_df.sort_values('class')

    print(f"\n--- [步骤 5/5] 数据导出与绘图... ---")
    csv_output_path = os.path.join(PLOT_OUTPUT_DIR, "plot_data_source.csv")
    try:
        long_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
        print(f"   - [成功] 已将绘图所用数据导出至: {csv_output_path}")
    except Exception as e:
        print(f"   - [错误] 导出CSV文件失败: {e}. (提示: 是否在Excel中打开了此文件?)")
    generate_final_bubble_plot(long_df.copy(), PLOT_OUTPUT_DIR)


if __name__ == "__main__":
    run_final_analysis_v38()
    print("\n===================================================================")
    print("  最终版高级分析已全部执行完毕！")
    print(f"  所有新的图表已保存至: {os.path.abspath(PLOT_OUTPUT_DIR)}")
    print("===================================================================")