import os
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 全局路径和参数配置 ---
PROJECT_ROOT_ANI = r"E:\lake-MP-ani"
PROJECT_ROOT_W = r"E:\lake-MP-W"
PLOT_OUTPUT_DIR = r"E:\lake-MP-W\draw\IUCN_Advanced2"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

DATA_SUMMARY_DIR = os.path.join(PROJECT_ROOT_ANI, "draw", "111Biological_Attribution_Analysis_SJoin_Fast",
                                "data_summaries")
INTERMEDIATE_SJOIN_GPKG = os.path.join(DATA_SUMMARY_DIR, "intermediate_lake_species_intersections.gpkg")
POLLUTION_HULLS_GPKG = os.path.join(PROJECT_ROOT_W, "draw", "11_Geographic_Clustering", "cluster_hulls_WGS84.gpkg")

THREAT_WEIGHTS = {'Critically Endangered': 50, 'Endangered': 25, 'Vulnerable': 10, 'Near Threatened': 5}

# --- [新增] CSV输出路径 ---
CSV_OUTPUT_PATH = os.path.join(PLOT_OUTPUT_DIR, "barchart_data_source1.csv")


def robust_bool_converter(series):
    map_dict = {'True': True, 'False': False, 'true': True, 'false': False,
                'TRUE': True, 'FALSE': False, 1: True, 0: False,
                1.0: True, 0.0: False, True: True, False: False}
    return series.map(map_dict).fillna(False)


# --- 2. 全局绘图设置 ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="white", font="Arial")
warnings.filterwarnings("ignore")


# --- 3. V9 核心可视化函数 (集成总体平均) ---
def generate_integrated_barchart(plot_df, output_dir):
    """
    V9核心可视化函数: 绘制包含“总体平均”作为第一组的集成式图表。
    """
    print(f"\n--- [绘图核心] 生成集成式风险降低柱状图 (V9)... ---")
    try:
        dim_order = ['Overall Summary', 'Lifecycle Group', 'Habitat Dependency', 'Origin Label', 'Risk Zone']

        plot_df['Analysis_Dimension'] = pd.Categorical(plot_df['Analysis_Dimension'], categories=dim_order,
                                                       ordered=True)
        plot_df = plot_df.sort_values(
            by=['Analysis_Dimension', 'Reduction_Percentage'],
            ascending=[True, False]
        ).reset_index(drop=True)

        baseline_color = '#081d58'
        optimized_color = '#3cb1c3'
        reduction_arrow_color = '#c44e52'

        fig, ax = plt.subplots(figsize=(30, 15))

        ax.bar(plot_df.index, plot_df['Avg_Weighted_Baseline'], color=baseline_color,
               label='Avg. Weighted Baseline Risk', width=0.5, zorder=2)
        ax.bar(plot_df.index, plot_df['Avg_Weighted_Optimized'], color=optimized_color,
               label='Avg. Weighted Optimized Risk', width=0.25, zorder=3)

        for i, row in plot_df.iterrows():
            baseline_h = row['Avg_Weighted_Baseline']
            optimized_h = row['Avg_Weighted_Optimized']
            text_y = baseline_h * 1.05
            if baseline_h > 0:
                ax.annotate(
                    '', xy=(i, optimized_h), xytext=(i, baseline_h),
                    arrowprops=dict(arrowstyle="->", color=reduction_arrow_color, lw=1.5), zorder=4
                )
                ax.text(i, text_y, f'↓{row["Reduction_Percentage"]:.1f}%',
                        fontsize=10, color=reduction_arrow_color, ha='center', va='bottom', weight='bold', zorder=5)

        ax.set_xticks(plot_df.index)
        ax.set_xticklabels(plot_df['Category'], rotation=45, ha='right', fontsize=14)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('grey'); ax.spines['bottom'].set_color('grey')

        dimension_boundaries = plot_df['Analysis_Dimension'].ne(plot_df['Analysis_Dimension'].shift())
        boundary_indices = dimension_boundaries[dimension_boundaries].index
        for idx in boundary_indices[1:]:
            ax.axvline(x=idx - 0.5, color='grey', linestyle='--', linewidth=1.5)

        for dim in dim_order:
            subset = plot_df[plot_df['Analysis_Dimension'] == dim]
            if not subset.empty:
                indices = subset.index
                label_pos = (indices.min() + indices.max()) / 2.0
                ax.text(label_pos, ax.get_ylim()[1] * 1.01, dim, ha='center', va='bottom', fontsize=18, weight='bold')

        ax.legend(fontsize=14, frameon=False, loc='upper right')
        ax.set_ylabel("Average Weighted Risk Score (Threatened Species Only)", fontsize=16)
        ax.set_xlabel("Category", fontsize=16)
        fig.suptitle('Integrated Analysis of Average Weighted Risk Reduction', fontsize=28, weight='bold')
        plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.94])
        output_path = os.path.join(output_dir, "Risk_Reduction_Analysis_Integrated.pdf")
        plt.savefig(output_path, format='pdf')
        plt.close('all')
        print(f"   - [成功] 已保存集成式图表至: {output_path}")

    except Exception as e:
        print(f"   - [错误] 绘制集成式图表失败: {e}")


# --- 4. 主分析流程 ---
def run_integrated_analysis_and_plot():
    """执行所有数据处理，并将总体平均数据合并到主数据集中进行绘图。"""
    print("--- [初始化] 集成式分析流程 ---")
    try:
        gdf = gpd.read_file(INTERMEDIATE_SJOIN_GPKG, engine='pyogrio')
        gdf_hulls = gpd.read_file(POLLUTION_HULLS_GPKG, engine='pyogrio')
    except Exception as e:
        print(f"[严重错误] 加载核心文件失败: {e}");
        return

    print("--- [步骤 1/3] 数据预处理与加权风险计算... ---")
    for col in ['freshwater', 'marine', 'terrestria']:
        if col in gdf.columns:
            gdf[col] = robust_bool_converter(gdf[col])
        else:
            gdf[col] = False
    for col in ['IRS_baseline', 'IRS_optimized']:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce').fillna(0)
    gdf['Threat_Score'] = gdf['redlist_category'].map(THREAT_WEIGHTS).fillna(0)
    gdf['Weighted_Baseline'] = gdf['IRS_baseline'].clip(lower=0) * gdf['Threat_Score']
    gdf['Weighted_Optimized'] = gdf['IRS_optimized'].clip(lower=0) * gdf['Threat_Score']

    conditions = [
        (gdf['freshwater']) & (gdf['marine']) & (gdf['terrestria']),
        (gdf['freshwater']) & (gdf['marine']) & (~gdf['terrestria']),
        (gdf['freshwater']) & (~gdf['marine']) & (gdf['terrestria']),
        (~gdf['freshwater']) & (gdf['marine']) & (gdf['terrestria']),
        (gdf['freshwater']) & (~gdf['marine']) & (~gdf['terrestria']),
        (~gdf['freshwater']) & (gdf['marine']) & (~gdf['terrestria']),
        (~gdf['freshwater']) & (~gdf['marine']) & (gdf['terrestria'])
    ]
    choices = ['FMT-Generalist', 'Freshwater&Marine', 'Freshwater&Terrestrial', 'Marine&Terrestrial', 'Freshwater-Only',
               'Marine-Only', 'Terrestrial-Only']
    gdf['habitat_dependency'] = np.select(conditions, choices, default='Undefined')
    gdf['lifecycle_group'] = np.where(gdf['seasonal'] == 2, 'Breeding Area', 'Other Areas')
    gdf['origin_label'] = gdf['origin'].map({1: 'Native', 2: 'Reintroduced', 3: 'Introduced'}).fillna('Unknown')
    gdf = gpd.sjoin(gdf.to_crs(gdf_hulls.crs), gdf_hulls[['cluster_id', 'geometry']], how='left',
                    predicate='intersects')
    gdf['risk_zone'] = gdf['cluster_id'].map(
        {16: 'a', 0: 'b', 12: 'b', 7: 'c', 11: 'd', 17: 'd', 4: 'e', 13: 'f', 18: 'f', 9: 'g', 14: 'h'}).fillna(
        'Outside Zones')

    df_plot_base = gdf[~gdf['habitat_dependency'].isin(['Undefined', 'Terrestrial-Only'])].copy()
    df_plot_base = df_plot_base[df_plot_base['Threat_Score'] > 0].copy()

    if df_plot_base.empty:
        print("[严重错误] 筛选后没有受威胁物种数据可供分析。");
        return

    print(f"--- [步骤 2/4] 聚合数据用于绘图... ---")
    dims_to_plot = ['lifecycle_group', 'habitat_dependency', 'origin_label', 'risk_zone']
    all_dims_df_list = []
    for dim in dims_to_plot:
        agg_df = df_plot_base.groupby(dim).agg(
            Avg_Weighted_Baseline=('Weighted_Baseline', 'mean'),
            Avg_Weighted_Optimized=('Weighted_Optimized', 'mean')
        ).reset_index()
        agg_df = agg_df[agg_df['Avg_Weighted_Baseline'] > 0]
        reduction = agg_df['Avg_Weighted_Baseline'] - agg_df['Avg_Weighted_Optimized']
        agg_df['Reduction_Percentage'] = (reduction / (agg_df['Avg_Weighted_Baseline'] + 1e-9)) * 100
        agg_df['Reduction_Percentage'] = agg_df['Reduction_Percentage'].clip(upper=100)
        agg_df['Analysis_Dimension'] = dim.replace('_', ' ').title()
        agg_df.rename(columns={dim: 'Category'}, inplace=True)
        all_dims_df_list.append(agg_df)
    detailed_df = pd.concat(all_dims_df_list, ignore_index=True).dropna(subset=['Category'])

    overall_avg_baseline = df_plot_base['Weighted_Baseline'].mean()
    overall_avg_optimized = df_plot_base['Weighted_Optimized'].mean()
    reduction_percentage = ((overall_avg_baseline - overall_avg_optimized) / (overall_avg_baseline + 1e-9)) * 100
    summary_df = pd.DataFrame({
        'Category': ['Overall Average'],
        'Avg_Weighted_Baseline': [overall_avg_baseline],
        'Avg_Weighted_Optimized': [overall_avg_optimized],
        'Reduction_Percentage': [reduction_percentage]
    })
    summary_df['Analysis_Dimension'] = 'Overall Summary'
    final_plot_df = pd.concat([detailed_df, summary_df], ignore_index=True)

    # --- [新增] 步骤 3/4: 将绘图数据源保存到CSV文件 ---
    print(f"\n--- [步骤 3/4] 将绘图数据源保存到CSV文件... ---")
    try:
        # 为了在CSV中有更好的可读性，重新排列一下列的顺序
        output_columns = [
            'Analysis_Dimension', 'Category', 'Avg_Weighted_Baseline',
            'Avg_Weighted_Optimized', 'Reduction_Percentage'
        ]
        final_plot_df.to_csv(CSV_OUTPUT_PATH, index=False, columns=output_columns, encoding='utf-8-sig')
        print(f"   - [成功] 已将条形图数据源保存至: {CSV_OUTPUT_PATH}")
    except Exception as e:
        print(f"   - [错误] 保存CSV文件失败: {e}")

    # --- 步骤 4/4: 开始最终绘图 ---
    print(f"\n--- [步骤 4/4] 开始最终绘图... ---")
    generate_integrated_barchart(final_plot_df, PLOT_OUTPUT_DIR)


if __name__ == "__main__":
    run_integrated_analysis_and_plot()

    print("\n===================================================================")
    print("  集成式分析已全部执行完毕！")
    print(f"  图表已保存至: {os.path.abspath(PLOT_OUTPUT_DIR)}")
    print(f"  数据源CSV已保存至: {os.path.abspath(CSV_OUTPUT_PATH)}")
    print("===================================================================")