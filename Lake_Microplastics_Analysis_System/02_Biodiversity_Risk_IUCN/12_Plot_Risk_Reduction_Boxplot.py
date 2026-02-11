# -*- coding: utf-8 -*-
import os
import sys
import warnings
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main_statistical_analysis_with_twrs():
    """
    主函数 (思路B版本):
    1. [计算核心] 使用思路B的逻辑，基于IRS数据为每个湖泊计算总加权风险降低分数 (TWRS)。
    2. [空间链接] 将湖泊的TWRS分数与地理分区、地区(Region)、收入(Income)信息进行空间连接。
    3. [输出数据] 将用于生成条形图的聚合后摘要数据保存为CSV文件。
    4. [输出图表] 使用TWRS数据绘制分区箱形图以及地区/收入的汇总条形图。
    """
    print("--- [初始化] 基于思路B (TWRS) 的空间统计分析 ---")

    # --- 1. 路径和常量定义 ---
    INTERMEDIATE_SJOIN_GPKG = r"E:\lake-MP-ani\draw\111Biological_Attribution_Analysis_SJoin_Fast\data_summaries\intermediate_lake_species_intersections.gpkg"
    PARTITION_CSV = r"E:\lake-MP-W\draw\11_Geographic_Clustering\ALL_points_annotated_with_cluster_id.csv"
    MP_LAKES_SHP_PATH = r"E:\lake-MP-W\data\generated_shp\predicted_lakes_2022.shp"

    OUTPUT_DIR = r"E:\lake-MP-W\draw\03_SHAP_Analysis\Statistical_Analysis_with_TWRS"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- [核心修改] 修改输出文件路径，不再输出详细数据，改为输出两个摘要数据 ---
    OUTPUT_INCOME_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_by_income.csv")
    OUTPUT_REGION_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_by_region.csv")
    OUTPUT_PLOT_PDF = os.path.join(OUTPUT_DIR, "partition_twrs_statistical_plots.pdf")

    IUCN_WEIGHTS = {'Critically Endangered': 50, 'Endangered': 25, 'Vulnerable': 10, 'Near Threatened': 5}
    LAKE_ID_COLUMN_GPKG = 'hylak_id'
    LAKE_ID_COLUMN_SHP = 'FID'
    partition_map = {16: 'a', 0: 'b', 12: 'b', 7: 'c', 11: 'd', 17: 'd', 4: 'e', 13: 'f', 9: 'g'}
    partition_order = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'Unpartitioned']

    # --- 2. 全局绘图设置 ---
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    sns.set_theme(style="whitegrid", font="Arial")
    warnings.filterwarnings("ignore")

    # --- 步骤 1: 计算每个湖泊的 TWRS (思路B核心逻辑) ---
    print("\n--- [步骤 1/5] 正在为每个湖泊计算 TWRS (总加权风险降低分数)... ---")
    try:
        df_irs = gpd.read_file(INTERMEDIATE_SJOIN_GPKG,
                               columns=[LAKE_ID_COLUMN_GPKG, 'redlist_category', 'IRS_baseline', 'IRS_optimized'],
                               engine='pyogrio')
        df_irs['Threat_Score'] = df_irs['redlist_category'].map(IUCN_WEIGHTS).fillna(0)
        df_irs_filtered = df_irs[df_irs['Threat_Score'] > 0].copy()
        df_irs_filtered['Reduction'] = (pd.to_numeric(df_irs_filtered['IRS_baseline'], errors='coerce').fillna(0) -
                                        pd.to_numeric(df_irs_filtered['IRS_optimized'], errors='coerce').fillna(
                                            0)).clip(lower=0)
        df_irs_filtered['Weighted_Reduction'] = df_irs_filtered['Reduction'] * df_irs_filtered['Threat_Score']

        df_lake_scores = df_irs_filtered.groupby(LAKE_ID_COLUMN_GPKG)[['Weighted_Reduction']].sum().reset_index()
        df_lake_scores.rename(columns={'Weighted_Reduction': 'TWRS'}, inplace=True)
        print(f"   - 成功为 {len(df_lake_scores)} 个湖泊计算了 TWRS 分数。")
    except Exception as e:
        print(f"❌ 致命错误: 在步骤1计算TWRS时失败: {e}")
        sys.exit()

    # --- 步骤 2: 准备空间数据 (湖泊 和 分区点) ---
    print("\n--- [步骤 2/5] 准备空间数据以供连接... ---")
    try:
        gdf_lakes = gpd.read_file(MP_LAKES_SHP_PATH, engine="pyogrio")
        gdf_lakes.rename(columns={LAKE_ID_COLUMN_SHP: LAKE_ID_COLUMN_GPKG}, inplace=True)
        gdf_lakes_with_twrs = gdf_lakes[[LAKE_ID_COLUMN_GPKG, 'geometry']].merge(df_lake_scores, on=LAKE_ID_COLUMN_GPKG,
                                                                                 how='left')
        gdf_lakes_with_twrs['TWRS'] = gdf_lakes_with_twrs['TWRS'].fillna(0)
        if gdf_lakes_with_twrs.crs is None:
            gdf_lakes_with_twrs.set_crs("EPSG:4326", inplace=True)

        df_partitions_csv = pd.read_csv(PARTITION_CSV, low_memory=False)
        df_partitions_csv['Partition'] = df_partitions_csv['hull_cluster_id'].map(partition_map).fillna('Unpartitioned')
        gdf_partition_points = gpd.GeoDataFrame(
            df_partitions_csv, geometry=gpd.points_from_xy(df_partitions_csv.lon, df_partitions_csv.lat),
            crs="EPSG:4326"
        )

        cols_to_keep = ['Partition', 'geometry']
        HAS_EXTRA_COLS = True
        for col in ['Region', 'income']:
            if col not in df_partitions_csv.columns:
                print(f"⚠️ 警告: 分区文件中未找到列 '{col}'。将跳过相关分析。")
                HAS_EXTRA_COLS = False
            else:
                cols_to_keep.append(col)
        gdf_partition_lookup = gdf_partition_points[list(set(cols_to_keep))].copy()
        print("   - 湖泊多边形和分区点准备完毕。")
    except Exception as e:
        print(f"❌ 致命错误: 在步骤2准备空间数据时失败: {e}")
        sys.exit()

    # --- 步骤 3: 执行空间连接 ---
    print("\n--- [步骤 3/5] 正在执行空间连接... ---")
    gdf_final_data = gpd.sjoin(gdf_lakes_with_twrs, gdf_partition_lookup, how='left', predicate='contains')
    gdf_final_data = gdf_final_data.drop_duplicates(subset=[LAKE_ID_COLUMN_GPKG])
    gdf_final_data['Partition'].fillna('Unpartitioned', inplace=True)
    print(f"   - 空间连接完成，共计 {len(gdf_final_data)} 条湖泊记录。")

    # --- [核心修改] 步骤 4: 绘制图表, 并保存聚合后的绘图数据 ---
    print("\n--- [步骤 4/5] 正在生成统计图表并保存摘要数据... ---")
    if HAS_EXTRA_COLS:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 27))
        ax1, ax2, ax3 = axes.flatten()
    else:
        fig, ax1 = plt.subplots(figsize=(16, 9))
        ax2, ax3 = None, None

    # 图1: 分区箱形图 (此图的源数据是详细数据，根据要求不保存)
    print("   - 绘制箱形图 (Partition)...")
    sns.boxplot(data=gdf_final_data, x='Partition', y='TWRS', order=partition_order, palette='YlGnBu', ax=ax1,
                showfliers=False)
    ax1.set_yscale('linear')
    ax1.set_title('Distribution of TWRS by Partition (Main Distribution Only)', fontsize=20, weight='bold', pad=20)
    ax1.set_ylabel('TWRS (Linear Scale - Outliers Not Shown)', fontsize=16, labelpad=15)
    ax1.set_xlabel('Geographic Partition', fontsize=16, labelpad=15)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=12)

    if HAS_EXTRA_COLS:
        df_plot_agg = gdf_final_data[gdf_final_data['TWRS'] > 0].copy()

        # --- 图2: 收入汇总条形图 & 保存其数据 ---
        print("   - 聚合并绘制条形图 (Income)...")
        income_agg = df_plot_agg.groupby('income')['TWRS'].sum().sort_values(ascending=False).reset_index()

        # 保存摘要数据
        try:
            income_agg.to_csv(OUTPUT_INCOME_SUMMARY_CSV, index=False)
            print(f"     ✅ 按收入汇总的数据已保存至: {OUTPUT_INCOME_SUMMARY_CSV}")
        except Exception as e:
            print(f"     ❌ 保存按收入汇总的CSV时出错: {e}")

        sns.barplot(data=income_agg, x='TWRS', y='income', ax=ax2, palette='Blues_r', orient='h')
        ax2.set_title('Total TWRS by Income Group', fontsize=20, weight='bold', pad=20)
        ax2.set_xlabel('Total TWRS (Sum of Positive Reductions)', fontsize=16, labelpad=15)
        ax2.set_ylabel('Income Group', fontsize=16)

        # --- 图3: 地区汇总条形图 & 保存其数据 ---
        print("   - 聚合并绘制条形图 (Region)...")
        region_agg = df_plot_agg.groupby('Region')['TWRS'].sum().sort_values(ascending=False).reset_index()

        # 保存摘要数据
        try:
            region_agg.to_csv(OUTPUT_REGION_SUMMARY_CSV, index=False)
            print(f"     ✅ 按地区汇总的数据已保存至: {OUTPUT_REGION_SUMMARY_CSV}")
        except Exception as e:
            print(f"     ❌ 保存按地区汇总的CSV时出错: {e}")

        sns.barplot(data=region_agg, x='TWRS', y='Region', ax=ax3, palette='Greens_r', orient='h')
        ax3.set_title('Total TWRS by World Bank Region', fontsize=20, weight='bold', pad=20)
        ax3.set_xlabel('Total TWRS (Sum of Positive Reductions)', fontsize=16, labelpad=15)
        ax3.set_ylabel('Region', fontsize=16)

    # --- 步骤 5: 保存最终图表 ---
    print("\n--- [步骤 5/5] 正在保存最终的PDF图表... ---")
    plt.tight_layout()
    try:
        plt.savefig(OUTPUT_PLOT_PDF, format='pdf', bbox_inches='tight')
        print(f"✅ 图表输出成功! 已保存至: {OUTPUT_PLOT_PDF}")
    except Exception as e:
        print(f"❌ 保存PDF文件时出错: {e}")
    plt.close(fig)


if __name__ == "__main__":
    main_statistical_analysis_with_twrs()