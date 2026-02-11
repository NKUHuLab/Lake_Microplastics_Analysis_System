# -*- coding: utf-8 -*-
"""
统一数据分析与报告生成脚本 (最终平衡版)

本脚本旨在输出核心的统计摘要、关键数值和总结性表格，
在信息详尽度与可读性之间取得平衡，以方便在报告或论文中直接引用。

版本: 7.0
- 分析六增强：根据用户需求，在丰度变化分析中引入'ori'列，计算并增加了“丰度变化百分比”(change_percentage)列。
- 分析六新增：增加了对特定区域（印度、东南亚）的平均变化和平均变化百分比的专门提取和展示。
- 恢复精简版的设计思路，主要输出统计摘要表。
- 保留并整合了所有历史修正：
  - 分析一：类别统计基于“数据点”数量。
  - 分析五：所有外部数据的目标列统一为'P'。
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import spearmanr  # 用于计算相关性
import os
import warnings

# --- 0. 全局设置 ---
warnings.filterwarnings("ignore", category=UserWarning, module='geopandas')
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 150)  # 增加宽度以容纳新列

# --- 1. 全局文件路径配置 ---
PROJECT_ROOT = r"E:\lake-MP-W"

# 相关文件路径
SHAPEFILE_PATH_ALL8 = os.path.join(PROJECT_ROOT, "data", "all8.shp")
CSV_PATH_XCHANGE = os.path.join(PROJECT_ROOT, "data", "opt", "data", "processed_output", "Xchange.csv")
CSV_PATH_YCHANGE = os.path.join(PROJECT_ROOT, "data", "opt", "data", "processed_output", "Ychange.csv")
CSV_PATH_LAKE_AREA = os.path.join(PROJECT_ROOT, "data", "opt", "data", "processed_output", "1.csv")
BASE_LAKES_SHP = os.path.join(PROJECT_ROOT, "data", "base_shp", "predicted_lakes_2022.shp")

# 修正：将所有Excel文件的目标列统一为 'P'
EXCEL_FILES_TO_ANALYZE = [
    {'path': os.path.join(PROJECT_ROOT, "data", "yueshu", "CL.xlsx"), 'target_col': 'P'},
    {'path': os.path.join(PROJECT_ROOT, "data", "yueshu", "WWTP.xlsx"), 'target_col': 'P'},
    {'path': os.path.join(PROJECT_ROOT, "data", "yueshu", "Fish.xlsx"), 'target_col': 'P'}
]

# 输出文件
OUTPUT_TXT_PATH = "优化讨论_最终平衡版_v7.txt"


# --- 2. 辅助函数 ---
def write_section_header(f, title, level=1):
    if level == 1:
        f.write("\n\n" + "=" * 100 + "\n")
        f.write(f"|| {title.upper()} ||\n")
        f.write("=" * 100 + "\n\n")
    else:
        f.write("\n\n" + "-" * 80 + "\n")
        f.write(f"--- {title} ---\n" + "-" * 80 + "\n\n")


def check_file_exists(path, f):
    if not os.path.exists(path):
        f.write(f"!!! 错误：无法找到文件 '{path}'。该部分的分析将被跳过。\n")
        print(f"错误：找不到文件 '{path}'")
        return False
    return True


# --- 3. 分析函数模块 (最终平衡版) ---

def analyze_geospatial_categorization(f, shapefile_path, csv_path):
    write_section_header(f, "分析一：基于驱动因素的湖泊空间聚合与分类")
    if not check_file_exists(shapefile_path, f) or not check_file_exists(csv_path, f): return

    gdf_polygons = gpd.read_file(shapefile_path)
    df_points = pd.read_csv(csv_path)

    geometry = [Point(xy) for xy in zip(df_points['lon'], df_points['lat'])]
    gdf_points = gpd.GeoDataFrame(df_points, geometry=geometry, crs=gdf_polygons.crs)
    joined_gdf = gpd.sjoin(gdf_points, gdf_polygons, how="inner", predicate="intersects")
    columns_to_aggregate = ['Advanced_Waste_Discharge', 'fish_gdp_sqkm', 'Cultivated_land']
    summed_data = joined_gdf.groupby(joined_gdf.index_right)[columns_to_aggregate].sum()

    write_section_header(f, "1.1 空间聚合结果的统计摘要", level=2)
    f.write("对每个面要素聚合后的各关键字段，其数值分布的统计摘要如下：\n")
    f.write(summed_data.describe().to_string())

    gdf_final = gdf_polygons.join(summed_data).fillna(0)

    def get_awd_level(v):
        return 0 if v <= 0 else 1 if v <= 5000 else 2 if v <= 20000 else 3 if v <= 100000 else 4

    def get_fish_level(v):
        return 0 if v >= 0 else 1 if v > -1000 else 2 if v > -10000 else 3 if v > -100000 else 4

    def get_cult_level(v):
        return 0 if v >= 0 else 1 if v > -5 else 2 if v > -10 else 3 if v > -20 else 4

    gdf_final['AWD_Level'] = gdf_final['Advanced_Waste_Discharge'].apply(get_awd_level)
    gdf_final['Fish_Level'] = gdf_final['fish_gdp_sqkm'].apply(get_fish_level)
    gdf_final['Cult_Level'] = gdf_final['Cultivated_land'].apply(get_cult_level)
    gdf_final['Max_Level'] = gdf_final[['AWD_Level', 'Fish_Level', 'Cult_Level']].max(axis=1)
    awd_codes = {1: 'AWD1', 2: 'AWD2', 3: 'AWD3', 4: 'AWD4'}
    fish_codes = {1: 'FISH1', 2: 'FISH2', 3: 'FISH3', 4: 'FISH4'}
    cult_codes = {1: 'CULT1', 2: 'CULT2', 3: 'CULT3', 4: 'CULT4'}

    def create_final_label(row):
        max_level = row['Max_Level']
        if max_level == 0: return 'No_Impact'
        parts = [code.get(row[f'{factor}_Level']) for factor, code in
                 [('AWD', awd_codes), ('Fish', fish_codes), ('Cult', cult_codes)] if
                 row[f'{factor}_Level'] == max_level]
        return '_'.join(filter(None, parts))

    gdf_final['cat_name'] = gdf_final.apply(create_final_label, axis=1)

    # 修正：统计每个类别下的 “数据点” 数量
    write_section_header(f, "1.2 最终分类结果与统计 (基于数据点)", level=2)
    points_with_category = joined_gdf.merge(
        gdf_final[['cat_name']], left_on='index_right', right_index=True, how='left'
    )
    point_counts_by_category = points_with_category['cat_name'].value_counts().reset_index()
    point_counts_by_category.columns = ['Category_Name', 'Data_Point_Count']
    f.write("最终各类别下的 **数据点 (行)** 数量统计：\n")
    f.write(point_counts_by_category.to_string())


def analyze_mitigation_potential_density(f, ychange_csv_path):
    write_section_header(f, "分析二：减排潜力密度分布数据 (对应山峦图)")
    if not check_file_exists(ychange_csv_path, f): return
    df = pd.read_csv(ychange_csv_path)
    plot_data_long = df.loc[df['opt_MPs_count'] <= 0, ['opt_MPs_count', 'income', 'Region']] \
        .melt(id_vars=['opt_MPs_count'], var_name='Group', value_name='Category')
    plot_data_long['Group'] = plot_data_long['Group'].replace({'income': 'Income Level'})
    write_section_header(f, "2.1 各分组样本量 (n)", level=2)
    sample_counts = plot_data_long.groupby(['Group', 'Category']).size().reset_index(name='n')
    f.write(sample_counts.to_string())
    write_section_header(f, "2.2 各分组详细描述性统计", level=2)
    stats_original = plot_data_long.groupby(['Group', 'Category'])['opt_MPs_count'].describe(percentiles=[.25, .5, .75])
    f.write(stats_original.to_string())


def analyze_lake_area_change(f, lake_area_csv_path):
    write_section_header(f, "分析三：不同面积湖泊的平均变化率分析 (对应宽度可变柱状图)")
    if not check_file_exists(lake_area_csv_path, f): return
    df = pd.read_csv(lake_area_csv_path, usecols=[0, 1], names=["Lake_area", "change_percent"], header=0)
    df['Lake_area'] = pd.to_numeric(df['Lake_area'].astype(str).str.replace(",", ""), errors='coerce')
    df['change_percent'] = pd.to_numeric(df['change_percent'].astype(str).str.replace("%", ""), errors='coerce')
    df.dropna(inplace=True)
    write_section_header(f, "3.1 分组统计摘要表", level=2)
    breaks = [0, 0.1, 1, 10, 100, 1000, 10000, np.inf]
    labels = ["< 0.1", "0.1-1", "1-10", "10-100", "100-1000", "1000-10000", "> 10000"]
    df['Area_Group'] = pd.cut(df['Lake_area'], bins=breaks, labels=labels, right=False)
    summary_stats = df.groupby('Area_Group').agg(
        mean_change=('change_percent', 'mean'), sd_change=('change_percent', 'std'), n=('change_percent', 'size')
    ).reset_index()
    summary_stats['se'] = summary_stats['sd_change'] / np.sqrt(summary_stats['n'])
    summary_stats['ci_lower'] = summary_stats['mean_change'] - 1.96 * summary_stats['se']
    summary_stats['ci_upper'] = summary_stats['mean_change'] + 1.96 * summary_stats['se']
    f.write(summary_stats.to_string())


def analyze_driver_relationships(f, xchange_csv_path):
    write_section_header(f, "分析四：驱动因素与生态变化关系分析 (对应三合一折线图)")
    f.write("此部分通过计算斯皮尔曼等级相关系数(Spearman's Rho)来总结驱动因素与生态变化指标之间的单调关系。\n")
    if not check_file_exists(xchange_csv_path, f): return
    main_df = pd.read_csv(xchange_csv_path)
    main_df['abs_change_clipped'] = main_df['change'].abs().clip(upper=main_df['change'].abs().quantile(0.998))
    plot_configs = [
        {'x_col': 'fish_gdp_sqkm', 'label': '渔业产值', 'use_log': True},
        {'x_col': 'Advanced_Waste_Discharge', 'label': '高级废水排放', 'use_log': True},
        {'x_col': 'Cultivated_land', 'label': '耕地', 'use_log': False}
    ]
    for config in plot_configs:
        x_col = config['x_col']
        write_section_header(f, f"4.{plot_configs.index(config) + 1} 驱动因素: {config['label']} ({x_col})", level=2)
        plot_data = main_df.dropna(subset=[x_col, 'change', 'income', 'Region']).copy()
        plot_data['x_transformed'] = np.log1p(np.abs(plot_data[x_col])) if config['use_log'] else np.abs(
            plot_data[x_col])
        f.write("\n--- 按 Region 分组的相关性分析 ---\n")
        corr_region_data = []
        for region in plot_data['Region'].unique():
            subset = plot_data[plot_data['Region'] == region]
            if len(subset) > 1:
                rho, pval = spearmanr(subset['x_transformed'], subset['abs_change_clipped'])
                corr_region_data.append(
                    {'Region': region, 'Spearman_Rho': rho, 'P_Value': pval, 'N_Samples': len(subset)})
        f.write(pd.DataFrame(corr_region_data).to_string(index=False))
        f.write("\n\n--- 按 Income Level 分组的相关性分析 ---\n")
        corr_income_data = []
        for income in plot_data['income'].unique():
            subset = plot_data[plot_data['income'] == income]
            if len(subset) > 1:
                rho, pval = spearmanr(subset['x_transformed'], subset['abs_change_clipped'])
                corr_income_data.append(
                    {'Income Level': income, 'Spearman_Rho': rho, 'P_Value': pval, 'N_Samples': len(subset)})
        f.write(pd.DataFrame(corr_income_data).to_string(index=False))


def analyze_spatial_join_summary(f, excel_files_info, base_lakes_shp_path):
    write_section_header(f, "分析五：外部数据空间匹配与分布特征")
    if not check_file_exists(base_lakes_shp_path, f): return
    gdf_lakes = gpd.read_file(base_lakes_shp_path)
    total_lakes = len(gdf_lakes)
    for i, file_info in enumerate(excel_files_info):
        excel_path = file_info['path']
        target_col = file_info['target_col']
        write_section_header(f, f"5.{i + 1} 数据集: {os.path.basename(excel_path)}", level=2)
        if not check_file_exists(excel_path, f): continue
        df_excel = pd.read_excel(excel_path)
        if not all(c in df_excel.columns for c in ['lon', 'lat', target_col]):
            f.write(f"!!! 错误: 文件缺少 'lon', 'lat' 或目标列 '{target_col}'。\n")
            continue
        df_excel.dropna(subset=['lon', 'lat', target_col], inplace=True)
        gdf_points = gpd.GeoDataFrame(df_excel, geometry=gpd.points_from_xy(df_excel.lon, df_excel.lat),
                                      crs=gdf_lakes.crs)
        merged_gdf = gpd.sjoin(gdf_lakes, gdf_points, how="inner", predicate="contains")
        write_section_header(f, f"5.{i + 1}.1 空间匹配效率分析", level=2)
        unique_matched_lakes = merged_gdf.index.nunique()
        f.write(f"  - 原始有效数据点总数: {len(gdf_points)}\n")
        f.write(f"  - 基础图层湖泊总数: {total_lakes}\n")
        f.write(f"  - 成功接收到至少一个数据点的湖泊数: {unique_matched_lakes}\n")
        f.write(f"  - 湖泊覆盖率: {unique_matched_lakes / total_lakes:.2%}\n")
        aggregation_counts = merged_gdf.groupby(merged_gdf.index).size()
        lakes_requiring_aggregation = (aggregation_counts > 1).sum()
        f.write(f"  - 需要聚合（一个湖泊接收到多个点）的湖泊数: {lakes_requiring_aggregation}\n")
        final_lake_data = merged_gdf.groupby(merged_gdf.index)[target_col].mean()
        write_section_header(f, f"5.{i + 1}.2 目标列 '{target_col}' 在匹配成功湖泊中的统计分布摘要 (聚合后)", level=2)
        f.write(final_lake_data.describe().to_string())


# --- 分析模块 (增强版) ---
def analyze_abundance_change(f, xchange_csv_path):
    """
    增强版分析模块：分析丰度变化，并计算其绝对值和百分比，同时提供特定区域的摘要。
    """
    write_section_header(f, "分析六：丰度变化与百分比分析 (Abundance Change & Percentage Analysis)")
    f.write("本部分分析 'Xchange.csv' 中的 'change' 和 'ori' 列，分别代表丰度变化量和原始丰度值。\n")
    f.write("负均值表示丰度平均下降，正均值表示平均上升。\n")
    if not check_file_exists(xchange_csv_path, f): return

    df = pd.read_csv(xchange_csv_path)

    # 确保核心列存在
    if not all(col in df.columns for col in ['change', 'ori', 'Region', 'income']):
        f.write("!!! 错误: 'Xchange.csv' 文件缺少 'change', 'ori', 'Region', 或 'income' 列。\n")
        return

    # --- 6.1 按 Region 分组的丰度变化分析 ---
    write_section_header(f, "6.1 按 Region 分组的丰度变化统计摘要", level=2)

    # 同时聚合 change 和 ori 列
    region_summary = df.groupby('Region').agg(
        mean_change=('change', 'mean'),
        mean_ori=('ori', 'mean'),
        sd_change=('change', 'std'),
        n=('change', 'size')
    ).reset_index()

    # 计算变化百分比，并处理分母为0的情况
    region_summary['change_percentage'] = np.where(
        region_summary['mean_ori'] != 0,
        (region_summary['mean_change'] / region_summary['mean_ori']) * 100,
        np.nan  # 如果原始值为0，则百分比无意义，记为NaN
    )

    # 计算置信区间
    region_summary['se'] = region_summary['sd_change'] / np.sqrt(region_summary['n'])
    region_summary['ci_lower'] = region_summary['mean_change'] - 1.96 * region_summary['se']
    region_summary['ci_upper'] = region_summary['mean_change'] + 1.96 * region_summary['se']

    f.write("该表显示了不同区域的平均丰度变化、平均原始丰度、变化百分比(%)及95%置信区间。\n")
    # 为了更好的可读性，格式化输出列的顺序和浮点数精度
    f.write(region_summary[[
        'Region', 'mean_change', 'mean_ori', 'change_percentage',
        'sd_change', 'n', 'ci_lower', 'ci_upper'
    ]].to_string(float_format="%.2f"))

    # --- 6.2 按 Income Level 分组的丰度变化分析 ---
    write_section_header(f, "6.2 按 Income Level 分组的丰度变化统计摘要", level=2)

    income_summary = df.groupby('income').agg(
        mean_change=('change', 'mean'),
        mean_ori=('ori', 'mean'),
        sd_change=('change', 'std'),
        n=('change', 'size')
    ).reset_index()

    income_summary['change_percentage'] = np.where(
        income_summary['mean_ori'] != 0,
        (income_summary['mean_change'] / income_summary['mean_ori']) * 100,
        np.nan
    )

    income_summary['se'] = income_summary['sd_change'] / np.sqrt(income_summary['n'])
    income_summary['ci_lower'] = income_summary['mean_change'] - 1.96 * income_summary['se']
    income_summary['ci_upper'] = income_summary['mean_change'] + 1.96 * income_summary['se']

    f.write("该表显示了不同收入水平的平均丰度变化、平均原始丰度、变化百分比(%)及95%置信区间。\n")
    f.write(income_summary[[
        'income', 'mean_change', 'mean_ori', 'change_percentage',
        'sd_change', 'n', 'ci_lower', 'ci_upper'
    ]].to_string(float_format="%.2f"))

    # --- 6.3 特定区域分析 (印度与东南亚) ---
    write_section_header(f, "6.3 特定区域分析 (印度与东南亚)", level=2)
    target_regions = ['India']
    specific_summary = region_summary[region_summary['Region'].isin(target_regions)]

    f.write("根据您的要求，以下是印度和东南亚的平均变化与平均变化百分比的摘要：\n")
    if not specific_summary.empty:
        f.write(specific_summary[[
            'Region', 'mean_change', 'mean_ori', 'change_percentage', 'n'
        ]].to_string(index=False, float_format="%.2f"))
    else:
        f.write("数据中未找到 'India' 或 'Southeast Asia' 的区域信息。\n")


# --- 4. 主执行函数 ---
def main():
    try:
        with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f:
            f.write("=" * 35 + " 综合数据分析报告 (最终平衡版 v7.0) " + "=" * 35 + "\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("本文档汇总了多个分析脚本的核心统计结果，旨在为讨论提供简洁、关键的数据支持。\n")
            # 依次执行各个分析模块
            analyze_geospatial_categorization(f, SHAPEFILE_PATH_ALL8, CSV_PATH_XCHANGE)
            analyze_mitigation_potential_density(f, CSV_PATH_YCHANGE)
            analyze_lake_area_change(f, CSV_PATH_LAKE_AREA)
            analyze_driver_relationships(f, CSV_PATH_XCHANGE)
            analyze_spatial_join_summary(f, EXCEL_FILES_TO_ANALYZE, BASE_LAKES_SHP)
            # --- 调用增强版的分析模块 ---
            analyze_abundance_change(f, CSV_PATH_XCHANGE)

            f.write("\n\n" + "=" * 45 + " 报告结束 " + "=" * 45 + "\n")
        print(f"\n[SUCCESS] 所有分析数据已成功写入到: '{OUTPUT_TXT_PATH}'")
    except Exception as e:
        print(f"\n[ERROR] 生成报告时发生严重错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()