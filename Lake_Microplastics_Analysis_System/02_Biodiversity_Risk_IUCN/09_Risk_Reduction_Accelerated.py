import os
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from tqdm import tqdm

# --- 0. 全局配置与初始化 ---
print("--- [初始化-情景分析] 设置全局参数 ---")

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
warnings.filterwarnings("ignore")

# --- 1. 路径和参数配置 ---
PROJECT_ROOT_ANI = r"E:\lake-MP-ani"
PROJECT_ROOT_W = r"E:\lake-MP-W"

# --- 输入路径 ---
SPECIES_GPKG_PATH = os.path.join(PROJECT_ROOT_ANI, "data", "IUCN_consolidated", "all_species_classified.gpkg")
MP_LAKES_SHP_PATH = os.path.join(PROJECT_ROOT_W, "data", "generated_shp", "predicted_lakes_2022.shp")
WORLD_BORDERS_SHP = os.path.join(PROJECT_ROOT_W, "data", "base_shp", "world map china line.shp")
# [重要] Ychange.csv 文件路径，包含 MPs-ori 和 change
REDUCTION_POTENTIAL_CSV = os.path.join(PROJECT_ROOT_W, "data", "opt", "data", "processed_output", "Ychange.csv")

# --- 输出路径 ---
DOWNSTREAM_OUTPUT_DIR = os.path.join(PROJECT_ROOT_ANI, "draw", "Biological_Attribution_Analysis_SJoin_Fast",
                                     "data_summaries")
os.makedirs(DOWNSTREAM_OUTPUT_DIR, exist_ok=True)
# [核心产物] 输出的GPKG文件名保持不变
FINAL_INTERSECTION_GPKG = os.path.join(DOWNSTREAM_OUTPUT_DIR, "intermediate_lake_species_intersections.gpkg")

OUTPUT_DIR_DELTA = os.path.join(PROJECT_ROOT_ANI, "draw", "Risk_Reduction_Delta_Maps1")
os.makedirs(OUTPUT_DIR_DELTA, exist_ok=True)
FINAL_DELTA_METRICS_CSV = os.path.join(OUTPUT_DIR_DELTA, "lakes_risk_reduction_delta_metrics.csv")

# --- 分析参数 ---
EQUAL_AREA_PROJ = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
LAKE_ID_COLUMN = 'FID'  # 原始湖泊 SHP 文件中的 ID 列
TARGET_CATEGORIES = ['Critically Endangered', 'Endangered', 'Vulnerable', 'Near Threatened']

# --- 2. 核心计算函数定义 (BVI) ---
def calculate_bvi(row):
    """根据 seasonal 字段计算生物脆弱性指数 (BVI)。"""
    W_season = 1.0
    if 'SEASONAL' in row and pd.notna(row['SEASONAL']) and row['SEASONAL'] == 2:
        W_season = 1.5
    return W_season

# --- 3. 数据加载与预处理 ---
print("\n--- [步骤 1] 加载所有基础数据 ---")
try:
    print("-> 加载物种数据并计算 BVI...")
    gdf_species_raw = gpd.read_file(SPECIES_GPKG_PATH, engine="pyogrio")
    gdf_species = gdf_species_raw[gdf_species_raw['redlist_category'].isin(TARGET_CATEGORIES)].copy()
    gdf_species['BVI'] = gdf_species.apply(calculate_bvi, axis=1)
    print(f"   - 筛选出 {len(gdf_species)} 条受威胁物种记录 (含NT)。")

    print(f"-> 加载湖泊地理形状数据 (predicted_lakes_2022.shp)...")
    gdf_lakes = gpd.read_file(MP_LAKES_SHP_PATH, engine="pyogrio")
    if gdf_lakes.crs is None: gdf_lakes.set_crs("EPSG:4326", inplace=True) # 通常是 WGS84
    # 重命名ID列以便于连接，并仅保留必要的几何信息
    gdf_lakes.rename(columns={LAKE_ID_COLUMN: 'hylak_id'}, inplace=True)
    gdf_lakes = gdf_lakes[['hylak_id', 'geometry']]
    print(f"   - 成功加载 {len(gdf_lakes)} 个湖泊的几何形状。")

    print(f"-> 加载 MP 丰度与优化潜力数据 (Ychange.csv)...")
    # 读取CSV并将其转换为GeoDataFrame
    potential_df = pd.read_csv(REDUCTION_POTENTIAL_CSV)
    cols_to_use = ['lon', 'lat', 'MPs-ori', 'change']
    potential_df = potential_df[cols_to_use].dropna(subset=['lon', 'lat', 'MPs-ori'])
    points_gdf = gpd.GeoDataFrame(
        potential_df, geometry=gpd.points_from_xy(potential_df.lon, potential_df.lat), crs=gdf_lakes.crs
    )
    print(f"   - 成功加载 {len(points_gdf)} 个 MP 数据点。")

except Exception as e:
    print(f"[严重错误] 数据加载失败: {e}");
    exit()

# --- 4. 情景构建 (完全重写) ---
print("\n--- [步骤 2] 构建基线与优化情景 ---")

print("-> 将 MP 数据点空间连接 (sjoin) 到湖泊面...")
# 将湖泊的hylak_id和几何图形与数据点进行空间连接
merged_gdf = gpd.sjoin(points_gdf, gdf_lakes, how="inner", predicate="within")

print("-> 聚合每个湖泊内的多个数据点 (计算均值)...")
# 如果一个湖泊(hylak_id)内有多个点，对这些点的'MPs-ori'和'change'值取平均
aggregation_rules = {
    'MPs-ori': 'mean',
    'change': 'mean'
}
aggregated_data = merged_gdf.groupby('hylak_id').agg(aggregation_rules).reset_index()
print(f"   - {len(aggregated_data)} 个湖泊具有 MP 数据。")

print("-> 将聚合后的 MP 数据合并回主湖泊数据框...")
# 使用 left merge，这样所有湖泊都被保留
gdf_lakes_scenario = gdf_lakes.merge(aggregated_data, on='hylak_id', how='left')

# --- [核心逻辑] 根据您的要求，严格筛选有优化数据的湖泊 ---
records_before_drop = len(gdf_lakes_scenario)
# 删除在 Ychange.csv 中没有对应数据点的湖泊记录
gdf_lakes_scenario.dropna(subset=['MPs-ori'], inplace=True)
records_after_drop = len(gdf_lakes_scenario)
print(f"   - {records_before_drop - records_after_drop} 个湖泊因缺少 'Ychange.csv' 数据点而被排除。")
print(f"   - 剩下 {records_after_drop} 个湖泊将用于风险计算。")

print("-> 创建基线和优化情景的 MP 丰度...")
# 1. 基线 (`mp_baseline`) 现在严格来自 'MPs-ori'
gdf_lakes_scenario['mp_baseline'] = gdf_lakes_scenario['MPs-ori']

# 2. 计算优化情景 (`mp_optimized`)
# 'change'列中的NaN用0填充（表示无变化），且确保change值<=0 (代表减少或不变)
gdf_lakes_scenario['change'] = gdf_lakes_scenario['change'].fillna(0).clip(upper=0)
gdf_lakes_scenario['mp_optimized'] = (gdf_lakes_scenario['mp_baseline'] + gdf_lakes_scenario['change']).clip(lower=0)
print("   - 'mp_baseline' (来自MPs-ori) 和 'mp_optimized' 两个情景已成功创建。")

# --- 5. 高效风险计算 ---
print("\n--- [步骤 3] 执行高效情景风险叠加分析 ---")

# --- [新增] 数据验证步骤 ---
print("-> 检查物种数据是否包含所有必需字段...")
required_species_cols = ['class', 'order_', 'freshwater', 'marine', 'terrestria', 'SEASONAL']
missing_cols = [col for col in required_species_cols if col not in gdf_species.columns]
if missing_cols:
    print(f"[严重错误] 物种数据文件 '{SPECIES_GPKG_PATH}' 缺少以下必需字段: {missing_cols}")
    exit()
print("   - 数据验证通过，所有必需字段均存在。")

print("-> 投影数据以进行空间分析...")
gdf_species_proj = gdf_species.to_crs(EQUAL_AREA_PROJ)
species_cols = [
    'SCI_NAME', 'redlist_category', 'BVI',
    'class', 'order_', 'freshwater', 'marine', 'terrestria', 'SEASONAL',
    'geometry'
]
lakes_cols = ['hylak_id', 'mp_baseline', 'mp_optimized', 'geometry']
gdf_lakes_proj = gdf_lakes_scenario[lakes_cols].to_crs(EQUAL_AREA_PROJ)

print("-> 简化几何图形以加快叠加速度...")
tolerance_meters = 5000
gdf_species_proj['geometry'] = gdf_species_proj.geometry.simplify(tolerance_meters)
gdf_lakes_proj['geometry'] = gdf_lakes_proj.geometry.simplify(tolerance_meters)

print("-> 执行空间叠加分析 (overlay)...")
intersected_gdf = gpd.overlay(gdf_species_proj[species_cols], gdf_lakes_proj, how='intersection', keep_geom_type=False)
print(f"   - 空间叠加完成，生成 {len(intersected_gdf)} 条 湖泊-物种 相交记录。")

print("-> 计算基线和优化情景的 IRS (综合风险分数)...")
intersected_gdf['IRS_baseline'] = intersected_gdf['mp_baseline'] * intersected_gdf['BVI']
intersected_gdf['IRS_optimized'] = intersected_gdf['mp_optimized'] * intersected_gdf['BVI']

# --- 6. 保存核心产物 ---
print(f"\n--- [步骤 4] 保存包含双情景风险的核心GPKG产物 ---")
try:
    # 定义最终要保存的列
    final_columns = [
        'hylak_id', 'SCI_NAME', 'redlist_category',
        'class', 'order_', 'SEASONAL', 'freshwater', 'marine', 'terrestria',
        'IRS_baseline', 'IRS_optimized', 'geometry'
    ]
    # 筛选出在 intersected_gdf 中真实存在的列进行保存
    final_columns_exist = [col for col in final_columns if col in intersected_gdf.columns]

    # 直接保存，因为所有需要的列都已在 intersected_gdf 中
    intersected_gdf[final_columns_exist].to_file(FINAL_INTERSECTION_GPKG, driver='GPKG', engine='pyogrio')
    print(f"   - [成功] 已将核心产物保存至: {FINAL_INTERSECTION_GPKG}")
except Exception as e:
    print(f"   - [错误] 保存最终GPKG文件失败: {e}")

# --- 7. 汇总与制图 (诊断性输出) ---
print("\n--- [步骤 5] (诊断性)汇总风险降低量并保存 ---")

# 使用新的 IRS_baseline 和 IRS_optimized 列进行聚合
risk_summary = intersected_gdf.groupby(['hylak_id', 'redlist_category']).agg(
    ACR_baseline=('IRS_baseline', 'sum'),
    ACR_optimized=('IRS_optimized', 'sum'),
    species_count=('SCI_NAME', 'nunique')
).reset_index()

risk_summary['CNEI_baseline'] = risk_summary.apply(
    lambda row: row['ACR_baseline'] / row['species_count'] if row['species_count'] > 0 else 0, axis=1)
risk_summary['CNEI_optimized'] = risk_summary.apply(
    lambda row: row['ACR_optimized'] / row['species_count'] if row['species_count'] > 0 else 0, axis=1)
risk_summary['CNEI_REDUCTION'] = risk_summary['CNEI_baseline'] - risk_summary['CNEI_optimized']

delta_pivot = risk_summary.pivot_table(index='hylak_id', columns='redlist_category',
                                       values=['CNEI_REDUCTION']).reset_index()
delta_pivot.columns = [f"{val}_{cat.split(' ')[0]}" if cat != '' else val for val, cat in delta_pivot.columns]
delta_pivot.rename(columns={'hylak_id_': 'hylak_id'}, inplace=True)

# 与包含地理位置的 gdf_lakes_scenario 合并以获取几何信息
gdf_lakes_with_delta = gdf_lakes_scenario.merge(delta_pivot, on='hylak_id', how='left').fillna(0)
gdf_lakes_with_delta.drop(columns='geometry').to_csv(FINAL_DELTA_METRICS_CSV, index=False)
print(f"-> 诊断性Delta指标已保存至CSV: {FINAL_DELTA_METRICS_CSV}")


print("\n==========================================================")
print("  情景分析 (已根据新逻辑全面修订) 计算任务成功完成！")
print(f"  核心产物已保存至: {FINAL_INTERSECTION_GPKG}")
print("==========================================================")