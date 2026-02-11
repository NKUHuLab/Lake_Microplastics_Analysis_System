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

# 设置绘图参数
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
warnings.filterwarnings("ignore")

# --- 1. 路径和参数配置 ---
PROJECT_ROOT_ANI = r"E:\lake-MP-ani"
PROJECT_ROOT_W = r"E:\lake-MP-W"

# --- 输入路径 (需要所有原始数据) ---
# 1. 物种数据
SPECIES_GPKG_PATH = os.path.join(PROJECT_ROOT_ANI, "data", "IUCN_consolidated", "all_species_classified.gpkg")
# 2. 湖泊数据 (包含基线MP丰度)
MP_LAKES_SHP_PATH = os.path.join(PROJECT_ROOT_W, "data", "generated_shp", "predicted_lakes_2022.shp")
# 3. 世界底图
WORLD_BORDERS_SHP = os.path.join(PROJECT_ROOT_W, "data", "base_shp", "world map china line.shp")
# 4. 您的优化潜力CSV文件 (关键的新输入)
REDUCTION_POTENTIAL_CSV = os.path.join(PROJECT_ROOT_W, "data", "opt", "data", "processed_output", "Ychange.csv")

# --- 输出路径 (新目录) ---
OUTPUT_DIR_DELTA = os.path.join(PROJECT_ROOT_ANI, "draw", "Risk_Reduction_Delta_Maps")
os.makedirs(OUTPUT_DIR_DELTA, exist_ok=True)
FINAL_DELTA_METRICS_CSV = os.path.join(OUTPUT_DIR_DELTA, "lakes_risk_reduction_delta_metrics.csv")

# --- 分析参数 ---
EQUAL_AREA_PROJ = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
GLOBAL_ROBINSON_PROJ = "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
LAKE_ID_COLUMN = 'FID'  # 湖泊SHP中的原始ID
MP_BASELINE_COLUMN = 'prediction'  # 湖泊SHP中的基线MP丰度列
TARGET_CATEGORIES = ['Critically Endangered', 'Endangered', 'Vulnerable', 'Near Threatened']

# --- [新] 地图颜色条 (采用您示例脚本中的 "优化潜力" 颜色条) ---
# 灰色 -> 淡黄 -> 绿 -> 深蓝 (代表 0 降低 -> 高降低)
REDUCTION_CMAP_COLORS = [
    "#E7E7E7",  # 灰色 (用于0值或低值)
    "#FFFFE0",  # 淡黄色
    "#afe1af",  # 淡绿色
    "#4CAF50",  # 鲜明的绿色
    "#6aa4bb",  # 淡蓝色
    "#4885c1",  # 蓝色
    "#08306B"  # 深蓝色 (最高潜力)
]
REDUCTION_CMAP = LinearSegmentedColormap.from_list("reduction_potential_cmap", REDUCTION_CMAP_COLORS)


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
    # 1. 加载物种数据并计算 BVI
    print("-> 加载物种数据并计算 BVI...")
    gdf_species_raw = gpd.read_file(SPECIES_GPKG_PATH, engine="pyogrio")
    gdf_species = gdf_species_raw[gdf_species_raw['redlist_category'].isin(TARGET_CATEGORIES)].copy()
    gdf_species['BVI'] = gdf_species.apply(calculate_bvi, axis=1)
    print(f"   - 筛选出 {len(gdf_species)} 条受威胁物种记录 (含NT)。")

    # 2. 加载湖泊地理数据并设置基线MP
    print(f"-> 加载湖泊微塑料基线数据...")
    gdf_lakes = gpd.read_file(MP_LAKES_SHP_PATH, engine="pyogrio")
    if gdf_lakes.crs is None:
        gdf_lakes.set_crs("EPSG:4326", inplace=True)

    # 重命名关键列
    gdf_lakes.rename(columns={LAKE_ID_COLUMN: 'hylak_id', MP_BASELINE_COLUMN: 'mp_baseline'}, inplace=True)
    print(f"   - 成功加载 {len(gdf_lakes)} 个湖泊 (基线情景)。")

    # 3. 加载并处理优化潜力数据 (Ychange.csv)
    print("-> 加载优化潜力数据 (Ychange.csv)...")
    data_df = pd.read_csv(REDUCTION_POTENTIAL_CSV)
    data_df = data_df[['lon', 'lat', 'change']].dropna()
    # 转换为地理点数据
    points_gdf = gpd.GeoDataFrame(
        data_df, geometry=gpd.points_from_xy(data_df.lon, data_df.lat), crs=gdf_lakes.crs
    )
    print(f"   - 成功加载 {len(points_gdf)} 个优化潜力点。")

except Exception as e:
    print(f"[严重错误] 数据加载失败: {e}")
    exit()

# --- 4. 情景构建：合并优化数据并创建两个MP情景 ---
print("\n--- [步骤 2] 构建基线与优化情景 ---")

print("-> 将优化点空间连接 (sjoin) 到湖泊面...")
# 执行空间连接 (类似您的示例脚本)
merged_gdf_points = gpd.sjoin(gdf_lakes, points_gdf, how="left", predicate="contains")

# 一个湖泊可能包含多个点，计算均值 (也类似您的示例)
if merged_gdf_points.index.duplicated().any():
    print("   - 聚合多个点的湖泊 (计算均值)...")
    # 保留原始地理信息和关键列
    original_geoms = gdf_lakes.loc[merged_gdf_points.index.unique()].geometry
    original_crs = gdf_lakes.crs
    original_data = gdf_lakes[['hylak_id', 'mp_baseline']].loc[merged_gdf_points.index.unique()]

    # 按索引（湖泊ID）分组并计算 'change' 的均值
    aggregated_change = merged_gdf_points.groupby(merged_gdf_points.index)[['change']].mean()

    # 重新组合
    gdf_lakes_scenario = original_data.join(aggregated_change)
    gdf_lakes_scenario = gpd.GeoDataFrame(gdf_lakes_scenario, geometry=original_geoms, crs=original_crs)
else:
    print("   - 每个湖泊最多1个点，直接合并。")
    gdf_lakes_scenario = merged_gdf_points.drop(columns=['index_right', 'lon', 'lat'])

# 确保所有湖泊都在 (那些没有匹配到任何点的湖泊)
gdf_lakes_scenario = gdf_lakes.merge(gdf_lakes_scenario.drop(columns=['geometry', 'mp_baseline']), on='hylak_id',
                                     how='left')

# 将没有优化点的湖泊的 'change' 设为 0
gdf_lakes_scenario['change'] = gdf_lakes_scenario['change'].fillna(0)
# 确保 change 值为负（或0）
gdf_lakes_scenario['change'] = gdf_lakes_scenario['change'].clip(upper=0)
print("   - 成功将优化潜力分配到湖泊。")

# 步骤 4.3: 创建最终的两个情景MP列
gdf_lakes_scenario['mp_optimized'] = gdf_lakes_scenario['mp_baseline'] + gdf_lakes_scenario['change']
# 确保优化后的丰度不会低于0
gdf_lakes_scenario['mp_optimized'] = gdf_lakes_scenario['mp_optimized'].clip(lower=0)
print("   - 'mp_baseline' 和 'mp_optimized' 两个情景已创建。")

# --- 5. 高效风险计算 (一次Overlay, 两次计算) ---
print("\n--- [步骤 3] 执行高效情景风险叠加分析 ---")

print("-> 投影数据以进行空间分析...")
gdf_species_proj = gdf_species.to_crs(EQUAL_AREA_PROJ)
# 只保留需要的列以提高效率
species_cols = ['SCI_NAME', 'redlist_category', 'BVI', 'geometry']

# 关键：湖泊层包含两个情景
lakes_cols = ['hylak_id', 'mp_baseline', 'mp_optimized', 'geometry']
gdf_lakes_proj = gdf_lakes_scenario[lakes_cols].to_crs(EQUAL_AREA_PROJ)

print("-> 执行空间叠加分析 (Overlay)...")
intersected_gdf = gpd.overlay(gdf_species_proj[species_cols], gdf_lakes_proj, how='intersection', keep_geom_type=False)
print(f"   - 空间叠加完成，生成 {len(intersected_gdf)} 条 湖泊-物种 相交记录。")

print("-> 同时计算两个情景的 IRS (综合风险分数)...")
# 情景 1: 基线风险
intersected_gdf['IRS_baseline'] = intersected_gdf['mp_baseline'] * intersected_gdf['BVI']
# 情景 2: 优化后风险
intersected_gdf['IRS_optimized'] = intersected_gdf['mp_optimized'] * intersected_gdf['BVI']

print("-> 汇总两个情景的ACR和CNEI...")
# 在一次 groupby 中同时聚合两个情景
risk_summary = intersected_gdf.groupby(['hylak_id', 'redlist_category']).agg(
    ACR_baseline=('IRS_baseline', 'sum'),
    ACR_optimized=('IRS_optimized', 'sum'),
    species_count=('SCI_NAME', 'nunique')
).reset_index()

# 计算两个情景的CNEI
risk_summary['CNEI_baseline'] = risk_summary['ACR_baseline'] / risk_summary['species_count']
risk_summary['CNEI_optimized'] = risk_summary['ACR_optimized'] / risk_summary['species_count']

# --- 6. 核心指标：计算风险降低量 (Delta) ---
print("\n--- [步骤 4] 计算最终风险降低量 (Delta-CNEI) ---")

# 计算绝对降低量 (这是我们要绘制的地图)
risk_summary['CNEI_REDUCTION'] = risk_summary['CNEI_baseline'] - risk_summary['CNEI_optimized']

# (可选) 计算百分比降低量
# risk_summary['CNEI_REDUCTION_PCT'] = (risk_summary['CNEI_REDUCTION'] / risk_summary['CNEI_baseline']).fillna(0) * 100
# risk_summary['CNEI_REDUCTION_PCT'] = risk_summary['CNEI_REDUCTION_PCT'].clip(0, 100) # 确保百分比在0-100之间

print("   - 风险降低量计算完成。")

print("-> 转换 (Pivot) 数据以便于制图...")
# 我们只需要绘制降低量
delta_pivot = risk_summary.pivot_table(
    index='hylak_id',
    columns='redlist_category',
    values=['CNEI_REDUCTION']  # , 'CNEI_REDUCTION_PCT']
).reset_index()

# 展平多重索引列名
delta_pivot.columns = [f"{val}_{cat.split(' ')[0]}" if cat != '' else val for val, cat in delta_pivot.columns]
delta_pivot.rename(columns={'hylak_id_': 'hylak_id'}, inplace=True)

# --- 7. 合并与制图 ---
print("\n--- [步骤 5] 合并地理数据并绘制最终风险降低地图 ---")

# 将计算得到的Delta指标合并回原始的湖泊地理数据
gdf_lakes_with_delta = gdf_lakes_scenario.merge(delta_pivot, on='hylak_id', how='left').fillna(0)

# 保存最终的Delta指标CSV文件
gdf_lakes_with_delta.drop(columns='geometry').to_csv(FINAL_DELTA_METRICS_CSV, index=False)
print(f"-> 最终的Delta指标已保存至CSV: {FINAL_DELTA_METRICS_CSV}")


# --- 绘图函数 (使用您的潜力色带) ---
def plot_global_delta_map(gdf, column, title, output_path, cmap, dpi=1000):
    """绘制全球风险降低量 (Delta) 地图的通用函数。"""
    print(f"-> 正在绘制地图: {title}")
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_facecolor('white')

    world = gpd.read_file(WORLD_BORDERS_SHP)
    world = world[world['NAME'] != 'Antarctica']
    world_proj = world.to_crs(GLOBAL_ROBINSON_PROJ)

    # 绘制背景 (移除俄罗斯边境)
    world_proj[world_proj['NAME'] == 'Russia'].plot(ax=ax, color='#E7E7E7', edgecolor=None, zorder=1)
    world_proj[world_proj['NAME'] != 'Russia'].plot(ax=ax, color='#E7E7E7', edgecolor=None, zorder=1)
    world_proj[world_proj['NAME'] != 'Russia'].boundary.plot(ax=ax, linewidth=0.3, color='darkgrey', zorder=2)

    gdf_proj = gdf.to_crs(GLOBAL_ROBINSON_PROJ)

    # 只绘制那些风险降低量 > 0 的湖泊 (即优化有效的湖泊)
    plot_data = gdf_proj[gdf_proj[column] > 0]

    if not plot_data.empty:
        # 使用归一化确保色带从0开始
        norm = Normalize(vmin=0, vmax=plot_data[column].quantile(0.99))  # 使用99%分位数避免极端值拉伸色带

        plot_data.plot(
            column=column,
            ax=ax,
            legend=True,
            cmap=cmap,  # 使用您示例中的潜力色带
            norm=norm,
            legend_kwds={'orientation': "horizontal", 'shrink': 0.6, 'pad': 0.01,
                         'label': '濒危物种暴露风险降低量 (Delta-CNEI)'},
            zorder=3
        )

    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_axis_off()

    plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# --- 循环调用绘图函数 ---
plot_global_delta_map(gdf_lakes_with_delta, 'CNEI_REDUCTION_Critically', "全球极危(CR)物种微塑料暴露风险降低效益图",
                      os.path.join(OUTPUT_DIR_DELTA, "Map_A_Delta_CR.png"), cmap=REDUCTION_CMAP, dpi=1000)

plot_global_delta_map(gdf_lakes_with_delta, 'CNEI_REDUCTION_Endangered', "全球濒危(EN)物种微塑料暴露风险降低效益图",
                      os.path.join(OUTPUT_DIR_DELTA, "Map_B_Delta_EN.png"), cmap=REDUCTION_CMAP, dpi=1000)

plot_global_delta_map(gdf_lakes_with_delta, 'CNEI_REDUCTION_Vulnerable', "全球易危(VU)物种微塑料暴露风险降低效益图",
                      os.path.join(OUTPUT_DIR_DELTA, "Map_C_Delta_VU.png"), cmap=REDUCTION_CMAP, dpi=1000)

plot_global_delta_map(gdf_lakes_with_delta, 'CNEI_REDUCTION_Near', "全球近危(NT)物种微塑料暴露风险降低效益图",
                      os.path.join(OUTPUT_DIR_DELTA, "Map_D_Delta_NT.png"), cmap=REDUCTION_CMAP, dpi=1000)

print("\n==========================================================")
print("  情景分析 (Delta-CNEI) 计算和制图任务成功完成！")
print(f"  所有输出文件已保存至: {OUTPUT_DIR_DELTA}")
print("==========================================================")