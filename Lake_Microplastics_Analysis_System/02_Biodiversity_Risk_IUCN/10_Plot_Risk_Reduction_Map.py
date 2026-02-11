# -*- coding: utf-8 -*-
import os
import warnings
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- 0. 脚本说明 ---
print("--- [初始化] 实施思路B：基于IRS数据计算湖泊总风险降低分数并绘图 ---")
print("--- 目标：放弃WRS，利用脚本1的高尺度风险数据，为每个湖泊计算总分并绘制地图 ---")

# --- 1. 全局配置与初始化 ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
warnings.filterwarnings("ignore")

# --- 2. 路径和参数配置 ---

# --- 输入路径 ---
INTERMEDIATE_SJOIN_GPKG = r"E:\lake-MP-ani\draw\111Biological_Attribution_Analysis_SJoin_Fast\data_summaries\intermediate_lake_species_intersections.gpkg"
MP_LAKES_SHP_PATH = r"E:\lake-MP-W\data\generated_shp\predicted_lakes_2022.shp"
WORLD_BORDERS_SHP = r"E:\lake-MP-W\data\base_shp\world map china line.shp"

# --- 输出路径 ---
OUTPUT_DIR = r"E:\lake-MP-ani\draw\Total_Reduction_Map_From_IRS"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FINAL_MAP_PNG = os.path.join(OUTPUT_DIR, "Map_Total_Weighted_Risk_Reduction_from_IRS.png")
FINAL_MAP_LEGEND_PDF = os.path.join(OUTPUT_DIR, "Map_Legend_Total_Weighted_Risk_Reduction_from_IRS.pdf")

# --- 分析参数 ---
GLOBAL_ROBINSON_PROJ = "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
LAKE_ID_COLUMN_SHP = 'FID'
LAKE_ID_COLUMN_GPKG = 'hylak_id'
THREAT_WEIGHTS = {'Critically Endangered': 50, 'Endangered': 25, 'Vulnerable': 10, 'Near Threatened': 5}
MAP_CMAP = 'YlGnBu'


# --- 3. 绘图函数定义 ([核心修改] 完全替换为脚本三的函数) ---

def plot_pure_global_map(gdf, column, output_path, cmap, norm, dpi=1000):
    """
    (源自脚本三的V3版本) 绘制纯净地图, 采用分层逻辑彻底移除所有不需要的边线。
    """
    print(f"-> 正在绘制全球地图 (使用脚本三的分层逻辑): {column}")
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_facecolor('white')  # 白色海洋

    # 1. 加载并准备世界底图 (采用脚本三的逻辑)
    world = gpd.read_file(WORLD_BORDERS_SHP)
    world_proj = world.to_crs(GLOBAL_ROBINSON_PROJ)
    world_proj['geometry'] = world_proj.geometry.buffer(0)  # 修复几何图形

    name_col = next((col for col in ['NAME', 'name', 'ADMIN', 'SOVEREIGNT'] if col in world_proj.columns), 'NAME')
    world_filtered = world_proj[world_proj[name_col] != 'Antarctica']
    world_for_borders = world_filtered[~world_filtered[name_col].str.contains("russia", case=False, na=False)]

    # --- 开始分层绘制 ---
    # 图层 1: 绘制白色陆地填充 (无边框)
    world_filtered.plot(ax=ax, color='white', edgecolor='none', zorder=1)

    # 2. 准备湖泊数据
    gdf_proj = gdf.to_crs(GLOBAL_ROBINSON_PROJ)
    data_with_data = gdf_proj[gdf_proj[column] > 0]
    data_na = gdf_proj[gdf_proj[column] <= 0]

    # 图层 2: 绘制无数据的湖泊 (灰色填充, 无边框)
    data_na.plot(
        color="#E7E7E7",  # 灰色填充
        linewidth=0,
        edgecolor='none',
        ax=ax,
        zorder=2
    )

    # 图层 3: 绘制有数据的湖泊 (彩色填充, 无边框)
    if not data_with_data.empty:
        data_with_data.plot(
            column=column,
            ax=ax,
            cmap=cmap,
            norm=norm,
            legend=False,
            zorder=3,
            linewidth=0,
            edgecolor='none'
        )

    # 图层 4: 绘制国界线 (不含俄罗斯)
    world_for_borders.plot(ax=ax, color='none', edgecolor='darkgrey', linewidth=0.3, zorder=4)

    ax.set_axis_off()
    plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"   - 全球地图已保存至: {output_path}")


def create_separate_legend(output_pdf, cmap, norm, label_text, extend_opt='neither'):
    """创建独立的、可编辑的PDF图例。"""
    print(f"-> 正在生成独立的PDF图例...")
    fig_legend = plt.figure(figsize=(1.5, 6))
    ax_legend = fig_legend.add_axes([0.1, 0.1, 0.2, 0.8])

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                        cax=ax_legend,
                        orientation='vertical',
                        extend=extend_opt)

    cbar.set_label(label_text, rotation=270, labelpad=25, fontsize=12, weight='bold')
    ax_legend.tick_params(labelsize=10)
    fig_legend.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig_legend)
    print(f"   - PDF图例已保存至: {output_pdf}")


# --- 4. 主执行流程 (此部分无任何改动) ---

def main_analysis_and_plotting():
    """执行数据处理、分数计算和最终绘图。"""

    # --- 步骤 1: 加载数据 ---
    print("\n--- [步骤 1/5] 加载核心数据... ---")
    try:
        df_irs = gpd.read_file(INTERMEDIATE_SJOIN_GPKG,
                               columns=[LAKE_ID_COLUMN_GPKG, 'redlist_category', 'IRS_baseline', 'IRS_optimized'],
                               engine='pyogrio')
        print(f"   - 成功加载 {len(df_irs)} 条物种-湖泊交互记录。")

        gdf_lakes_geom = gpd.read_file(MP_LAKES_SHP_PATH, engine="pyogrio")
        if gdf_lakes_geom.crs is None:
            print("   - [警告] 湖泊地理数据缺少CRS信息，将强制设置为 WGS 84 (EPSG:4326)。")
            gdf_lakes_geom.set_crs("EPSG:4326", inplace=True)

        gdf_lakes_geom.rename(columns={LAKE_ID_COLUMN_SHP: LAKE_ID_COLUMN_GPKG}, inplace=True)
        gdf_lakes_geom = gdf_lakes_geom[[LAKE_ID_COLUMN_GPKG, 'geometry']].copy()
        print(f"   - 成功加载 {len(gdf_lakes_geom)} 个湖泊的地理信息。")

    except Exception as e:
        print(f"[严重错误] 数据加载失败: {e}")
        return

    # --- 步骤 2: 计算每个交互记录的加权风险降低值 ---
    print("\n--- [步骤 2/5] 计算每个交互记录的加权风险降低值... ---")
    df_irs['Threat_Score'] = df_irs['redlist_category'].map(THREAT_WEIGHTS).fillna(0)
    df_irs = df_irs[df_irs['Threat_Score'] > 0].copy()
    df_irs['Reduction'] = (pd.to_numeric(df_irs['IRS_baseline'], errors='coerce').fillna(0) -
                           pd.to_numeric(df_irs['IRS_optimized'], errors='coerce').fillna(0)).clip(lower=0)
    df_irs['Weighted_Reduction'] = df_irs['Reduction'] * df_irs['Threat_Score']
    print("   - 加权风险降低值计算完成。")

    # --- 步骤 3: 按湖泊ID聚合，计算总分 ---
    print(f"\n--- [步骤 3/5] 按湖泊ID '{LAKE_ID_COLUMN_GPKG}' 进行分组求和... ---")
    if df_irs.empty:
        print("[错误] 筛选后没有可供分析的数据。")
        return
    df_lake_scores = df_irs.groupby(LAKE_ID_COLUMN_GPKG)[['Weighted_Reduction']].sum().reset_index()
    df_lake_scores.rename(columns={'Weighted_Reduction': 'TWRS'}, inplace=True)
    print(f"   - 成功为 {len(df_lake_scores)} 个湖泊计算了总风险降低分数 (TWRS)。")

    # --- 步骤 4: 合并分数与地理信息，并准备绘图 ---
    print("\n--- [步骤 4/5] 合并分数与地理信息，并准备色彩方案... ---")
    gdf_to_plot = gdf_lakes_geom.merge(df_lake_scores, on=LAKE_ID_COLUMN_GPKG, how='left')
    gdf_to_plot['TWRS'] = gdf_to_plot['TWRS'].fillna(0)
    data_gt_zero = gdf_to_plot[gdf_to_plot['TWRS'] > 0]['TWRS']
    if not data_gt_zero.empty:
        vmin = 0
        vmax_q95 = data_gt_zero.quantile(0.95)
        if vmax_q95 <= vmin:
            vmax_q95 = data_gt_zero.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax_q95, clip=True)
        legend_extend = 'max'
        legend_label = "Total Weighted Risk Reduction Score (TWRS)\n(Linear Scale, Clipped at 95th Percentile)"
        print(f"   - 绘图色彩方案已设定 (线性截断): Min=0, Max(Q95)={vmax_q95:.2f}")
    else:
        print("   - [警告] 未发现任何风险降低大于0的湖泊。")
        norm = mcolors.Normalize(vmin=0, vmax=1)
        legend_extend = 'neither'
        legend_label = "Total Weighted Risk Reduction Score (TWRS)"

    # --- 步骤 5: 生成地图和图例 ---
    print("\n--- [步骤 5/5] 开始生成最终输出文件... ---")
    plot_pure_global_map(
        gdf=gdf_to_plot,
        column='TWRS',
        output_path=FINAL_MAP_PNG,
        cmap=MAP_CMAP,
        norm=norm,
        dpi=1000
    )
    create_separate_legend(
        output_pdf=FINAL_MAP_LEGEND_PDF,
        cmap=MAP_CMAP,
        norm=norm,
        label_text=legend_label,
        extend_opt=legend_extend
    )

# --- 脚本执行入口 ---
if __name__ == "__main__":
    main_analysis_and_plotting()

    print("\n===================================================================")
    print("  思路B执行完毕！")
    print("  已成功使用IRS数据源为每个湖泊计算总分并绘制了新的全球地图。")
    print(f"  所有输出文件已保存至: {os.path.abspath(OUTPUT_DIR)}")
    print("===================================================================")