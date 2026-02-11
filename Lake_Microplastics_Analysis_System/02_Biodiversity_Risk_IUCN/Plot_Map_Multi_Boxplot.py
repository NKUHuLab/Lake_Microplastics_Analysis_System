import os
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np  # 导入Numpy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap  # 导入颜色条工具
from matplotlib.patches import Patch  # 导入图例工具

# --- 0. 全局配置与初始化 ---
print("--- [初始化] 设置全局参数和输出目录 ---")

# 设置绘图参数以确保PDF中的文本是可编辑的
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

# --- 1. 路径和参数配置 ---
# --- 输入路径 ---
SPECIES_GPKG_PATH = r"E:\lake-MP-ani\data\IUCN_consolidated\all_species_classified.gpkg"
MP_LAKES_SHP_PATH = r"E:\lake-MP-W\data\generated_shp\predicted_lakes_2022.shp"
WORLD_BORDERS_SHP = r"E:\lake-MP-W\data\base_shp\world map china line.shp"

# --- 输出路径 ---
OUTPUT_DIR = r"E:\lake-MP-ani\draw\final_risk_analysis_outputs_v2"  # 使用新目录以避免覆盖旧结果
os.makedirs(OUTPUT_DIR, exist_ok=True)
FINAL_METRICS_CSV = os.path.join(OUTPUT_DIR, "lakes_risk_metrics_final_with_NT.csv")

# --- 分析参数 ---
EQUAL_AREA_PROJ = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
GLOBAL_ROBINSON_PROJ = "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
LAKE_ID_COLUMN = 'FID'
MP_ABUNDANCE_COLUMN = 'prediction'

# --- [修改点 2] ---
# 扩展目标类别，加入 'Near Threatened' (NT)
TARGET_CATEGORIES = ['Critically Endangered', 'Endangered', 'Vulnerable', 'Near Threatened']

# --- [修改点 3] ---
# 定义统一的 "蓝-黄-红" 渐变颜色条
# 我们选择一个平静的蓝色 (#006D90), 中性的黄色 (#FEE89F), 和一个强烈的红色 (#A50026)
CUSTOM_CMAP_COLORS = ['#006D90', '#FEE89F', '#A50026']
UNIFIED_RISK_CMAP = LinearSegmentedColormap.from_list("unified_risk_cmap", CUSTOM_CMAP_COLORS)

# --- 2. 核心计算函数定义 ---
print("\n--- [定义模型] 定义 BVI 和 IRS 计算逻辑 ---")


def calculate_bvi(row):
    """
    根据 seasonal 字段计算生物脆弱性指数 (BVI)。
    """
    W_season = 1.0
    if 'SEASONAL' in row and pd.notna(row['SEASONAL']) and row['SEASONAL'] == 2:
        W_season = 1.5
    return W_season


# --- 3. 数据加载与预处理 ---
# (由于必须包含NT，这部分需要重新运行)
print("\n--- [步骤 2.1a] 加载与预处理数据 (包含NT) ---")
try:
    print(f"-> 正在加载物种数据: {SPECIES_GPKG_PATH}")
    gdf_species_raw = gpd.read_file(SPECIES_GPKG_PATH, engine="pyogrio")

    # 使用更新后的 TARGET_CATEGORIES 进行筛选
    gdf_species = gdf_species_raw[gdf_species_raw['redlist_category'].isin(TARGET_CATEGORIES)].copy()
    print(f"   - 成功加载并筛选出 {len(gdf_species)} 条受威胁物种记录 (包括 CR, EN, VU, NT)。")

    print(f"-> 正在加载湖泊微塑料数据: {MP_LAKES_SHP_PATH}")
    gdf_lakes = gpd.read_file(MP_LAKES_SHP_PATH, engine="pyogrio")

    if gdf_lakes.crs is None:
        print("   - [警告] 湖泊数据未定义坐标系，将手动设置为 WGS84 (EPSG:4326)。")
        gdf_lakes.set_crs("EPSG:4326", inplace=True)

    gdf_lakes.rename(columns={LAKE_ID_COLUMN: 'hylak_id', MP_ABUNDANCE_COLUMN: 'mp_abundance'}, inplace=True)
    print(f"   - 成功加载 {len(gdf_lakes)} 个湖泊记录。")

except Exception as e:
    print(f"[严重错误] 数据加载失败: {e}")
    exit()

# --- 4. 空间连接与风险计算 ---
# (由于必须包含NT，这部分需要重新运行)
print("\n--- [步骤 2.1b] 执行空间连接与风险计算 (包含NT) ---")

print("-> 计算每个物种的 BVI...")
gdf_species['BVI'] = gdf_species.apply(calculate_bvi, axis=1)

print("-> 投影数据以进行空间分析...")
gdf_species_proj = gdf_species.to_crs(EQUAL_AREA_PROJ)
gdf_lakes_proj = gdf_lakes.to_crs(EQUAL_AREA_PROJ)

print("-> 执行空间叠加分析 (overlay)，这可能需要较长时间...")
species_cols = ['SCI_NAME', 'redlist_category', 'BVI', 'geometry']
lakes_cols = ['hylak_id', 'mp_abundance', 'geometry']
intersected_gdf = gpd.overlay(gdf_species_proj[species_cols], gdf_lakes_proj[lakes_cols], how='intersection',
                              keep_geom_type=False)

print("-> 计算每个'湖泊-物种'组合的综合风险分数 (IRS)...")
intersected_gdf['IRS_species'] = intersected_gdf['mp_abundance'] * intersected_gdf['BVI']

# --- 5. 汇总与指标生成 ---
# (由于必须包含NT，这部分需要重新运行)
print("\n--- [步骤 2.1c] 汇总数据并生成ACR和CNEI指标 (包含NT) ---")

risk_summary = intersected_gdf.groupby(['hylak_id', 'redlist_category']).agg(
    ACR=('IRS_species', 'sum'),
    species_count=('SCI_NAME', 'nunique')
).reset_index()

risk_summary['CNEI'] = risk_summary['ACR'] / risk_summary['species_count']

final_metrics = risk_summary.pivot_table(
    index='hylak_id',
    columns='redlist_category',
    values=['ACR', 'CNEI', 'species_count']
).reset_index()

# 自动处理列名 (现在将自动包括 '..._Near')
final_metrics.columns = [f"{val}_{cat.split(' ')[0]}" for val, cat in final_metrics.columns]
final_metrics.rename(columns={'hylak_id_': 'hylak_id'}, inplace=True)
final_metrics.fillna(0, inplace=True)

lakes_with_metrics = gdf_lakes.merge(final_metrics, on='hylak_id', how='left').fillna(0)

print(f"-> 正在保存最终的(含NT)湖泊风险指标数据至: {FINAL_METRICS_CSV}")
lakes_with_metrics.drop(columns='geometry').to_csv(FINAL_METRICS_CSV, index=False)

# --- 6. 专题地图绘制 (ACR) ---
print("\n--- [步骤 2.2] 绘制分级风险总量图 (ACR) [新版地图样式] ---")


# --- [修改点 1 & 3] ---
def plot_global_risk_map_v2(gdf, column, title, output_path, cmap, dpi=1000):
    """
    绘制全球风险地图的通用函数 (V2):
    - 移除南极洲
    - 移除俄罗斯边境线，保留其填充
    - 使用统一颜色条 (传入)
    - 输出为高DPI PNG
    """
    print(f"-> 正在绘制地图 (PNG @ {dpi}dpi): {title}")
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))  # DPI在savefig时指定
    ax.set_facecolor('white')

    # 加载世界地图并移除南极洲
    world = gpd.read_file(WORLD_BORDERS_SHP)
    world = world[world['NAME'] != 'Antarctica']
    world_proj = world.to_crs(GLOBAL_ROBINSON_PROJ)

    # 绘制背景地图:
    # 1. 绘制俄罗斯 (仅填充，无边框)
    world_proj[world_proj['NAME'] == 'Russia'].plot(ax=ax, color='#E7E7E7', edgecolor=None, zorder=1)

    # 2. 绘制所有其他国家 (填充)
    world_proj[world_proj['NAME'] != 'Russia'].plot(ax=ax, color='#E7E7E7', edgecolor=None, zorder=1)

    # 3. 绘制所有其他国家的边境线
    world_proj[world_proj['NAME'] != 'Russia'].boundary.plot(ax=ax, linewidth=0.3, color='darkgrey', zorder=2)

    gdf_proj = gdf.to_crs(GLOBAL_ROBINSON_PROJ)

    plot_data = gdf_proj[gdf_proj[column] > 0]
    if not plot_data.empty:
        plot_data.plot(
            column=column,
            ax=ax,
            legend=True,
            cmap=cmap,  # 使用传入的统一颜色条
            legend_kwds={'orientation': "horizontal", 'shrink': 0.6, 'pad': 0.01, 'label': '累积类别风险 (ACR)'},
            zorder=3  # 确保湖泊在最上层
        )

    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_axis_off()

    # 保存为高DPI PNG
    plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# --- 使用新函数绘制所有4个等级的地图 ---
plot_global_risk_map_v2(lakes_with_metrics, 'ACR_Critically', "全球极危(CR)物种微塑料累积风险热点分布图",
                        os.path.join(OUTPUT_DIR, "Map_A_ACR_CR.png"), cmap=UNIFIED_RISK_CMAP, dpi=1000)

plot_global_risk_map_v2(lakes_with_metrics, 'ACR_Endangered', "全球濒危(EN)物种微塑料累积风险热点分布图",
                        os.path.join(OUTPUT_DIR, "Map_B_ACR_EN.png"), cmap=UNIFIED_RISK_CMAP, dpi=1000)

plot_global_risk_map_v2(lakes_with_metrics, 'ACR_Vulnerable', "全球易危(VU)物种微塑料累积风险热点分布图",
                        os.path.join(OUTPUT_DIR, "Map_C_ACR_VU.png"), cmap=UNIFIED_RISK_CMAP, dpi=1000)

# [新增的NT地图]
plot_global_risk_map_v2(lakes_with_metrics, 'ACR_Near', "全球近危(NT)物种微塑料累积风险热点分布图",
                        os.path.join(OUTPUT_DIR, "Map_D_ACR_NT.png"), cmap=UNIFIED_RISK_CMAP, dpi=1000)

# --- 7. 核心统计分析 (CNEI) ---
# --- [修改点 4] ---
# 绘制新的分组箱线图 (替换掉所有旧统计图)
print("\n--- [步骤 2.3] 执行新的分组统计分析 (CNEI vs 生物多样性分箱) ---")


def plot_grouped_risk_boxplot(data_melted, bin_labels, cat_order, colors_dict, output_path):
    """
    绘制按物种多样性分箱的、按风险等级分组的 CNEI 箱线图。
    (风格类似于用户提供的示例)
    """
    print("-> 正在绘制 CNEI vs 生物多样性 分组箱线图...")

    n_categories = len(cat_order)
    n_bins = len(bin_labels)

    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

    x_centers = np.arange(n_bins)  # 每个bin的中心X坐标

    # 计算每个箱体的偏移量和宽度，使其并排排列
    total_width = 0.8  # 用于一组箱体的总宽度
    single_width = total_width / n_categories
    offsets = np.linspace(-total_width / 2 + single_width / 2, total_width / 2 - single_width / 2, n_categories)

    # 1. 绘制交替背景带 (类似示例)
    for i in range(len(x_centers)):
        if i % 2 == 1:  # 奇数索引添加背景色
            ax.axvspan(i - 0.5, i + 0.5, color='0.95', zorder=0, alpha=0.7)

    legend_patches = []

    # 2. 循环绘制每个类别的箱线图
    for i, category in enumerate(cat_order):
        cat_color = colors_dict[category]
        cat_offset = offsets[i]

        # 准备该类别下所有bin的数据 (列表的列表)
        data_groups_for_cat = []
        for bin_label in bin_labels:
            group_data = data_melted[
                (data_melted['Category'] == category) &
                (data_melted['species_bin'] == bin_label)
                ]['CNEI']
            data_groups_for_cat.append(group_data)

        # 定义样式 (类似示例)
        boxprops = dict(facecolor=cat_color, edgecolor='black', linewidth=0.8, alpha=0.9)
        medianprops = dict(color='black', linewidth=1.2)
        whiskerprops = dict(color='black', linewidth=0.8, linestyle='--')
        capprops = dict(color='black', linewidth=0.8)

        # 绘制箱线图 (sym='' 用于隐藏异常值，即示例中的 0, '')
        bp = ax.boxplot(
            data_groups_for_cat,
            positions=x_centers + cat_offset,
            widths=single_width * 0.9,  # 留一点点空隙
            patch_artist=True,
            sym='',  # 隐藏异常值 (fliers)
            boxprops=boxprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            zorder=3
        )

        # 为图例创建Patch
        legend_patches.append(Patch(facecolor=cat_color, edgecolor='black', label=category))

    # 3. 坐标轴和标题设置
    ax.set_yscale('log')  # 风险数据通常使用对数坐标轴
    ax.set_xticks(x_centers)
    ax.set_xticklabels(bin_labels, rotation=30, ha='right', fontsize=12)
    ax.set_xlabel('总受威胁物种数量 (生物多样性分箱)', fontsize=14, weight='bold', labelpad=15)
    ax.set_ylabel('类别归一化暴露指数 (CNEI)', fontsize=14, weight='bold')
    ax.set_title('微塑料暴露强度 (CNEI) vs. 受威胁物种生物多样性', fontsize=18, weight='bold', pad=20)

    # 4. 图例
    ax.legend(handles=legend_patches, loc='upper left', frameon=True, fontsize=12,
              edgecolor='grey', fancybox=True, title='濒危等级', title_fontsize='13')

    # 5. 网格和边框
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5, zorder=0)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"   - 分组统计图已保存至: {output_path}")


# --- 为新的分组箱线图准备数据 ---
print("-> 准备分组箱线图的数据...")

# 1. 确定所有物种数量列 (现在应该是4个)
count_cols = [c for c in lakes_with_metrics.columns if 'species_count_' in c]
print(f"   - 汇总的物种数量列: {count_cols}")

# 2. 计算每个湖泊的 "总受威胁物种数量"
lakes_with_metrics['species_count_Total'] = lakes_with_metrics[count_cols].sum(axis=1)

# 3. 仅保留至少有1个受威胁物种的湖泊
data_to_plot = lakes_with_metrics[lakes_with_metrics['species_count_Total'] > 0].copy()
print(f"   - 发现 {len(data_to_plot)} 个湖泊至少有1个受威胁物种。")

# 4. 对总物种数量进行分箱 (使用qcut创建6个基于分位数的组)
try:
    data_to_plot['species_bin_raw'], bins_edges = pd.qcut(data_to_plot['species_count_Total'], q=6, retbins=True,
                                                          duplicates='drop')

    # 创建可读的标签
    bin_labels = []
    for i in range(len(bins_edges) - 1):
        # 将浮点数边缘转换为整数以提高可读性
        start = int(np.floor(bins_edges[i]))
        end = int(np.ceil(bins_edges[i + 1]))
        if i == 0:
            start = 1  # 最小值从1开始

        if start == end:
            bin_labels.append(f'{start}')
        else:
            bin_labels.append(f'{start}-{end}')

    # 重新应用可读标签
    data_to_plot['species_bin'] = pd.qcut(data_to_plot['species_count_Total'], q=len(bin_labels), labels=bin_labels,
                                          duplicates='drop')
    print(f"   - 已创建生物多样性分箱 (标签): {bin_labels}")

except ValueError as e:
    print(f"[错误] 分箱失败: {e}. 可能是数据分布问题。退出统计绘图。")
    # 如果分箱失败，后续绘图将无法进行
    data_to_plot = pd.DataFrame()  # 置空以跳过绘图

if not data_to_plot.empty:
    # 5. Melt CNEI数据以进行绘图
    cnei_cols = ['CNEI_Near', 'CNEI_Vulnerable', 'CNEI_Endangered', 'CNEI_Critically']
    melted_data = data_to_plot.melt(
        id_vars=['hylak_id', 'species_bin'],
        value_vars=cnei_cols,
        var_name='Category',
        value_name='CNEI'
    )

    # 6. 清理类别名称并过滤掉CNEI为0的记录 (这些湖泊没有该类别的物种)
    melted_data['Category'] = melted_data['Category'].replace({
        'CNEI_Near': 'NT',
        'CNEI_Vulnerable': 'VU',
        'CNEI_Endangered': 'EN',
        'CNEI_Critically': 'CR'
    })
    melted_data = melted_data[melted_data['CNEI'] > 0]

    # 7. 定义类别顺序和颜色
    category_order = ['NT', 'VU', 'EN', 'CR']
    # 使用一个适合分类且色觉友好的调色板
    category_colors = {'NT': '#00A08A', 'VU': '#F2A900', 'EN': '#FF6347', 'CR': '#A50026'}

    # 8. 调用新的绘图函数
    plot_grouped_risk_boxplot(
        data_melted=melted_data,
        bin_labels=bin_labels,
        cat_order=category_order,
        colors_dict=category_colors,
        output_path=os.path.join(OUTPUT_DIR, "Stat_GroupedBoxPlot_CNEI_vs_Biodiversity.pdf")
    )

print("\n==========================================================")
print("  第二阶段所有分析和制图任务(V2版)成功完成！")
print(f"  所有输出文件已保存至: {OUTPUT_DIR}")
print("==========================================================")