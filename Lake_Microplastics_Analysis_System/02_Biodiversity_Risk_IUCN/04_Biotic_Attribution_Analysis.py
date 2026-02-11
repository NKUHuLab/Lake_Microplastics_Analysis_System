import os
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import time

# --- 0. 全局配置与初始化 ---
print("--- [初始化] 终极可靠版 V10 (已实施更稳健的列名冲突解决方案) ---")

# --- Matplotlib and Warnings Configuration (No changes) ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# --- 1. 路径和参数配置 (No changes) ---
PROJECT_ROOT_ANI = r"E:\lake-MP-ani"
PROJECT_ROOT_W = r"E:\lake-MP-W"
MAIN_OUTPUT_DIR = os.path.join(PROJECT_ROOT_ANI, "draw", "111Biological_Attribution_Analysis_SJoin_Fast")
DATA_SUMMARY_DIR = os.path.join(MAIN_OUTPUT_DIR, "data_summaries")
PLOT_OUTPUT_DIR = os.path.join(MAIN_OUTPUT_DIR, "plots")
os.makedirs(DATA_SUMMARY_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
print(f"-> 所有输出将被保存到: {MAIN_OUTPUT_DIR}")
SPECIES_GPKG_PATH = os.path.join(PROJECT_ROOT_ANI, "data", "IUCN_consolidated", "all_species_classified.gpkg")
SPECIES_ATTRIBUTES_CSV = r"E:\lake-MP-W\dataset\IUCN\bio\matched_species_output.csv"
MP_LAKES_SHP_PATH = os.path.join(PROJECT_ROOT_W, "data", "generated_shp", "predicted_lakes_2022.shp")
REDUCTION_POTENTIAL_CSV = os.path.join(PROJECT_ROOT_W, "data", "opt", "data", "processed_output", "Ychange.csv")
TAXONOMIC_SUMMARY_CSV = os.path.join(DATA_SUMMARY_DIR, "summary_analysis_1_taxonomic.csv")
LIFECYCLE_SUMMARY_CSV = os.path.join(DATA_SUMMARY_DIR, "summary_analysis_2_lifecycle.csv")
HABITAT_SUMMARY_CSV = os.path.join(DATA_SUMMARY_DIR, "summary_analysis_3_habitat_total.csv")
HABITAT_AVG_RISK_CSV = os.path.join(DATA_SUMMARY_DIR, "summary_analysis_3_habitat_avg_per_species.csv")
ORIGIN_SUMMARY_CSV = os.path.join(DATA_SUMMARY_DIR, "summary_analysis_4_origin.csv")
INTERMEDIATE_SJOIN_GPKG = os.path.join(DATA_SUMMARY_DIR, "intermediate_lake_species_intersections.gpkg")
TAXONOMIC_PLOT_PDF = os.path.join(PLOT_OUTPUT_DIR, "Plot_1_Taxonomic_Risk_Top15.pdf")
LIFECYCLE_PLOT_PDF = os.path.join(PLOT_OUTPUT_DIR, "Plot_2_Lifecycle_Risk_Comparison.pdf")
HABITAT_TOTAL_PLOT_PDF = os.path.join(PLOT_OUTPUT_DIR, "Plot_3a_Habitat_Total_Risk_Portfolio.pdf")
HABITAT_AVG_PLOT_PDF = os.path.join(PLOT_OUTPUT_DIR, "Plot_3b_Habitat_Average_Risk_per_Species.pdf")
ORIGIN_PLOT_PDF = os.path.join(PLOT_OUTPUT_DIR, "Plot_4_Origin_Risk_Comparison.pdf")
EQUAL_AREA_PROJ = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
LAKE_ID_COLUMN = 'FID'
TARGET_CATEGORIES = ['Critically Endangered', 'Endangered', 'Vulnerable', 'Near Threatened']
ORIGIN_MAP = {1: 'Native', 2: 'Reintroduced', 3: 'Introduced', 4: 'Vagrant', 5: 'Origin Uncertain',
              6: 'Assisted Colonisation'}


# --- 2. 辅助与绘图函数 (No changes) ---
def calculate_bvi(row):
    """Calculates the Biological Vulnerability Index (BVI)."""
    W_season = 1.0
    if 'seasonal' in row and pd.notna(row['seasonal']) and row['seasonal'] == 2:
        W_season = 1.5
    return W_season


def plot_taxonomic_summary(df, output_pdf, top_n=15):
    """Plots taxonomic summary results."""
    print(f"-> [绘图 B1] 正在绘制分类学风险图 (Top {top_n})...")
    if df.empty:
        print("   - [跳过] 分类学数据为空。")
        return
    df_plot = df.nlargest(top_n, 'Total_Baseline_Risk_ACR').sort_values(by='Total_Baseline_Risk_ACR', ascending=True)
    labels = df_plot['order_'] + " (" + df_plot['class'].astype(str).str.slice(0, 4) + ".)"
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.barh(labels, df_plot['Total_Baseline_Risk_ACR'], color='#D9534F', alpha=0.7, label='Total Baseline Risk (ACR)')
    ax.barh(labels, df_plot['Total_Risk_Reduction_Delta'], color='#5CB85C', alpha=0.9,
            label='Risk Reduction Benefit (Delta)')
    ax.set_xlabel('Cumulative Risk (ACR / Delta)', fontsize=12)
    ax.set_ylabel('Taxonomic Order', fontsize=12)
    ax.set_title(f'Top {top_n} Threatened Orders by Baseline Risk & Protection Benefit', fontsize=14, weight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf')
    plt.close(fig)
    print(f"   - [成功] 已保存绘图至: {output_pdf}")


# --- Other plotting functions (no changes) ---
def plot_lifecycle_summary(df, output_pdf):
    """Plots lifecycle summary results."""
    print(f"-> [绘图 B2] 正在绘制生命周期风险图...")
    if df.empty: print("   - [跳过] 生命周期数据为空。"); return
    df_melted = df.melt(id_vars='lifecycle_group', value_vars=['Total_Baseline_Risk_ACR', 'Total_Risk_Reduction_Delta'],
                        var_name='Metric', value_name='Value')
    df_melted['Metric'] = df_melted['Metric'].replace(
        {'Total_Baseline_Risk_ACR': 'Baseline Risk', 'Total_Risk_Reduction_Delta': 'Reduction Benefit'})
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.barplot(x='lifecycle_group', y='Value', hue='Metric', data=df_melted,
                palette={'Baseline Risk': '#D9534F', 'Reduction Benefit': '#5CB85C'}, ax=ax)
    ax.set_title('Risk Comparison: Breeding (Seasonal=2) vs. Other Areas', fontsize=14, weight='bold')
    ax.set_ylabel('Cumulative Risk (ACR / Delta)', fontsize=12);
    ax.set_xlabel('Habitat Type', fontsize=12)
    ax.legend(title='Metric');
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0));
    plt.tight_layout();
    plt.savefig(output_pdf, format='pdf');
    plt.close(fig)
    print(f"   - [成功] 已保存绘图至: {output_pdf}")


def plot_habitat_summary(df_total, df_avg, total_pdf, avg_pdf):
    """Plots habitat dependency summary results."""
    print(f"-> [绘图 B3] 正在绘制栖息地依赖性风险图...")
    if not df_total.empty:
        df_total_melted = df_total.melt(id_vars='habitat_dependency',
                                        value_vars=['Total_Baseline_Risk_ACR', 'Total_Risk_Reduction_Delta'],
                                        var_name='Metric', value_name='Value')
        df_total_melted['Metric'] = df_total_melted['Metric'].replace(
            {'Total_Baseline_Risk_ACR': 'Baseline Risk', 'Total_Risk_Reduction_Delta': 'Reduction Benefit'})
        fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
        sns.barplot(x='habitat_dependency', y='Value', hue='Metric', data=df_total_melted,
                    palette={'Baseline Risk': '#0275D8', 'Reduction Benefit': '#F0AD4E'}, ax=ax1)
        ax1.set_title('Total Risk Portfolio: Pure Freshwater vs. Mixed-Habitat Species', fontsize=14, weight='bold')
        ax1.set_ylabel('Cumulative Risk (ACR / Delta)', fontsize=12);
        ax1.set_xlabel('Habitat Dependency Type', fontsize=12)
        ax1.legend(title='Metric');
        ax1.grid(axis='y', linestyle='--', alpha=0.6);
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0));
        plt.tight_layout();
        plt.savefig(total_pdf, format='pdf');
        plt.close(fig1)
        print(f"   - [成功] 已保存绘图 (3a) 至: {total_pdf}")
    else:
        print("   - [跳过] 栖息地总风险数据为空。")
    if not df_avg.empty:
        fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=300)
        sns.barplot(x='habitat_dependency', y='Average_Global_ACR_per_Species', data=df_avg,
                    palette=['#0275D8', '#5BC0DE'], ax=ax2)
        ax2.set_title('Average Risk per Species (Pure Freshwater vs. Mixed-Habitat)', fontsize=14, weight='bold')
        ax2.set_ylabel('Average Global Cumulative Risk per Species (Avg. ACR)', fontsize=10);
        ax2.set_xlabel('Habitat Dependency Type', fontsize=10)
        ax2.grid(axis='y', linestyle='--', alpha=0.6);
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0));
        plt.tight_layout();
        plt.savefig(avg_pdf, format='pdf');
        plt.close(fig2)
        print(f"   - [成功] 已保存绘图 (3b) 至: {avg_pdf}")
    else:
        print("   - [跳过] 栖息地平均风险数据为空。")


def plot_origin_summary(df, output_pdf):
    """Plots species origin summary results."""
    print(f"-> [绘图 B4] 正在绘制物种来源风险图...")
    if df.empty: print("   - [跳过] 物种来源数据为空。"); return
    df_plot = df[df['origin_label'] != 'Vagrant'].copy()
    df_melted = df_plot.melt(id_vars='origin_label',
                             value_vars=['Total_Baseline_Risk_ACR', 'Total_Risk_Reduction_Delta'], var_name='Metric',
                             value_name='Value')
    df_melted['Metric'] = df_melted['Metric'].replace(
        {'Total_Baseline_Risk_ACR': 'Baseline Risk', 'Total_Risk_Reduction_Delta': 'Reduction Benefit'})
    order = df_plot.sort_values(by='Total_Baseline_Risk_ACR', ascending=False)['origin_label'].unique()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    sns.barplot(x='origin_label', y='Value', hue='Metric', data=df_melted,
                palette={'Baseline Risk': '#A94442', 'Reduction Benefit': '#3C763D'}, ax=ax, order=order)
    ax.set_title('Risk Comparison by Species Origin (Threatened Species Only)', fontsize=14, weight='bold')
    ax.set_ylabel('Cumulative Risk (ACR / Delta)', fontsize=12);
    ax.set_xlabel('Species Origin Category', fontsize=12)
    ax.legend(title='Metric');
    ax.grid(axis='y', linestyle='--', alpha=0.6);
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(rotation=15, ha='right');
    plt.tight_layout();
    plt.savefig(output_pdf, format='pdf');
    plt.close(fig)
    print(f"   - [成功] 已保存绘图至: {output_pdf}")


# --- 3. 主执行函数 ---
def run_full_pipeline():
    try:
        # --- [步骤 A1] 加载和预处理物种数据 (生物属性合并) ---
        print("\n--- [步骤 A1] 加载所有基础数据 (并合并生物属性) ---")

        print("-> 加载物种几何数据 (GPKG)...")
        gdf_species_raw = gpd.read_file(SPECIES_GPKG_PATH, engine="pyogrio", use_arrow=True)

        # --- [关键修复 V10] 实施稳健的列名标准化，防止因同时存在大写和小写列名而崩溃 ---
        print("   - [修复] 正在稳健地标准化列名 (e.g., ID_NO -> id_no)...")
        column_map = {
            'id_no': 'ID_NO',
            'seasonal': 'SEASONAL',
            'origin': 'ORIGIN'
        }
        for lower, upper in column_map.items():
            if lower in gdf_species_raw.columns and upper in gdf_species_raw.columns:
                print(f"     - [警告] 发现了 '{lower}' 和 '{upper}' 两列。将删除 '{upper}' 以避免冲突。")
                del gdf_species_raw[upper]
            elif upper in gdf_species_raw.columns and lower not in gdf_species_raw.columns:
                gdf_species_raw.rename(columns={upper: lower}, inplace=True)

        print("-> [预检] 检查物种GPKG必需字段...")
        required_gpkg_cols = ['seasonal', 'origin', 'id_no', 'redlist_category', 'SCI_NAME']
        missing_gpkg_cols = [col for col in required_gpkg_cols if col not in gdf_species_raw.columns]
        if missing_gpkg_cols:
            raise ValueError(f"物种 GPKG 文件中缺少关键字段: {missing_gpkg_cols}")
        print("   - [检查通过] 所有必需的 GPKG 字段均已找到。")

        print("-> 加载物种属性查找表 (CSV)...")
        df_attributes_raw = pd.read_csv(SPECIES_ATTRIBUTES_CSV)
        # 对CSV文件也应用同样稳健的重命名逻辑
        if 'id_no' in df_attributes_raw.columns and 'ID_NO' in df_attributes_raw.columns:
            del df_attributes_raw['ID_NO']
        elif 'ID_NO' in df_attributes_raw.columns and 'id_no' not in df_attributes_raw.columns:
            df_attributes_raw.rename(columns={'ID_NO': 'id_no'}, inplace=True)

        print("-> [预检] 检查物种属性CSV必需字段...")
        required_csv_cols = ['id_no', 'class', 'order_', 'freshwater', 'marine', 'terrestria']
        missing_csv_cols = [col for col in required_csv_cols if col not in df_attributes_raw.columns]
        if missing_csv_cols:
            raise ValueError(f"属性 CSV 文件中缺少关键字段: {missing_csv_cols}")
        print("   - [检查通过] 所有必需的 CSV 属性字段均已找到。")

        df_attributes_lookup = df_attributes_raw[required_csv_cols].drop_duplicates(subset=['id_no'])
        print(f"   - 成功创建 {len(df_attributes_lookup)} 条唯一的物种属性记录。")

        print("-> 正在将属性合并到几何数据中...")
        # 在合并前，删除gdf_species_raw中与df_attributes_lookup中冲突的列
        cols_from_csv_to_drop = ['class', 'order_', 'freshwater', 'marine', 'terrestria']
        for col in cols_from_csv_to_drop:
            if col in gdf_species_raw.columns:
                del gdf_species_raw[col]

        gdf_species_enriched = gdf_species_raw.merge(df_attributes_lookup, on='id_no', how='left')
        print("   - 属性合并完成。")

        # Filter for threatened species and calculate BVI
        gdf_species_filtered = gdf_species_enriched[
            gdf_species_enriched['redlist_category'].isin(TARGET_CATEGORIES)].copy()
        print("   - 正在计算 BVI (依赖 'seasonal' 列)...")
        gdf_species_filtered['BVI'] = gdf_species_filtered.apply(calculate_bvi, axis=1)
        print(f"   - 筛选出 {len(gdf_species_filtered)} 条受威胁物种记录 (已富集)。")

        # --- [步骤 A2] 基于Ychange.csv构建湖泊情景 (No changes) ---
        print("\n--- [步骤 A2] 构建基线与优化情景 ---")

        print("-> 加载湖泊几何形状 (predicted_lakes_2022.shp)...")
        gdf_lakes_geom = gpd.read_file(MP_LAKES_SHP_PATH, engine="pyogrio", use_arrow=True)
        gdf_lakes_geom.set_crs("EPSG:4326", inplace=True, allow_override=True)
        gdf_lakes_geom = gdf_lakes_geom[[LAKE_ID_COLUMN, 'geometry']].rename(columns={LAKE_ID_COLUMN: 'hylak_id'})
        print(f"   - 成功加载 {len(gdf_lakes_geom)} 个湖泊多边形。")

        print("-> 加载湖泊丰度与优化数据 (Ychange.csv)...")
        abundance_df = pd.read_csv(REDUCTION_POTENTIAL_CSV)

        if 'MPs-ori' not in abundance_df.columns:
            raise ValueError("Ychange.csv 文件中缺少 'MPs-ori' 列。")
        abundance_df.rename(columns={'MPs-ori': 'mp_baseline'}, inplace=True)

        abundance_df = abundance_df[['lon', 'lat', 'mp_baseline', 'change']].dropna(
            subset=['lon', 'lat', 'mp_baseline'])
        points_gdf = gpd.GeoDataFrame(abundance_df, geometry=gpd.points_from_xy(abundance_df.lon, abundance_df.lat),
                                      crs=gdf_lakes_geom.crs)
        print(f"   - 成功加载 {len(points_gdf)} 个有效数据点。")

        print("-> 将丰度数据点空间连接 (sjoin) 到湖泊面...")
        lakes_with_points = gpd.sjoin(gdf_lakes_geom, points_gdf, how="left", predicate="contains")

        print("-> 聚合每个湖泊内的多个数据点 (计算均值)...")
        lake_aggregated_data = lakes_with_points.groupby('hylak_id').agg({
            'mp_baseline': 'mean',
            'change': 'mean'
        })

        gdf_lakes_scenario = gdf_lakes_geom.merge(lake_aggregated_data, on='hylak_id', how='left')

        lakes_without_data_count = gdf_lakes_scenario['mp_baseline'].isna().sum()
        if lakes_without_data_count > 0:
            print(
                f"   - [注意] 有 {lakes_without_data_count} 个湖泊多边形内部没有找到任何数据点。它们的风险值将为NA并被忽略。")

        print("-> 计算优化后的丰度值...")
        gdf_lakes_scenario['change'] = gdf_lakes_scenario['change'].fillna(0).clip(upper=0)
        gdf_lakes_scenario['mp_optimized'] = (gdf_lakes_scenario['mp_baseline'] + gdf_lakes_scenario['change']).clip(
            lower=0)

        gdf_lakes_scenario.dropna(subset=['mp_baseline'], inplace=True)
        print(f"   - 两个情景已创建。共有 {len(gdf_lakes_scenario)} 个湖泊拥有丰度数据并进入下一步分析。")

        # --- [步骤 A3] 执行空间连接 (No changes) ---
        print("\n--- [步骤 A3] 执行空间连接 (高精度简化 + 标准SJoin) ---")
        print("-> 准备用于连接的列...")
        species_cols = ['SCI_NAME', 'redlist_category', 'BVI', 'geometry', 'class', 'order_', 'seasonal', 'freshwater',
                        'marine', 'terrestria', 'origin']
        species_cols_to_use = [col for col in species_cols if col in gdf_species_filtered.columns]
        gdf_species_proj = gdf_species_filtered[species_cols_to_use].to_crs(EQUAL_AREA_PROJ)

        lakes_cols = ['hylak_id', 'mp_baseline', 'mp_optimized', 'geometry']
        gdf_lakes_proj = gdf_lakes_scenario[lakes_cols].to_crs(EQUAL_AREA_PROJ)

        print("-> [最终预检] 检查SJoin所需的所有生物学列是否存在于物种数据中...")
        required_bio_cols_for_sjoin = ['class', 'order_', 'seasonal', 'freshwater', 'marine', 'terrestria', 'origin']
        missing_sjoin_cols = [col for col in required_bio_cols_for_sjoin if col not in gdf_species_proj.columns]
        if missing_sjoin_cols:
            raise ValueError(f"物种数据中缺少执行归因分析所必需的列: {missing_sjoin_cols}")
        print("   - [检查通过] 所有必需的生物学列均已准备好进行空间连接。")

        print("-> [加速策略] 正在以高容差简化几何图形...")
        tolerance_meters = 500
        gdf_species_proj['geometry'] = gdf_species_proj.geometry.simplify(tolerance_meters)
        gdf_lakes_proj['geometry'] = gdf_lakes_proj.geometry.simplify(tolerance_meters)
        print(f"   - 几何简化完成 (容差: {tolerance_meters}米)。")

        print("-> [可靠性策略] 正在执行标准 SJoin (使用内置空间索引优化)...")
        start_sjoin = time.time()
        intersected_gdf = gpd.sjoin(gdf_lakes_proj, gdf_species_proj, how='inner', predicate='intersects')
        end_sjoin = time.time()
        print(f"   - 空间连接完成！耗时: {end_sjoin - start_sjoin:.2f} 秒。")
        print(f"   - 生成了 {len(intersected_gdf)} 条 湖泊-物种 交叉配对记录。")

        # --- 后续风险计算和分析 (No changes) ---
        print("-> 计算 IRS 和 IRS_REDUCTION ...")
        intersected_gdf['IRS_baseline'] = intersected_gdf['mp_baseline'] * intersected_gdf['BVI']
        intersected_gdf['IRS_optimized'] = intersected_gdf['mp_optimized'] * intersected_gdf['BVI']
        intersected_gdf['IRS_REDUCTION'] = intersected_gdf['IRS_baseline'] - intersected_gdf['IRS_optimized']
        print("   - IRS 计算完成。")

        print("\n--- [步骤 A4-A] 保存中间空间连接结果 (GeoPackage)... ---")
        columns_to_save = ['hylak_id', 'SCI_NAME', 'redlist_category', 'class', 'order_', 'seasonal', 'freshwater',
                           'marine', 'terrestria', 'origin', 'BVI', 'IRS_baseline', 'IRS_optimized', 'IRS_REDUCTION',
                           'geometry']
        save_cols_exist = [col for col in columns_to_save if col in intersected_gdf.columns]
        intersected_gdf_wgs84 = intersected_gdf[save_cols_exist].to_crs("EPSG:4326")
        intersected_gdf_wgs84.to_file(INTERMEDIATE_SJOIN_GPKG, driver='GPKG', engine='pyogrio')
        print(f"   - [成功] 已将 {len(intersected_gdf)} 条详细记录保存至: {INTERMEDIATE_SJOIN_GPKG}")

        print("\n--- [步骤 A4-B] 执行生物学归因分析并保存CSV ---")
        taxonomic_summary_df = intersected_gdf.groupby(['class', 'order_']).agg(
            Total_Baseline_Risk_ACR=('IRS_baseline', 'sum'), Total_Risk_Reduction_Delta=('IRS_REDUCTION', 'sum'),
            Unique_Threatened_Species=('SCI_NAME', 'nunique')
        ).reset_index().sort_values(by='Total_Baseline_Risk_ACR', ascending=False)
        taxonomic_summary_df.to_csv(TAXONOMIC_SUMMARY_CSV, index=False, encoding='utf-8-sig')
        print(f"      - [成功] 已保存分类学CSV")

        intersected_gdf['lifecycle_group'] = np.where(intersected_gdf['seasonal'] == 2, 'Breeding Area (seasonal=2)',
                                                      'Other Areas (Resident/Non-Breeding etc.)')
        lifecycle_summary_df = intersected_gdf.groupby('lifecycle_group').agg(
            Total_Baseline_Risk_ACR=('IRS_baseline', 'sum'), Total_Risk_Reduction_Delta=('IRS_REDUCTION', 'sum')
        ).reset_index().sort_values(by='Total_Baseline_Risk_ACR', ascending=False)
        lifecycle_summary_df.to_csv(LIFECYCLE_SUMMARY_CSV, index=False, encoding='utf-8-sig')
        print(f"      - [成功] 已保存生命周期CSV")

        intersected_gdf[['freshwater', 'marine', 'terrestria']] = intersected_gdf[
            ['freshwater', 'marine', 'terrestria']].fillna(False).astype(bool)
        is_pure_fw = (intersected_gdf['freshwater']) & (~intersected_gdf['marine']) & (~intersected_gdf['terrestria'])
        is_mixed_fw = (intersected_gdf['freshwater']) & (intersected_gdf['marine'] | intersected_gdf['terrestria'])
        conditions = [is_pure_fw, is_mixed_fw];
        choices = ['Purely Freshwater', 'Mixed-Habitat (Freshwater+)']
        intersected_gdf['habitat_dependency'] = np.select(conditions, choices, default='Other/Non-Freshwater')
        habitat_analysis_df = intersected_gdf[
            intersected_gdf['habitat_dependency'].isin(['Purely Freshwater', 'Mixed-Habitat (Freshwater+)'])].copy()
        habitat_total_df = habitat_analysis_df.groupby('habitat_dependency').agg(
            Total_Baseline_Risk_ACR=('IRS_baseline', 'sum'), Total_Risk_Reduction_Delta=('IRS_REDUCTION', 'sum'),
            Unique_Threatened_Species=('SCI_NAME', 'nunique')
        ).reset_index()
        habitat_total_df.to_csv(HABITAT_SUMMARY_CSV, index=False, encoding='utf-8-sig')
        print(f"      - [成功] 已保存栖息地总风险CSV")

        species_global_risk_df = habitat_analysis_df.groupby(['habitat_dependency', 'SCI_NAME']).agg(
            Species_Total_Baseline_ACR=('IRS_baseline', 'sum')
        ).reset_index()
        habitat_avg_df = species_global_risk_df.groupby('habitat_dependency')[
            ['Species_Total_Baseline_ACR']].mean().reset_index()
        habitat_avg_df.rename(columns={'Species_Total_Baseline_ACR': 'Average_Global_ACR_per_Species'}, inplace=True)
        habitat_avg_df.to_csv(HABITAT_AVG_RISK_CSV, index=False, encoding='utf-8-sig')
        print(f"      - [成功] 已保存栖息地平均风险CSV")

        intersected_gdf['origin'] = intersected_gdf['origin'].fillna(5)
        intersected_gdf['origin_label'] = intersected_gdf['origin'].map(ORIGIN_MAP).fillna('Other/Unknown')
        origin_summary_df = intersected_gdf.groupby('origin_label').agg(
            Total_Baseline_Risk_ACR=('IRS_baseline', 'sum'), Total_Risk_Reduction_Delta=('IRS_REDUCTION', 'sum'),
            Unique_Threatened_Species=('SCI_NAME', 'nunique')
        ).reset_index().sort_values(by='Total_Baseline_Risk_ACR', ascending=False)
        origin_summary_df.to_csv(ORIGIN_SUMMARY_CSV, index=False, encoding='utf-8-sig')
        print(f"      - [成功] 已保存物种来源CSV")
        print("--- [步骤 A] 计算和CSV保存已全部完成 ---")

        print("\n--- [步骤 B] 开始执行下游绘图任务 ---")
        plot_taxonomic_summary(taxonomic_summary_df, TAXONOMIC_PLOT_PDF, top_n=15)
        plot_lifecycle_summary(lifecycle_summary_df, LIFECYCLE_PLOT_PDF)
        plot_habitat_summary(habitat_total_df, habitat_avg_df, HABITAT_TOTAL_PLOT_PDF, HABITAT_AVG_PLOT_PDF)
        plot_origin_summary(origin_summary_df, ORIGIN_PLOT_PDF)

    except Exception as e:
        print(f"[严重错误] 流程执行失败: {e}")


# --- 4. 脚本执行 ---
if __name__ == "__main__":
    run_full_pipeline()
    print("\n===================================================================")
    print("  脚本执行流程已结束。")
    print(f"  所有数据摘要CSV已保存至: {DATA_SUMMARY_DIR}")
    print(f"  所有分析图表PDF已保存至: {PLOT_OUTPUT_DIR}")
    print(f"  [重要] 详细的中间对应关系数据已保存至: {INTERMEDIATE_SJOIN_GPKG}")
    print("===================================================================")