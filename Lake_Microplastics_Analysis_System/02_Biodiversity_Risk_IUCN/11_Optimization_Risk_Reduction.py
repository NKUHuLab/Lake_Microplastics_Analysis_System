import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings


def main_statistical_analysis_spatial():
    """
    主函数：
    1. 从分区文件加载Region/Income。
    2. 使用空间连接链接分区/Region/Income 和 WRS。
    3. (图1) [V9 更新] 绘制WRS按分区的箱形图 (线性刻度, 隐藏离群点)。
    4. (图2&3) 绘制WRS总和按Region/Income分类的条形图。
    """
    print("--- [Initialization] Starting Enhanced Partition Risk Reduction Analysis (V9 - Linear Scale, No Fliers) ---")

    # --- 1. 路径和常量定义 ---
    DATA_SOURCE_DIR = r"E:\lake-MP-ani\draw\Risk_Reduction_Delta_Maps"
    DELTA_METRICS_CSV = os.path.join(DATA_SOURCE_DIR, "lakes_risk_reduction_delta_metrics.csv")
    PARTITION_CSV = r"E:\lake-MP-W\draw\11_Geographic_Clustering\ALL_points_annotated_with_cluster_id.csv"
    MP_LAKES_SHP_PATH = r"E:\lake-MP-W\data\generated_shp\predicted_lakes_2022.shp"
    OUTPUT_DIR = r"E:\lake-MP-W\draw\03_SHAP_Analysis"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "partition_WRS_statistical_analysis_Boxplot_NoFliers.pdf")  # 新文件名
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 风险计算常量
    IUCN_WEIGHTS = {'Critically': 50, 'Endangered': 25, 'Vulnerable': 10, 'Near': 5}
    CNEI_REDUCTION_COLS = {
        'Critically': 'CNEI_REDUCTION_Critically', 'Endangered': 'CNEI_REDUCTION_Endangered',
        'Vulnerable': 'CNEI_REDUCTION_Vulnerable', 'Near': 'CNEI_REDUCTION_Near'
    }

    # 分区规则
    partition_map = {
        16: 'a', 0: 'b', 12: 'b', 7: 'c', 11: 'd', 17: 'd',
        4: 'e', 13: 'f', 9: 'g'
    }
    partition_order = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'Unpartitioned']

    # --- 2. 全局绘图设置 (确保字体可编辑) ---
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    sns.set_theme(style="whitegrid", font="Arial")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        from shapely.errors import ShapelyWarning
        warnings.filterwarnings("ignore", category=ShapelyWarning)
    except ImportError:
        pass

    print(f"--- [Step 1/5] Loading All Data Sources ---")
    try:
        df_metrics = pd.read_csv(DELTA_METRICS_CSV)
        print(f"   - Loaded metrics: {DELTA_METRICS_CSV}")

        df_partitions_csv = pd.read_csv(PARTITION_CSV, low_memory=False)
        print(f"   - Loaded partitions CSV: {PARTITION_CSV}")

        gdf_lakes = gpd.read_file(MP_LAKES_SHP_PATH, engine="pyogrio", use_arrow=True)
        print(f"   - Loaded lakes Shapefile (Bridge): {MP_LAKES_SHP_PATH} (using pyogrio+arrow)")

    except Exception as e:
        print(f"❌ 致命错误: 加载文件失败: {e}")
        sys.exit()

    # --- 3. 检查额外列 (来自 df_partitions_csv) ---
    required_extra_cols = ['Region', 'income']
    cols_to_keep_from_partition = ['Partition', 'geometry']
    HAS_EXTRA_COLS = True

    for col in required_extra_cols:
        if col not in df_partitions_csv.columns:
            print(f"⚠️ 警告: 在分区文件 (PARTITION_CSV) 中未找到列 '{col}'。将跳过此项分析。")
            HAS_EXTRA_COLS = False
        else:
            cols_to_keep_from_partition.append(col)

    print(f"--- [Step 2/5] Processing Data and Calculating WRS ---")

    # 步骤 3a: 计算 WRS
    df_metrics['WRS'] = (
            df_metrics[CNEI_REDUCTION_COLS['Critically']].fillna(0) * IUCN_WEIGHTS['Critically'] +
            df_metrics[CNEI_REDUCTION_COLS['Endangered']].fillna(0) * IUCN_WEIGHTS['Endangered'] +
            df_metrics[CNEI_REDUCTION_COLS['Vulnerable']].fillna(0) * IUCN_WEIGHTS['Vulnerable'] +
            df_metrics[CNEI_REDUCTION_COLS['Near']].fillna(0) * IUCN_WEIGHTS['Near']
    )
    df_risk_analysis = df_metrics[['hylak_id', 'WRS']].copy()

    # 步骤 3b: 准备湖泊多边形 (合并 WRS)
    gdf_lakes.rename(columns={'FID': 'hylak_id'}, inplace=True)
    gdf_lakes_with_risk = gdf_lakes[['hylak_id', 'geometry']].merge(df_risk_analysis, on='hylak_id', how='left')
    gdf_lakes_with_risk['WRS'] = gdf_lakes_with_risk['WRS'].fillna(0)

    if gdf_lakes_with_risk.crs is None:
        gdf_lakes_with_risk.set_crs("EPSG:4326", inplace=True, allow_override=True)
    else:
        gdf_lakes_with_risk = gdf_lakes_with_risk.to_crs("EPSG:4326")

    # 步骤 3c: 准备分区点 (携带额外元数据)
    df_partitions_csv['Partition'] = df_partitions_csv['hull_cluster_id'].map(partition_map)
    df_partitions_csv['Partition'].fillna('Unpartitioned', inplace=True)
    gdf_partition_points = gpd.GeoDataFrame(
        df_partitions_csv, geometry=gpd.points_from_xy(df_partitions_csv.lon, df_partitions_csv.lat), crs="EPSG:4326"
    )
    gdf_partition_lookup = gdf_partition_points[list(set(cols_to_keep_from_partition))].copy()

    print("   - WRS, Lake Polygons, and Partition Points (with metadata) prepared.")

    # --- 4. 空间连接 ---
    print(f"--- [Step 3/5] Performing Spatial Join ---")

    gdf_merged_spatial = gpd.sjoin(
        gdf_lakes_with_risk,
        gdf_partition_lookup,
        how='left',
        predicate='contains'
    )
    gdf_merged_spatial = gdf_merged_spatial.drop_duplicates(subset=['hylak_id'])
    gdf_merged_spatial['Partition'].fillna('Unpartitioned', inplace=True)
    print(f"   - Spatial Join complete. Final dataset linked.")

    # 步骤 4d: 准备绘图数据
    df_plot_data_dist = gdf_merged_spatial.copy()
    df_plot_data_agg = gdf_merged_spatial[gdf_merged_spatial['WRS'] > 0].copy()

    print(f"   - Total lakes with WRS > 0 (for aggregation): {len(df_plot_data_agg)}")

    if df_plot_data_agg.empty and HAS_EXTRA_COLS:
        print("❌ 警告: 没有 WRS > 0 的数据, 将跳过聚合图表。")
        HAS_EXTRA_COLS = False

        # --- 5. 动态设置绘图布局 ---
    if HAS_EXTRA_COLS:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 27))
        ax1, ax2, ax3 = axes.flatten()
        print(f"--- [Step 4/5] Generating 3 plots (Partition, Income, Region) ---")
    else:
        fig, ax1 = plt.subplots(figsize=(16, 9))
        print(f"--- [Step 4/5] Generating 1 plot (Partition only) ---")
        ax2, ax3 = None, None

    # --- 6. 绘图: 图1 (分区的箱形图) ---
    print("   - Plotting Boxplot (Partition)...")
    sns.boxplot(
        data=df_plot_data_dist,
        x='Partition',
        y='WRS',
        order=partition_order,
        palette='YlGnBu',
        ax=ax1,
        showfliers=False  # <--- [关键修改] 按照您的要求, 不再绘制离群点
    )

    # 保持严格的线性刻度
    ax1.set_yscale('linear')

    # 修改标题和Y轴标签以反映内容的变化
    ax1.set_title('Distribution of WRS by Partition (Main Distribution Only)',
                  fontsize=20, weight='bold', pad=20, fontname='Arial')
    ax1.set_ylabel('WRS (Linear Scale - Outliers Not Shown)',
                   fontsize=16, fontname='Arial', labelpad=15)
    ax1.set_xlabel('Geographic Partition', fontsize=16, fontname='Arial', labelpad=15)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=12)

    # --- 7. 绘图: 图2 & 3 (条形图不受影响) ---
    if HAS_EXTRA_COLS:
        print("   - Plotting Barplot (Income)...")
        income_agg = df_plot_data_agg.groupby('income')['WRS'].sum().sort_values(ascending=False).reset_index()
        sns.barplot(
            data=income_agg, x='WRS', y='income',
            ax=ax2, palette='Blues_r', orient='h'
        )
        ax2.set_title('Total Weighted Risk Reduction (WRS) by Income Group',
                      fontsize=20, weight='bold', pad=20, fontname='Arial')
        ax2.set_xlabel('Total WRS (Sum of Positive Reductions)', fontsize=16, fontname='Arial', labelpad=15)
        ax2.set_ylabel('Income Group', fontsize=16, fontname='Arial')
        ax2.tick_params(labelsize=12)

        print("   - Plotting Barplot (Region)...")
        region_agg = df_plot_data_agg.groupby('Region')['WRS'].sum().sort_values(ascending=False).reset_index()
        sns.barplot(
            data=region_agg, x='WRS', y='Region',
            ax=ax3, palette='Greens_r', orient='h'
        )
        ax3.set_title('Total Weighted Risk Reduction (WRS) by World Bank Region',
                      fontsize=20, weight='bold', pad=20, fontname='Arial')
        ax3.set_xlabel('Total WRS (Sum of Positive Reductions)', fontsize=16, fontname='Arial', labelpad=15)
        ax3.set_ylabel('Region', fontsize=16, fontname='Arial')
        ax3.tick_params(labelsize=12)

    # --- 8. 保存 ---
    print(f"--- [Step 5/5] Saving Final PDF Output ---")
    plt.tight_layout()

    try:
        plt.savefig(OUTPUT_FILE, format='pdf', bbox_inches='tight')
        print(f"✅ 成功! 复合分析图表已保存到: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")

    plt.close(fig)


# --- 脚本执行 ---
if __name__ == "__main__":
    main_statistical_analysis_spatial()