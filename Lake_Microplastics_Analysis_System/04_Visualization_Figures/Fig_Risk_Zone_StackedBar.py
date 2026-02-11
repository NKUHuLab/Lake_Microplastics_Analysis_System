import pandas as pd
import matplotlib.pyplot as plt
import sys
import os  # 导入 os 模块来处理文件路径和创建文件夹


def plot_stacked_chart(ax, data_df, global_color_palette, chart_title, bar_width=0.7):
    """
    一个可重用的函数，用于在指定的 matplotlib axis 上绘制100%堆叠柱状图。

    Args:
        ax (matplotlib.axes.Axes): 要绘制的子图坐标轴。
        data_df (pd.DataFrame): 已经过百分比计算和排序的数据框。
        global_color_palette (list): 用于聚类的全局颜色列表。
        chart_title (str): 该子图的标题。
        bar_width (float): 柱状图的宽度（小于1可增加间距）。
    """

    if data_df.empty:
        ax.text(0.5, 0.5, 'No data available for this group.',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(chart_title, fontsize=20, pad=20, fontname='Arial', weight='bold')
        ax.set_axis_off()
        return

    # --- 动态颜色映射 ---
    # 确保颜色始终一致地映射到 cluster ID
    # [修改] 从df的列名中获取聚类ID，这些ID应该是数字
    unique_clusters_in_data = sorted([col for col in data_df.columns if isinstance(col, (int, float))])

    cluster_color_map = {
        cluster_id: global_color_palette[i % len(global_color_palette)]
        for i, cluster_id in enumerate(unique_clusters_in_data)
    }
    # [修改] 只绘制数值型的列（聚类ID列）
    plot_colors = [cluster_color_map.get(cluster_id, '#808080') for cluster_id in unique_clusters_in_data]

    # --- 绘制堆叠柱状图 ---
    data_df[unique_clusters_in_data].plot(
        kind='bar',
        stacked=True,
        color=plot_colors,
        ax=ax,
        width=bar_width,  # 使用传入的柱体宽度
        legend=False  # 图例将在 Figure 级别统一添加
    )

    # --- 设置美学风格 (来自您的示例) ---
    ax.set_title(chart_title, fontsize=20, pad=20, fontname='Arial', weight='bold')
    ax.set_ylabel('Cluster Percentage (%)', fontsize=16, labelpad=15, fontname='Arial')
    ax.set_xlabel('')  # X轴标签在子图级别留空

    # 设置X轴刻度标签
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14, fontname='Arial')

    # 设置Y轴刻度
    ax.tick_params(axis='y', labelsize=12)

    # 移除顶部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加水平网格线
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)


def main():
    """
    主函数：加载数据、处理分区、并将高/低风险图表分别绘制到两个子图中。
    """

    # --- 1. 全局配置 ---
    file_path = r"E:\lake-MP-W\draw\11_Geographic_Clustering\ALL_points_annotated_with_cluster_id.csv"

    # [新需求] 指定输出目录和文件名
    output_dir = r"E:\lake-MP-W\draw\03_SHAP_Analysis"
    output_filename = os.path.join(output_dir, "partition_cluster_distribution_split.pdf")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 设置字体
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # 颜色列表
    map_script_colors = [
        "#4885c1", "#c96734", "#6aa4bb", "#e4cb3a",
        "#8c5374", "#aab381", "#ffc000", "#ae3a4e",
        "#afe1af", "#cad675", "#8dc2b5", "#6a6c9b"
    ]

    # --- 2. [已按要求修改] 定义分区规则 ---
    partition_map = {
        16: 'a',
        0: 'b', 12: 'b', 14: 'b',
        7: 'c',
        11: 'd', 17: 'd',
        4: 'e',
        13: 'f', 18: 'f',
        9: 'g'
    }

    # 定义基础分区顺序
    partition_order_keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    # --- 3. 加载和预处理数据 ---
    try:
        # 添加 low_memory=False 解决 DtypeWarning
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"❌ 致命错误: 在 '{file_path}' 找不到数据文件。程序退出。")
        sys.exit()

    # 步骤 3a: 应用分区映射
    # 注意：这里的 'hull_cluster_id' 包含了点所在的聚类区域ID
    df['Partition'] = df['hull_cluster_id'].map(partition_map)

    # 步骤 3b: [已按要求修改] 创建用于X轴的类别列，使用 'point_status'
    def create_x_category_from_status(row):
        status = row['point_status']
        partition_label = row['Partition']

        if status == 'Point_Outside_Hull' or pd.isna(partition_label):
            return 'Unpartitioned'
        elif status == 'High_Pollution_Point_In_Hull':
            return f"Partition {partition_label} (High Risk)"
        elif status == 'Low_Pollution_Point_In_Hull':
            return f"Partition {partition_label} (Low Risk)"
        else:
            return 'Unpartitioned'  # 其他意外情况也归为 Unpartitioned

    df['x_category'] = df.apply(create_x_category_from_status, axis=1)

    # --- 4. 聚合数据 (计算百分比) ---
    # 注意：这里的 'df_for_clustering' 中的 'cluster' 列代表点本身的特征聚类
    # 我们需要确保数据中同时存在 'x_category' 和 'cluster' 列
    if 'cluster' not in df.columns:
        # 假设如果 'cluster' 不存在，则使用 'hull_cluster_id' 作为替代
        # 更好的做法是确认哪个 'cluster' 是正确的
        # 根据之前的脚本，'cluster' 是特征聚类，而 'hull_cluster_id' 是地理聚类
        # 我们假设堆叠的部分是 '特征聚类'
        print("⚠️ 警告: 'cluster' 列不存在，将使用 'hull_cluster_id' 进行堆叠。请确认这是否是预期行为。")
        df.rename(columns={'hull_cluster_id': 'cluster'}, inplace=True)

    # 过滤掉没有有效聚类ID的行以进行制表
    df_for_crosstab = df.dropna(subset=['x_category', 'cluster'])
    df_for_crosstab['cluster'] = df_for_crosstab['cluster'].astype(int)

    category_cluster_counts = pd.crosstab(df_for_crosstab['x_category'], df_for_crosstab['cluster'])
    category_cluster_percentage = category_cluster_counts.div(category_cluster_counts.sum(axis=1), axis=0) * 100

    # 4b. 定义所有可能的类别（用于初始排序）
    ordered_categories = []
    for p in partition_order_keys:
        ordered_categories.append(f"Partition {p} (Low Risk)")
        ordered_categories.append(f"Partition {p} (High Risk)")
    ordered_categories.append('Unpartitioned')

    # 4c. 获取完整的排序数据
    df_processed = category_cluster_percentage.reindex(ordered_categories).dropna(how='all')

    if df_processed.empty:
        print("❌ 错误: 数据处理后没有可绘制的内容。")
        return

    # --- 5. 拆分数据用于两个图表 ---

    # 图1: 低风险 (包含 'Unpartitioned')
    data_low_risk = df_processed[
        df_processed.index.str.contains("Low Risk") | (df_processed.index == "Unpartitioned")
        ]
    # 清理索引标签
    data_low_risk.index = data_low_risk.index.str.replace(" (Low Risk)", "", regex=False)

    # 图2: 高风险 (不含 'Unpartitioned')
    data_high_risk = df_processed[df_processed.index.str.contains("High Risk")]
    # 清理索引标签
    data_high_risk.index = data_high_risk.index.str.replace(" (High Risk)", "", regex=False)

    # 5b. 定义最终的X轴排序 (确保两张图对齐)
    final_x_order = [f"Partition {p}" for p in partition_order_keys] + ['Unpartitioned']

    # 5c. 按最终顺序重新索引两个数据框
    data_low_risk = data_low_risk.reindex(final_x_order).dropna(how='all')
    data_high_risk = data_high_risk.reindex(final_x_order).dropna(
        how='all')  # 这将自动丢弃 'Unpartitioned' 行（因为它在高风险DF中是全NaN）

    # --- 6. 绘图 (设置两个子图) ---

    # 根据两个图的内容计算宽度比例
    num_bars_1 = len(data_low_risk.index) if not data_low_risk.empty else 1
    num_bars_2 = len(data_high_risk.index) if not data_high_risk.empty else 1

    # 创建1行2列的子图，figsize更宽更扁 (24宽, 8高)
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(24, 6),
        gridspec_kw={'width_ratios': [num_bars_1, num_bars_2]}  # 按内容分配空间
    )

    bar_width = 0.7

    # 绘制图1 (低风险)
    plot_stacked_chart(
        ax=ax1,
        data_df=data_low_risk,
        global_color_palette=map_script_colors,
        chart_title="Cluster Distribution (Low Risk & Unpartitioned)",
        bar_width=bar_width
    )

    # 绘制图2 (高风险)
    plot_stacked_chart(
        ax=ax2,
        data_df=data_high_risk,
        global_color_palette=map_script_colors,
        chart_title="Cluster Distribution (High Risk)",
        bar_width=bar_width
    )

    # --- 7. 添加共享图例 ---
    # 从 ax1 和 ax2 合并所有可能的图例项
    handles1, labels1 = ax1.get_legend_handles_labels() if not data_low_risk.empty else ([], [])
    handles2, labels2 = ax2.get_legend_handles_labels() if not data_high_risk.empty else ([], [])

    # 合并并去重
    all_labels = dict(zip(labels1, handles1))
    all_labels.update(dict(zip(labels2, handles2)))

    sorted_labels = sorted(all_labels.keys(), key=lambda x: int(x))
    sorted_handles = [all_labels[label] for label in sorted_labels]

    fig.legend(
        sorted_handles,
        [f'Cluster {label}' for label in sorted_labels],
        title='Feature Cluster',
        loc='upper right',
        bbox_to_anchor=(0.99, 0.95),  # 调整位置以适应新布局
        prop={'family': 'Arial', 'size': 12}
    )
    plt.setp(fig.legends[0].get_title(), fontname='Arial', fontsize=14)

    # --- 8. 保存并关闭 ---
    fig.suptitle('Cluster Distribution by Geographic Partition and Risk Status',
                 fontsize=24, fontname='Arial', weight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # 调整布局为图例留出右侧空间

    try:
        plt.savefig(output_filename, format='pdf', bbox_inches='tight')
        print(f"✅ 成功! 分区图表已保存到: '{output_filename}'")
    except Exception as e:
        print(f"❌ 保存 '{output_filename}' 时出错: {e}")

    plt.close(fig)


# --- 执行主函数 ---
if __name__ == '__main__':
    main()