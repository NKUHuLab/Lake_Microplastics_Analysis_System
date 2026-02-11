# -*- coding: utf-8 -*-
# 02_visualize_from_csv.py
#
# 功能:
# 1. 从CSV加载预计算数据。
# 2. 生成四份出版级图表。
# 3. (最终修改) 图4采用IQR方法剔除异常值后，绘制对称、无截断的误差线图，以更真实地反映数据变异性。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 全局配置 ---

# --- 核心文件路径 ---
PROJECT_ROOT = r"E:\lake-MP-W"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "draw", "Publication_Figures_Combined")
GROUPED_CSV_PATH = os.path.join(OUTPUT_DIR, "grouped_percentage_contributions.csv")
INDIVIDUAL_CSV_PATH = os.path.join(OUTPUT_DIR, "individual_feature_contributions.csv")

# --- 特征分组定义 (用于图例、颜色和新图表的分类) ---
ALL_FEATURES = [
    'Lake_area', 'Shore_dev', 'Vol_total', 'Res_time',
    'Total_POP_SERVED', 'Average_DF', 'Primary_Waste_Discharge',
    'Secondary_Waste_Discharge', 'Advanced_Waste_Discharge',
    'RSE_paved', 'RSE_gravel', 'RSE_other', 'prec', 'emis_tyre_TSP_HEG',
    'emis_brake_TSP_HEG', 'PM2_5', 'PM10', 'Mismanaged', 'Total_Plast',
    'fish_gdp_sqkm', 'Cultivated_land', 'Artificial_surface'
]
FEATURE_GROUPS = {
    'Atmospheric Input': ['emis_tyre_TSP_HEG', 'emis_brake_TSP_HEG', 'PM2_5', 'PM10'],
    'Hydrological Input': ['Total_POP_SERVED', 'Average_DF', 'Primary_Waste_Discharge', 'Secondary_Waste_Discharge',
                           'Advanced_Waste_Discharge'],
    'Watershed Non-Point Pollution': ['Mismanaged', 'Total_Plast', 'fish_gdp_sqkm', 'Cultivated_land',
                                      'Artificial_surface']
}
assigned_features = {f for features in FEATURE_GROUPS.values() for f in features}
FEATURE_GROUPS['Other'] = [f for f in ALL_FEATURES if f not in assigned_features]

COLORS = ['#c96734', '#4885c1', '#aab381', '#8c564b']  # 橙, 蓝, 灰绿, 棕

# --- Matplotlib 绘图参数 ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


# --- 2. 可视化函数 ---

# 图1, 2, 3 的函数保持不变
def plot_individual_method_grid(data, output_dir):
    """图1: 绘制各个独立方法贡献度的网格图。"""
    print("--- 正在生成图 1: 独立方法贡献度网格图 ---")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    axes = axes.flatten()
    df = pd.DataFrame(data)
    for i, method in enumerate(df.columns):
        ax = axes[i]
        method_data = df.loc[FEATURE_GROUPS.keys(), method].sort_index(ascending=False)
        ax.barh(method_data.index, method_data.values / 100.0, color=COLORS[::-1])
        ax.set_title(method, fontsize=16, fontweight='bold')
        ax.set_xlabel('Contribution', fontsize=12)
        ax.set_xlim(0, 1.0)
        if i % 3 != 0:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout(pad=3.0)
    path = os.path.join(output_dir, "figure1_individual_method_grid.pdf")
    plt.savefig(path, format='pdf')
    plt.close(fig)
    print(f"✅ 图 1 已保存至: {path}")


def plot_summary_horizontal_bar_with_errors(data, output_dir):
    """图2: 绘制带误差棒的汇总条形图。"""
    print("--- 正在生成图 2: 带误差棒的汇总条形图 ---")
    df = pd.DataFrame(data).T
    mean_contributions = df.mean().sort_values(ascending=True)
    std_contributions = df.std().reindex(mean_contributions.index)
    fig, ax = plt.subplots(figsize=(12, 8))
    group_order = list(mean_contributions.index)
    bar_colors = [COLORS[list(FEATURE_GROUPS.keys()).index(name)] for name in group_order]
    ax.barh(mean_contributions.index, mean_contributions.values, xerr=std_contributions.values,
            color=bar_colors, capsize=5, ecolor='gray', alpha=0.8)
    ax.set_title('Mean Contribution of Input Groups with Variability', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Mean Contribution (%)', fontsize=14)
    max_val = (mean_contributions + std_contributions).max()
    ax.set_xlim(0, max_val * 1.25 if not pd.isna(max_val) else 100)
    for i, (mean, std) in enumerate(zip(mean_contributions, std_contributions)):
        ax.text(mean + std + 1, i, f'{mean:.1f} ± {std:.1f}%', ha='left', va='center', fontsize=11)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "figure2_summary_bar_with_errors.pdf")
    plt.savefig(path, format='pdf')
    plt.close(fig)
    print(f"✅ 图 2 已保存至: {path}")


def plot_annotated_donut_chart(data, output_dir):
    """图3: 绘制带注释图例的汇总环形图。"""
    print("--- 正在生成图 3: 带注释的汇总环形图 ---")
    df = pd.DataFrame(data).T
    mean_contributions = df.mean().sort_values(ascending=False)
    std_contributions = df.std().reindex(mean_contributions.index)
    sorted_labels = mean_contributions.index
    sorted_colors = [COLORS[list(FEATURE_GROUPS.keys()).index(name)] for name in sorted_labels]
    fig, ax = plt.subplots(figsize=(10, 10))
    labels_with_std = [f'{name}\n({mean:.1f} ± {std:.1f}%)'
                       for name, mean, std in zip(sorted_labels, mean_contributions, std_contributions)]
    wedges, _, autotexts = ax.pie(
        mean_contributions, autopct='%1.1f%%', startangle=90, colors=sorted_colors,
        pctdistance=0.85, wedgeprops=dict(width=0.4, edgecolor='w'))
    plt.setp(autotexts, size=14, weight="bold", color="white")
    ax.set_title('Contribution Proportions of Input Groups', fontsize=18, fontweight='bold', pad=20)
    ax.legend(wedges, labels_with_std, title="Input Groups (Mean ± Std Dev)",
              loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=12, title_fontsize=13)
    ax.text(0, 0, 'Mean\nProportion', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.axis('equal')
    path = os.path.join(output_dir, "figure3_summary_donut_with_annotations.pdf")
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 图 3 已保存至: {path}")


def plot_feature_contributions_with_outlier_removal(feature_data, groups, output_dir):
    """图4 (最终版): 通过IQR剔除异常值后，绘制对称、无截断的误差线图。"""
    print("--- 正在生成图 4 (最终版): 组内各特征贡献度误差线图 ---")

    df_percent = feature_data.div(feature_data.sum(axis=0), axis=1) * 100
    n_groups = len(groups)

    fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(12, 3 * n_groups + 2), sharex=True)
    if n_groups == 1:
        axes = [axes]

    # --- 新增：计算和存储剔除异常值后的统计数据 ---
    all_cleaned_stats = {}
    for group_name, features in groups.items():
        valid_features = [f for f in features if f in df_percent.index]
        if not valid_features:
            continue

        group_df = df_percent.loc[valid_features]

        cleaned_means = pd.Series(index=group_df.index, dtype=float)
        cleaned_stds = pd.Series(index=group_df.index, dtype=float)

        for feature, row in group_df.iterrows():
            # 使用IQR方法识别和剔除异常值
            Q1 = row.quantile(0.25)
            Q3 = row.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 保留非异常值
            cleaned_row = row[(row >= lower_bound) & (row <= upper_bound)]

            # 基于清洗后的数据计算均值和标准差
            cleaned_means[feature] = cleaned_row.mean()
            cleaned_stds[feature] = cleaned_row.std(ddof=0)  # 使用总体标准差

        all_cleaned_stats[group_name] = {'means': cleaned_means, 'stds': cleaned_stds}

    # 统一X轴范围
    max_x_val = 0
    for group_name in all_cleaned_stats:
        max_val_in_group = (all_cleaned_stats[group_name]['means'] + all_cleaned_stats[group_name]['stds']).max()
        if max_val_in_group > max_x_val:
            max_x_val = max_val_in_group

    # --- 绘制图表 ---
    for i, (group_name, features) in enumerate(groups.items()):
        ax = axes[i]
        color = COLORS[i % len(COLORS)]

        if group_name not in all_cleaned_stats:
            ax.set_title(f'{group_name}: No features to display', fontsize=14, fontweight='bold', color='gray')
            ax.set_yticks([])
            continue

        mean_vals = all_cleaned_stats[group_name]['means'].sort_values(ascending=True)
        std_vals = all_cleaned_stats[group_name]['stds'].reindex(mean_vals.index)

        # 绘制对称的误差线图
        ax.errorbar(x=mean_vals.values, y=mean_vals.index, xerr=std_vals.values,
                    fmt='o', linestyle='', capsize=5, color=color,
                    markerfacecolor=color, markeredgecolor=color,
                    elinewidth=1.5, markersize=6)

        ax.set_title(f'Feature Contributions for: {group_name}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xlim(left=0, right=max_x_val * 1.1)

    fig.suptitle('Mean Contribution of Individual Features (Outliers Removed)', fontsize=18, fontweight='bold', y=1.0)
    plt.xlabel('Mean Relative Contribution (%)', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    path = os.path.join(output_dir, "figure4_individual_feature_contributions_final.pdf")
    plt.savefig(path, format='pdf')
    plt.close(fig)
    print(f"✅ 最终版图 4 已保存至: {path}")


# --- 3. 主执行模块 ---
if __name__ == '__main__':
    if not os.path.exists(GROUPED_CSV_PATH) or not os.path.exists(INDIVIDUAL_CSV_PATH):
        print(f"!!! 错误: 一个或多个输入文件未找到。")
        print(f"  - 检查路径: {GROUPED_CSV_PATH}")
        print(f"  - 检查路径: {INDIVIDUAL_CSV_PATH}")
        print("!!! 请先成功运行 '01_calculate_contributions.py' 来生成数据文件。")
    else:
        print("--- 成功找到所有输入文件 ---")

        print(f"--- 正在从以下文件加载分组数据:\n    {GROUPED_CSV_PATH}")
        grouped_data = pd.read_csv(GROUPED_CSV_PATH, index_col=0)
        grouped_data_dict = grouped_data.to_dict()

        print(f"--- 正在从以下文件加载单个特征数据:\n    {INDIVIDUAL_CSV_PATH}")
        individual_data = pd.read_csv(INDIVIDUAL_CSV_PATH, index_col=0)

        print("\n--- 开始生成所有可视化图表 ---")
        plot_individual_method_grid(grouped_data_dict, OUTPUT_DIR)
        plot_summary_horizontal_bar_with_errors(grouped_data_dict, OUTPUT_DIR)
        plot_annotated_donut_chart(grouped_data_dict, OUTPUT_DIR)

        plot_feature_contributions_with_outlier_removal(individual_data, FEATURE_GROUPS, OUTPUT_DIR)

        print("\n--- 第2阶段完成: 所有图表均已成功生成! ---")