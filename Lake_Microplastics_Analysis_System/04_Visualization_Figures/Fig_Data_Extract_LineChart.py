# -*- coding: utf-8 -*-
"""
三合一折线图数据提取脚本 (增强版：平均值与置信区间)

本脚本旨在从原始数据中，提取用于生成“驱动因素与生态变化关系”折线图的核心聚合数据。
此版本专注于计算每个数据分箱的“平均值”和“95%置信区间”，以进行更强的统计推断。
输出的结果可直接用于数据分析、讨论或报告撰写。
"""
import pandas as pd
import numpy as np
import os

# --- 1. 全局设置 ---

# 输入/输出路径 (仅使用输入)
input_file_path = "E:\\lake-MP-W\\data\\opt\\data\\processed_output\\Xchange.csv"

# 设置Pandas显示选项，以便完整打印数据
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 180)

# --- 2. 加载并预处理主数据 ---
print(f"--- 正在加载并预处理数据 ---")
try:
    main_df = pd.read_csv(input_file_path)
    print(f"成功加载主文件: {input_file_path}")
except FileNotFoundError:
    print(f"错误：找不到文件 '{input_file_path}'。")
    # 此处省略创建示例数据的代码以保持简洁
    exit()

# -- 异常值处理 (全局一次) --
main_df['abs_change'] = main_df['change'].abs()
q99 = main_df['abs_change'].quantile(0.998)
print(f"Y轴(abs_change)的99.8%分位数为: {q99:.2f}，将使用此值作为上限。")
main_df['abs_change_clipped'] = main_df['abs_change'].clip(upper=q99)
print("已将Y轴数据在99.8%分位数处进行裁剪。")
print("-" * 40)


# --- 3. 定义数据提取函数 (平均值与置信区间版) ---

def generate_discussion_data_mean_ci(data, plot_configs):
    """
    循环处理每个驱动因素，提取其背后的聚合数据（平均值、标准误、置信区间）。
    """
    all_aggregated_data = []
    y_col = 'abs_change_clipped'

    for config in plot_configs:
        x_col = config['x_col']
        use_log = config['use_log']
        print(f"正在处理驱动因素: {x_col} (对数变换: {use_log})")

        # 1. 预处理和X轴变换 (同前)
        key_columns = [x_col, 'change', 'income', 'Region']
        plot_data = data.dropna(subset=key_columns).copy()
        if plot_data.empty: continue

        if use_log:
            plot_data['x_transformed'] = np.log1p(np.abs(plot_data[x_col]))
        else:
            plot_data['x_transformed'] = np.abs(plot_data[x_col])

        if not np.isfinite(plot_data['x_transformed']).all(): continue

        # 2. 数据分箱 (同前)
        if use_log:
            bin_width = 0.2
            min_val = np.floor(plot_data['x_transformed'].min() / bin_width) * bin_width
            max_val = np.ceil(plot_data['x_transformed'].max() / bin_width) * bin_width
            bins = np.arange(min_val, max_val + bin_width, bin_width)
        else:
            bin_count = 30
            min_val, max_val = plot_data['x_transformed'].min(), plot_data['x_transformed'].max()
            if min_val == max_val:
                bins = np.linspace(min_val - 1, max_val + 1, bin_count + 1)
            else:
                bins = np.linspace(min_val, max_val, bin_count + 1)

        plot_data['x_bin'] = pd.cut(plot_data['x_transformed'], bins=bins, right=False, include_lowest=True)

        # 3. 按分组(Region, Income)聚合数据
        grouping_cols = {'Region': sorted(plot_data['Region'].unique()), 'Income': sorted(plot_data['income'].unique())}
        for group_type, categories in grouping_cols.items():
            for category in categories:
                actual_col = 'Region' if group_type == 'Region' else 'income'
                subset = plot_data[plot_data[actual_col] == category]
                if subset.empty: continue

                # 【核心修改点】聚合操作改为计算 mean, std, size
                agg = subset.groupby('x_bin', observed=False).agg(
                    y_mean=(y_col, 'mean'),
                    y_std=(y_col, 'std'),
                    n_samples=(y_col, 'size')
                ).reset_index().dropna(subset=['y_mean'])

                if agg.empty: continue

                # 【新增计算】计算标准误(SE)和95%置信区间(CI)
                # 仅在样本量大于1时计算标准差和置信区间
                agg = agg[agg['n_samples'] > 1].copy()
                if agg.empty: continue

                agg['y_se'] = agg['y_std'] / np.sqrt(agg['n_samples'])
                agg['y_ci_lower'] = agg['y_mean'] - 1.96 * agg['y_se']  # 95% CI下限
                agg['y_ci_upper'] = agg['y_mean'] + 1.96 * agg['y_se']  # 95% CI上限

                # 添加描述性列
                agg['driving_factor'] = x_col
                agg['grouping_type'] = group_type
                agg['group_category'] = category
                agg['x_bin_center'] = agg['x_bin'].apply(lambda b: b.mid)

                all_aggregated_data.append(agg)

    if not all_aggregated_data: return pd.DataFrame()

    final_df = pd.concat(all_aggregated_data, ignore_index=True)

    # 调整列顺序以优化可读性
    ordered_cols = [
        'driving_factor', 'grouping_type', 'group_category', 'x_bin_center',
        'y_mean', 'y_ci_lower', 'y_ci_upper', 'y_std', 'n_samples', 'x_bin'
    ]
    final_df = final_df[ordered_cols]

    return final_df


# --- 4. 执行数据提取并打印结果 ---
if __name__ == "__main__":
    plot_configurations = [
        {'x_col': 'fish_gdp_sqkm', 'use_log': True},
        {'x_col': 'Advanced_Waste_Discharge', 'use_log': True},
        {'x_col': 'Cultivated_land', 'use_log': False}
    ]

    discussion_table_mean_ci = generate_discussion_data_mean_ci(main_df, plot_configurations)

    print("\n\n" + "=" * 80)
    print("|| " + "生成的可供讨论的聚合数据 (平均值与95%置信区间版)".center(68) + " ||")
    print("=" * 80)

    if not discussion_table_mean_ci.empty:
        print(discussion_table_mean_ci.to_string())
    else:
        print("未能生成任何数据。请检查输入文件和配置。")