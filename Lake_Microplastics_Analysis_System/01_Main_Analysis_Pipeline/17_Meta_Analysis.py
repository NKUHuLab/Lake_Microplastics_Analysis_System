# 18_advanced_meta_analysis.py

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

# 假设 data_preparation.py 文件在同一目录下
# 如果不在，请确保其路径在 sys.path 中
try:
    from data_preparation import setup_environment, save_plot
except ImportError:
    print("错误：无法导入 'data_preparation' 模块。")
    print("请确保 'data_preparation.py' 文件与此脚本位于同一目录中。")


    # 提供一个临时的 setup_environment 函数以避免脚本完全崩溃
    def setup_environment():
        print("正在使用临时的 setup_environment 函数。")
        # 尝试在当前目录创建 output 文件夹
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 返回一个空的 DataFrame 和输出路径
        return pd.DataFrame(), output_dir


    def save_plot(fig, name, dir):
        print(f"临时的 save_plot: 图像将保存为 {dir}/{name}.png")
        fig.savefig(os.path.join(dir, f"{name}.png"))


def run_ridge_analysis_for_group(df_group):
    """为给定的数据子集运行完整的岭回归分析。"""
    # 样本量过少时，模型不稳定，不进行分析
    if len(df_group) < 20:
        return None

    all_predictors = [
        'Primary_Waste_Discharge', 'Secondary_Waste_Discharge', 'Advanced_Waste_Discharge',
        'RSE_paved', 'RSE_gravel', 'RSE_other',
        'Mismanaged', 'fish_gdp_sqkm', 'Cultivated_land', 'Artificial_surface'
    ]
    # 确保所有预测变量都存在于DataFrame中，不存在则用0填充
    for col in all_predictors:
        if col not in df_group.columns:
            df_group[col] = 0

    X_df = df_group[all_predictors].copy()
    Y_df = df_group['ln']

    # 对特定列进行对数转换以处理数据偏度
    log_transform_cols = ['Mismanaged', 'fish_gdp_sqkm']
    for col in log_transform_cols:
        if col in X_df.columns:
            X_df[col] = X_df[col].clip(lower=0)  # 确保没有负值
            X_df[col] = np.log1p(X_df[col])

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=all_predictors)

    # 使用交叉验证的岭回归寻找最佳alpha值并拟合模型
    ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 100)).fit(X_scaled_df, Y_df)

    # 返回标准化后的回归系数
    return pd.Series(ridge_model.coef_, index=all_predictors)


def plot_multigroup_dumbbell_final(effects_df, group_by_var, output_dir):
    """
    使用高度优化的、ggplot2风格的哑铃图可视化效应大小的差异。
    [修改版：按效应系数绝对值的平均值降序排序]
    """

    # --- 修改开始 ---
    # 原始排序逻辑:
    # effects_df['range'] = effects_df.max(axis=1) - effects_df.min(axis=1)
    # df_sorted = effects_df.sort_values('range', ascending=True).drop(columns='range')

    # 新的排序逻辑: 按效应系数绝对值的平均值降序排序
    # 1. 计算每个特征在所有组中的绝对值的平均值
    effects_df['abs_mean'] = effects_df.abs().mean(axis=1)
    # 2. 按这个平均值降序排序 (ascending=False)
    df_sorted = effects_df.sort_values('abs_mean', ascending=False).drop(columns='abs_mean')

    # 按照您的要求，打印排序后的数据
    print(f"\n--- 按“{group_by_var}”分组，并按 [绝对值平均效应] 降序排列的数据 ---")
    print(df_sorted)
    # --- 修改结束 ---

    group_names = df_sorted.columns
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 为不同分组指定颜色

    fig, ax = plt.subplots(figsize=(14, 12))

    # --- 1. 绘制哑铃的“杠”（粗黑线） ---
    ax.hlines(y=df_sorted.index,
              xmin=df_sorted.min(axis=1),
              xmax=df_sorted.max(axis=1),
              color='black', alpha=0.7, linewidth=2.5, zorder=1)

    # --- 2. 绘制哑铃的“点”（按组着色） ---
    for i, group in enumerate(group_names):
        ax.scatter(df_sorted[group], df_sorted.index,
                   color=colors[i],
                   s=180,
                   label=group,
                   zorder=3,
                   edgecolors='black',  # 为点添加黑色描边，使其更清晰
                   linewidth=0.5)

    # --- 3. 主题和美学 ---
    # 为正/负效应区域添加阴影
    xmin, xmax = ax.get_xlim()
    ax.axvspan(xmin, 0, color='royalblue', alpha=0.05, zorder=0, label='_nolegend_')
    ax.axvspan(0, xmax, color='tomato', alpha=0.05, zorder=0, label='_nolegend_')

    # 垂直网格线
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)
    ax.yaxis.grid(False)

    # 简化坐标轴
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)

    # --- 4. 标签、标题和图例 ---
    ax.set_xlabel('效应系数 (标准化)', fontsize=14)
    ax.set_ylabel('')
    ax.set_title(f'效应按“{group_by_var}”的调节作用 (按绝对值均值排序)', fontsize=22, pad=20)

    # 垂直图例
    legend = ax.legend(title=f'按“{group_by_var}”分组',
                       fontsize=12,
                       loc='upper right',
                       bbox_to_anchor=(1.15, 0.95))
    plt.setp(legend.get_title(), fontsize=14)

    ax.tick_params(axis='x', labelsize=12)
    # 确保Y轴标签字体清晰
    ax.set_yticklabels(df_sorted.index, fontdict={'family': 'Arial', 'size': 12})

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 调整布局以容纳图例
    save_plot(fig, f"18_meta_analysis_final_style_by_{group_by_var}_sorted_by_abs_mean11111", output_dir)


if __name__ == '__main__':
    # 设置全局字体，以确保中文字符和负号能正确显示
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Heiti TC', 'SimHei']  # 添加中文字体备选
    plt.rcParams['axes.unicode_minus'] = False

    print("--- 运行高级元分析脚本 (18) ---")
    df, output_dir = setup_environment()

    if df is not None and not df.empty:
        # 定义用于分组的湖泊自身特性变量
        grouping_variables = ['Lake_area', 'Shore_dev', 'Vol_total', 'Res_time']
        # 定义所有预测变量的列表，以便后续循环使用
        all_predictors = [
            'Primary_Waste_Discharge', 'Secondary_Waste_Discharge', 'Advanced_Waste_Discharge',
            'RSE_paved', 'RSE_gravel', 'RSE_other',
            'Mismanaged', 'fish_gdp_sqkm', 'Cultivated_land', 'Artificial_surface'
        ]

        # 对每个分组变量进行循环分析
        for group_var in grouping_variables:
            print(f"\n{'=' * 25} 按 “{group_var}” 进行分组分析 {'=' * 25}")

            # 尝试将数据分为高、中、低三组
            try:
                group_labels = ['低', '中', '高']
                df[f'{group_var}_group'] = pd.qcut(df[group_var].rank(method='first'), q=3, labels=group_labels)
                print(f"成功为“{group_var}”创建了3个分组:\n{df[f'{group_var}_group'].value_counts()}")
            except ValueError as e:
                print(f"无法为“{group_var}”创建3个独立分组，跳过此变量。原因: {e}")
                continue

            # 存储每个分组的效应系数
            group_effects = {}
            for group_name in group_labels:
                df_subset = df[df[f'{group_var}_group'] == group_name]
                effects = run_ridge_analysis_for_group(df_subset)

                if effects is not None:
                    group_effects[group_name] = effects

                    # ===================================================================
                    # ### 新增核心功能：计算并打印具体影响的百分比 ###
                    # ===================================================================
                    print(f"\n--- 在 “{group_var}” 为 “{group_name}” 的组中，各特征增加一个标准差的影响: ---")

                    # 打印表头
                    print(
                        f"{'特征名称':<28} | {'标准差 (真实变化量)':<22} | {'效应系数 (ln变化)':<20} | {'丰度百分比变化 (%)':<20}")
                    print("-" * 100)

                    for predictor in all_predictors:
                        # 从效应Series中获取该特征的系数 (beta)
                        beta = effects.get(predictor)

                        if beta is not None:
                            # 计算该特征在当前子集中的标准差
                            std_dev = df_subset[predictor].std()

                            # **第一步: 计算ln的变化量**
                            # 对于标准化回归，特征增加一个标准差，ln的变化量就是其系数beta本身
                            delta_ln = beta

                            # **第二步: 将ln的变化转换为百分比变化**
                            # 公式: (e^(delta_ln) - 1) * 100
                            percentage_change = (np.exp(delta_ln) - 1) * 100

                            # 打印格式化的结果
                            print(
                                f"{predictor:<28} | {std_dev:<22.4f} | {delta_ln:<20.4f} | {percentage_change:<+20.2f}%")

            # 如果没有生成任何有效结果，则跳到下一个分组变量
            if not group_effects:
                print(f"未能为分组变量“{group_var}”生成任何有效结果。")
                continue

            # 将字典转换为DataFrame以便于比较和绘图
            comparison_df = pd.DataFrame(group_effects)
            print(f"\n--- “{group_var}” 各分组的效应系数对比矩阵 ---")
            print(comparison_df.head())

            # 调用最终的、经过美化的绘图函数
            plot_multigroup_dumbbell_final(comparison_df, group_var, output_dir)

        print("\n所有元分析已成功完成。")
    else:
        print("\n未能加载数据或数据为空，脚本执行终止。")