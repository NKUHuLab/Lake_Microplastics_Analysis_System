# -*- coding: utf-8 -*-
# code/13_risk_driver_correlation_all_categories_V4_PDF_FIX.py
# (修复版：添加了全局 rcParams 以确保 PDF 文本在 Adobe Illustrator 中可编辑)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings

# --- 0. 全局配置与初始化 ---
print("--- 启动 风险-驱动因素 关联分析 (V4 - 修复PDF字体) ---")

# --- [关键修复：确保 PDF 文本可编辑] ---
# 这会告诉 matplotlib 将字体嵌入为可编辑的Type 42 (TrueType) 字体，而不是路径
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'  # 使用通用的无衬线字体
# --- [修复结束] ---

warnings.filterwarnings('ignore')

# --- 1. 配置区 ---

# --- 输入数据文件 ---
DATA_FILE_PATH = r"E:\lake-MP-ani\data\generated_risk_data\lakes_risk_metrics_final.csv"

# --- 输出图表目录 ---
OUTPUT_DIR = r"E:\lake-MP-W\draw\13_Risk_Driver_Correlation_All_V2"

# --- 分析的列配置 ---
X_COLUMN = 'fish_gdp_s'

Y_COLS_TO_TEST = {
    'CNEI_CR': 'Critically Endangered (CR)',
    'CNEI_EN': 'Endangered (EN)',
    'CNEI_VU': 'Vulnerable (VU)',
    'CNEI_NT': 'Near Threatened (NT)'
}


# --- 2. 核心绘图函数 (已修改) ---

def create_correlation_plot_v2(df_clean_log, x_col_log, y_col_log, x_label, y_label, title, output_filename):
    """
    (V4 版) 创建高级 "Hexbin + Regplot" 联合图表。
    - 修复了 KDE TypeError
    - 强制边框闭合
    - 确保字体设置正确
    """
    print(f"  Creating plot for: {title}...")

    # 1. 计算统计数据
    valid_data = df_clean_log[[x_col_log, y_col_log]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_data) < 2:
        print("    [Warning] Not enough valid data points to calculate correlation.")
        return

    r, p_value = stats.pearsonr(valid_data[x_col_log], valid_data[y_col_log])

    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    stats_text = f"Pearson's $r = {r:.3f}$\n{p_text}"

    print(f"    Stats: r={r:.3f}, p-value={p_value} (on {len(valid_data)} points)")

    # 2. [修改] 设置Seaborn主题 (移除硬编码的 'Arial'，使其继承全局 rcParams)
    sns.set_theme(style="ticks")

    # 3. 创建 JointGrid
    g = sns.JointGrid(data=valid_data, x=x_col_log, y=y_col_log, height=8)

    # 4. 图层 1: Hexbin 密度图
    g.plot_joint(
        plt.hexbin,
        gridsize=100,
        cmap='cividis',
        bins='log',
        mincnt=1
    )

    # 5. 图层 2: 回归线
    sns.regplot(
        data=valid_data,
        x=x_col_log,
        y=y_col_log,
        scatter=False,
        color="#D90429",
        line_kws={'linewidth': 3},
        ax=g.ax_joint
    )

    # 6. 边缘直方图 (已修复 TypeError)
    sns.histplot(data=valid_data, x=x_col_log, ax=g.ax_marg_x, bins=100,
                 color='#555555', kde=False)  # 只绘制条
    sns.kdeplot(data=valid_data, x=x_col_log, ax=g.ax_marg_x,
                color='black', lw=2)  # 单独绘制线
    sns.histplot(data=valid_data, y=y_col_log, ax=g.ax_marg_y, bins=100,
                 color='#555555', kde=False)  # 只绘制条
    sns.kdeplot(data=valid_data, y=y_col_log, ax=g.ax_marg_y,
                color='black', lw=2)  # 单独绘制线

    # 7. 添加统计注释
    g.ax_joint.annotate(
        stats_text,
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        ha='left',
        va='top',
        fontsize=14,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='none')
    )

    # 8. 设置标签和标题
    g.fig.suptitle(title, fontsize=20, y=1.03, fontweight='bold')
    g.set_axis_labels(xlabel=x_label, ylabel=y_label, fontsize=16, fontweight='bold')

    # 9. 强制所有边框闭合
    for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)

    # 10. 保存图表
    for fmt in ['png', 'pdf']:
        output_path = os.path.join(OUTPUT_DIR, f"{output_filename}.{fmt}")
        g.fig.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"    Plot saved to '{os.path.join(OUTPUT_DIR, output_filename)}.[png/pdf]'")
    plt.close(g.fig)


# --- 3. 主执行逻辑 (不变) ---

def main():
    """
    主执行函数：加载数据一次，过滤0值，然后循环遍历所有四个风险类别。
    """
    print("--- 启动 风险-驱动因素 关联分析 (V4 - 修复PDF字体) ---")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 数据加载 ---
    required_cols = [X_COLUMN] + list(Y_COLS_TO_TEST.keys())
    print(f"Loading data from '{DATA_FILE_PATH}'...")
    try:
        df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"--- 错误: 数据文件未找到! ---")
        print(f"请检查路径: {DATA_FILE_PATH}")
        return

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"--- 错误: 输入文件中缺少必需的列: {missing_cols} ---")
        return

    print(f"Loaded {len(df)} total lake records.")

    # --- 循环分析 ---
    x_axis_label = f"Fishery Intensity ({X_COLUMN}, log(x+1))"

    print("\n--- 开始循环分析四个风险类别 (仅保留 X > 0 且 Y > 0 的数据) ---")

    for y_col_name, y_full_name in Y_COLS_TO_TEST.items():
        print(f"\n[分析类别: {y_full_name} ({y_col_name})]")

        # 1. 过滤数据: 只保留 X 和 Y 的 *原始值* 都大于 0 的行
        data_filtered = df[(df[X_COLUMN] > 0) & (df[y_col_name] > 0)].copy()

        if data_filtered.empty:
            print("  [Warning] 在过滤掉0值后，没有剩余数据点。跳过此类别。")
            continue

        # 2. 在 *过滤后的数据* 上执行对数转换
        data_filtered['log_x'] = np.log1p(data_filtered[X_COLUMN])
        data_filtered['current_log_y'] = np.log1p(data_filtered[y_col_name])

        print(f"  移除了 0 值点。正在分析 {len(data_filtered)} 个有效数据点。")

        # 3. 定义此循环的标签和文件名
        current_y_label = f"Species Risk ({y_col_name}, log(x+1))"
        current_title = f"Hypothesis Test: Fishery Intensity vs. {y_full_name} Species Risk"
        output_filename = f"correlation_fishery_vs_{y_col_name}_GTzero"

        # 4. 调用新的绘图函数
        create_correlation_plot_v2(
            df_clean_log=data_filtered,
            x_col_log='log_x',
            y_col_log='current_log_y',
            x_label=x_axis_label,
            y_label=current_y_label,
            title=current_title,
            output_filename=output_filename
        )

    print("\n========================================")
    print("  所有四个类别的关联分析已全部完成。")
    print(f"  图表已保存至: {OUTPUT_DIR}")
    print("========================================")


if __name__ == "__main__":
    main()