# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages


# ==============================================================================
# 1. 全局绘图设置 (包含中文字体自动配置)
# ==============================================================================

def set_chinese_font():
    """
    自动查找并应用系统中的中文字体
    """
    font_names = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Heiti TC']
    for font_name in font_names:
        if font_name in [f.name for f in font_manager.fontManager.ttflist]:
            plt.rcParams['font.family'] = font_name
            print(f"中文字体已自动设置为: {font_name}")
            return
    print("警告: 未找到指定的中文字体, 图表中的中文可能无法正常显示。")


set_chinese_font()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ==============================================================================
# 2. 数据加载与采样
# ==============================================================================
input_csv_path = 'E:/lake-MP-W/data/opt/data/processed_output/Ychange.csv'
output_pdf_path = '减排潜力密度分布图.pdf'  # 新的文件名

try:
    df = pd.read_csv(input_csv_path)
    print("数据加载成功！")
except FileNotFoundError:
    print(f"错误：无法在指定路径找到文件 '{input_csv_path}'。")
    exit()

# 为了绘图效率，继续使用采样
SAMPLE_SIZE = 50000
if len(df) > SAMPLE_SIZE:
    print(f"为快速生成密度图，将随机抽取 {SAMPLE_SIZE} 行数据作为计算样本...")
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
else:
    df_sample = df.copy()
print("\n" + "=" * 50 + "\n")

# ==============================================================================
# 3. 数据准备
# ==============================================================================
# 进行对数变换前，确保数据是非负的，以避免错误
plot_data = df_sample.loc[df_sample['opt_MPs_count'] >= 0].copy()
plot_data['log_opt_MPs_count'] = np.log1p(plot_data['opt_MPs_count'])

# ==============================================================================
# 4. 生成图表并保存到PDF
# ==============================================================================
# 创建一个PDF文件来保存所有的图表
with PdfPages(output_pdf_path) as pdf:
    # --- 图表1: 按收入水平 (Income) 划分的密度图 ---
    print("正在生成图表1: 按收入水平划分的密度图...")
    fig1, ax1 = plt.subplots(figsize=(14, 9))

    sns.kdeplot(
        data=plot_data,
        x='log_opt_MPs_count',
        hue='income',
        fill=True,
        alpha=0.2,
        linewidth=2.5,
        palette='viridis',
        ax=ax1,
        warn_singular=False  # 忽略0方差数据的警告
    )

    ax1.set_title('不同收入水平国家的MPs减排潜力密度分布', fontsize=20, pad=20)
    ax1.set_xlabel('MPs减排潜力 (对数变换后)', fontsize=14)
    ax1.set_ylabel('密度', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(title='收入水平', fontsize=12)

    plt.tight_layout()
    pdf.savefig(fig1)  # 将第一张图保存到PDF
    plt.close(fig1)  # 关闭图形，以免影响下一张图
    print("图表1 生成完毕。")

    # --- 图表2: 按区域 (Region) 划分的密度图 ---
    print("正在生成图表2: 按区域划分的密度图...")
    fig2, ax2 = plt.subplots(figsize=(14, 9))

    sns.kdeplot(
        data=plot_data,
        x='log_opt_MPs_count',
        hue='Region',
        fill=True,
        alpha=0.2,
        linewidth=2.5,
        palette='plasma',
        ax=ax2,
        warn_singular=False  # 忽略0方差数据的警告
    )

    ax2.set_title('不同区域的MPs减排潜力密度分布', fontsize=20, pad=20)
    ax2.set_xlabel('MPs减排潜力 (对数变换后)', fontsize=14)
    ax2.set_ylabel('密度', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(title='区域', fontsize=12)

    plt.tight_layout()
    pdf.savefig(fig2)  # 将第二张图保存到PDF
    plt.close(fig2)  # 关闭图形
    print("图表2 生成完毕。")

print("\n" + "=" * 50)
print(f"所有分析图表已成功生成并保存到PDF文件：'{output_pdf_path}'")
print("该PDF文件包含2页，每页一张图。")
print("分析完成！")