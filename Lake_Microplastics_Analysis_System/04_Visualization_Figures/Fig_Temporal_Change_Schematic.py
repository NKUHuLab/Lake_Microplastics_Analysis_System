import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def process_and_aggregate_land_use(time_series_path):
    """
    读取并聚合时间序列土地利用数据。
    （代码注释仍为中文）
    """
    print(f"正在从 '{time_series_path}' 读取土地利用数据...")
    try:
        files = glob.glob(os.path.join(time_series_path, "20*.csv"))
        if not files:
            print(f"错误: 在 '{time_series_path}' 中未找到年份CSV文件。")
            return None

        df_list = []
        for f in sorted(files):
            year = int(os.path.basename(f).split('.')[0])
            df_year = pd.read_csv(f)
            df_year['year'] = year
            df_list.append(df_year)

        full_df = pd.concat(df_list, ignore_index=True)
        full_df.rename(columns={'Cultivated_land': 'Cultivated_Land', 'Artificial_surface': 'Artificialsurface'},
                       inplace=True)

        land_use_cols = ['Cultivated_Land', 'Artificialsurface']
        if not all(col in full_df.columns for col in land_use_cols):
            print(f"错误: DataFrame中缺少一列或多列: {land_use_cols}")
            return None

        aggregated_df = full_df.groupby('year')[land_use_cols].sum()
        print("土地利用数据聚合完成。")
        return aggregated_df
    except Exception as e:
        print(f"处理土地利用数据时发生错误: {e}")
        return None


def process_and_aggregate_fishery(fao_data_path):
    """
    读取并聚合FAO全球渔业产量数据。
    （代码注释仍为中文）
    """
    print(f"\n正在从 '{fao_data_path}' 读取渔业数据...")
    try:
        if not os.path.exists(fao_data_path):
            print(f"错误: 在 '{fao_data_path}' 未找到FAO数据文件。")
            return None

        fao_df = pd.read_csv(fao_data_path, encoding='utf-8')

        # --- 已修正的行 ---
        # 根据您的信息，将重命名的源列改为 'PERIOD'。
        fao_df.rename(columns={'PERIOD': 'year', 'VALUE': 'production'}, inplace=True)

        # 检查 'year' 列现在是否存在
        if 'year' not in fao_df.columns:
            print("错误: 在重命名后，'year' 列仍然不存在。请检查CSV文件中的年份列名是否为 'PERIOD'。")
            # 打印列名以帮助调试
            print(f"文件中的列: {fao_df.columns.tolist()}")
            return None

        country_yearly_prod = fao_df.groupby('year')['production'].sum().reset_index()
        country_yearly_prod = country_yearly_prod.set_index('year')
        print("渔业数据聚合完成。")
        return country_yearly_prod
    except Exception as e:
        print(f"处理渔业数据时发生错误: {e}")
        return None


def create_and_save_plot(df, y_col, y_label, title, y_axis_label, color, output_filepath):
    """
    根据提供的样式创建时间序列图，并将其保存为可编辑的PDF文件。
    （代码注释仍为中文）
    """
    # --- 设置全局字体和PDF输出格式 ---
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # --- 设置图表样式 (模仿示例) ---
    plt.rcParams["figure.figsize"] = (12, 7)
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.axis"] = "y"
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['ytick.major.width'] = 0
    plt.rcParams['ytick.major.size'] = 0

    fig, ax = plt.subplots()

    rolling_window = 5
    y = df[y_col]
    x = df.index

    smoothed = y.rolling(window=rolling_window, center=True).mean()
    std_dev = y.rolling(window=rolling_window, center=True).std()

    ax.plot(x, y, marker='o', linestyle='-', markersize=4, color=color, alpha=0.4, label=f'{y_label} (Annual Data)')
    ax.plot(x, smoothed, color=color, linewidth=2.5, label=f'{y_label} ({rolling_window}-Year Smoothed)')
    ax.fill_between(x, smoothed - std_dev, smoothed + std_dev,
                    color=color, alpha=0.15, label=f'{y_label} Fluctuation Band (±1 std dev)')

    # --- 格式化图表 ---
    ax.set_title(title, pad=20, fontsize=16)
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel("Year")

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 4))
    ax.yaxis.set_major_formatter(formatter)

    ax.legend(loc='upper left')
    plt.tight_layout()

    # --- 保存图表 ---
    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        plt.savefig(output_filepath, format='pdf', bbox_inches='tight')
        print(f"成功: 图表已保存到 -> {output_filepath}")
    except Exception as e:
        print(f"失败: 无法保存图表到 '{output_filepath}'. 错误: {e}")

    plt.close(fig)


# --- 主执行程序 ---
if __name__ == "__main__":
    # 定义输入数据路径
    land_use_path = r"E:\lakemicroplastic\draw\全球预测\timestep"
    fao_file_path = r"E:\lake-MP-W\dataset\FAO\GlobalProduction_2024.1.0\Global_production_quantity.csv"
    # 定义输出PDF的文件夹
    output_dir = r"E:\lake-MP-W\draw\约束年际变化"

    # 1. 处理土地利用数据并为每个类别绘制图表
    land_use_agg_df = process_and_aggregate_land_use(land_use_path)
    if land_use_agg_df is not None:
        # 绘制耕地图
        create_and_save_plot(
            df=land_use_agg_df,
            y_col='Cultivated_Land',
            y_label='Cultivated Land',
            title='Historical Fluctuation of Total Cultivated Land Area',
            y_axis_label='Total Area (square meters)',
            color='#3b7d1d',  # 绿色
            output_filepath=os.path.join(output_dir, "historical_cultivated_land.pdf")
        )

        # 绘制人造地表图
        create_and_save_plot(
            df=land_use_agg_df,
            y_col='Artificialsurface',
            y_label='Artificial Surface',
            title='Historical Fluctuation of Total Artificial Surface Area',
            y_axis_label='Total Area (square meters)',
            color='#c44e52',  # 红色
            output_filepath=os.path.join(output_dir, "historical_artificial_surface.pdf")
        )

    # 2. 处理渔业数据并绘制图表
    fishery_agg_df = process_and_aggregate_fishery(fao_file_path)
    if fishery_agg_df is not None:
        create_and_save_plot(
            df=fishery_agg_df,
            y_col='production',
            y_label='Fishery Production',
            title='Historical Fluctuation of Global Fishery Production',
            y_axis_label='Total Production (tonnes)',
            color='#006ba4',  # 蓝色
            output_filepath=os.path.join(output_dir, "historical_fishery_production.pdf")
        )

    print("\n所有处理和绘图任务已完成。")