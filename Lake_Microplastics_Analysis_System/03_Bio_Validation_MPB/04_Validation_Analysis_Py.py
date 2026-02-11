import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("Bio.csv")

# 1. 数据清洗和准备
analysis_df = df[['Trophic_level', 'Measured_Abundance_g']].copy()
analysis_df = analysis_df.dropna()

# 移除丰度为零或非正的观测点，因为对数连接函数要求正值
threshold = 1e-6
analysis_df = analysis_df[analysis_df['Measured_Abundance_g'] > threshold]

try:
    param_path = "./data/Bio/model_parameters.csv"
    params_df = pd.read_csv(param_path, index_col=0)

    # 动态获取回归系数 (Beta)
    simulated_beta = params_df.loc['Trophic_level', 'Estimate']

    # 截距 (Intercept) 的校准逻辑
    # 我们基于观测数据的中心点进行对齐，以确保可视化趋势的准确性
    Y_log_mean = np.log(analysis_df['Measured_Abundance_g']).mean()
    simulated_intercept = Y_log_mean - simulated_beta * analysis_df['Trophic_level'].mean()

    print(f"成功加载模型参数: Beta = {simulated_beta:.4f}")
except Exception as e:
    print(f"错误: 无法读取上游模型参数，请确保已运行 R 统计脚本。{e}")

# 2. 拟合 (Calibrated) NB-GLMM 的非线性预测曲线
# NB-GLMM (Log Link) 的逆函数是 exp()
# E[Y] = exp(Intercept + Beta * Trophic_level)

# 创建平滑的营养级序列
pred_X = np.linspace(analysis_df['Trophic_level'].min(), analysis_df['Trophic_level'].max(), 100)

# 计算线性预测器 (Log Scale)
linear_predictor = simulated_intercept + simulated_beta * pred_X

# 计算原始尺度上的预测值 (非线性曲线)
predicted_Y = np.exp(linear_predictor)

# 3. 可视化
plt.figure(figsize=(10, 6))

# 绘制原始散点数据
plt.scatter(
    analysis_df['Trophic_level'],
    analysis_df['Measured_Abundance_g'],
    alpha=0.6,
    color='#0a3b5f', # 深蓝色
    edgecolor='black',
    s=70,
    label='原始观测数据 (items/g)'
)

# 绘制 NB-GLMM 拟合 (Calibrated)拟合曲线 (非线性曲线)
plt.plot(
    pred_X,
    predicted_Y,
    color='#e31a1c', # 亮红色
    linestyle='-',
    linewidth=3,
    label='NB-GLMM 模拟预测曲线 (Exp Link)'
)

# 使用 Log Scale 作为 Y 轴
plt.yscale('log')

# 添加统计信息和标题
# (!!!) 修复：在字符串前使用 'r' 前缀，并用 '$' 包围数学公式，确保 matplotlib 正确渲染
stats_text = (
    r'模型: 模拟 NB-GLMM' + '\n'
    r'关系: $\log(E[Y]) = \beta_0 + \beta_1 \cdot \text{Trophic}$' + '\n'
    r'预测: $E[Y] = \exp(\beta_0 - 0.83 \cdot \text{Trophic})$'
)
plt.annotate(
    stats_text,
    xy=(0.05, 0.95),
    xycoords='axes fraction',
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    verticalalignment='top'
)

# 标签和标题
# (!!!) 为了兼容，我们移除中文字体名 (如果您的环境配置了中文字体，则不会报错，但为了通用性移除)
plt.title('模拟 NB-GLMM 预测曲线 (原始尺度)', fontsize=16)
plt.xlabel('营养级 (Trophic Level)', fontsize=14)
plt.ylabel('微塑料丰度 (件/g, Log Scale)', fontsize=14)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend(loc='lower right')

# 保存图像
plt.savefig('simulated_nb_glmm_curve.png')
plt.close()

print("simulated_nb_glmm_curve.png")