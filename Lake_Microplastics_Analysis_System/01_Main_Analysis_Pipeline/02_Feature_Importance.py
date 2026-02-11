# -*- coding: utf-8 -*-
# code/02_feature_importance.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.inspection import permutation_importance
from code import config

warnings.filterwarnings("ignore")
plt.rcParams.update(config.PLT_RC_PARAMS)


def plot_feature_importance():
    """
    计算并绘制基于置换重要性的特征重要性图。
    """
    print("--- 步骤 2: 开始生成特征重要性图 ---")

    # 1. 加载模型和数据
    try:
        model = joblib.load(config.MODEL_PATH)
        data = pd.read_csv(config.TRAIN_DATA_PATH).dropna(subset=[config.TARGET_VARIABLE])
        X = data[config.MODEL_FEATURES]
        y = data[config.TARGET_VARIABLE]
    except (FileNotFoundError, KeyError) as e:
        print(f"错误: 加载模型或数据失败，请确保已运行步骤1。错误信息: {e}")
        return

    # 2. 计算置换重要性
    print("正在计算置换重要性 (这可能需要一些时间)...")
    results = permutation_importance(model, X, y, n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1)

    # 3. 整理数据并排序
    features_import = pd.DataFrame({
        'feature': config.MODEL_FEATURES,
        'importance': results.importances_mean
    })
    features_import.sort_values('importance', inplace=True, ascending=False)

    # 4. 仅选择前20个最重要的特征进行可视化
    top_n = 20
    features_import_top = features_import.head(top_n)

    # 5. 绘制条形图
    print("正在绘制特征重要性图...")
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.barh(features_import_top['feature'], features_import_top['importance'],
            height=0.7, color='#008792', edgecolor='#005344')

    ax.set_xlabel("置换重要性 (Permutation Importance)")
    ax.set_ylabel("特征")
    ax.set_title(f"前 {top_n} 特征重要性")
    ax.invert_yaxis()  # 将最重要的特征放在顶部
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # 6. 保存图像
    output_path = config.FEATURE_IMPORTANCE_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"正在将图像保存到: {output_path}")
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print("--- 特征重要性图生成完毕 ---\n")


if __name__ == '__main__':
    config.ensure_directories_exist()
    plot_feature_importance()