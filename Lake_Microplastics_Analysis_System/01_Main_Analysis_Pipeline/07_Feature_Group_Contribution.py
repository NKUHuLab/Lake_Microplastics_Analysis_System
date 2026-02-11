# -*- coding: utf-8 -*-
# 01_calculate_contributions.py
#
# 功能:
# 1. 加载用户指定的模型和训练数据。
# 2. 使用六种方法计算每个特征的原始贡献度。
# 3. 按预设规则将贡献度汇总为百分比。
# 4. 将"原始特征贡献度"和"分组百分比贡献度"保存为两个独立的CSV文件。

import pandas as pd
import numpy as np
import os
import joblib
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import Lasso
import warnings

# --- 1. 全局配置 (根据您的 config.py 文件) ---
warnings.filterwarnings("ignore")

# --- 核心文件路径 ---
PROJECT_ROOT = r"E:\lake-MP-W"
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train", "train_data.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "random_forest_model.pkl")

# --- 输出目录和文件 ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "draw", "Publication_Figures_Combined")
GROUPED_CSV_PATH = os.path.join(OUTPUT_DIR, "grouped_percentage_contributions.csv")
INDIVIDUAL_CSV_PATH = os.path.join(OUTPUT_DIR, "individual_feature_contributions.csv")

# --- 模型和特征定义 ---
TARGET_VARIABLE = 'ln'
ALL_FEATURES = [
    'Lake_area', 'Shore_dev', 'Vol_total', 'Res_time',
    'Total_POP_SERVED', 'Average_DF', 'Primary_Waste_Discharge',
    'Secondary_Waste_Discharge', 'Advanced_Waste_Discharge',
    'RSE_paved', 'RSE_gravel', 'RSE_other', 'prec', 'emis_tyre_TSP_HEG',
    'emis_brake_TSP_HEG', 'PM2_5', 'PM10', 'Mismanaged', 'Total_Plast',
    'fish_gdp_sqkm', 'Cultivated_land', 'Artificial_surface'
]

# --- 特征分组 (自动计算 'Other') ---
FEATURE_GROUPS = {
    'Atmospheric Input': ['emis_tyre_TSP_HEG', 'emis_brake_TSP_HEG', 'PM2_5', 'PM10'],
    'Hydrological Input': ['Total_POP_SERVED', 'Average_DF', 'Primary_Waste_Discharge', 'Secondary_Waste_Discharge',
                           'Advanced_Waste_Discharge'],
    'Watershed Non-Point Pollution': ['Mismanaged', 'Total_Plast', 'fish_gdp_sqkm', 'Cultivated_land',
                                      'Artificial_surface']
}
assigned_features = {f for features in FEATURE_GROUPS.values() for f in features}
FEATURE_GROUPS['Other'] = [f for f in ALL_FEATURES if f not in assigned_features]

print("--- 特征分组已确认 ---")
for name, features in FEATURE_GROUPS.items():
    print(f"- {name}: {features}")
print("-" * 25)


def calculate_contributions():
    """
    加载数据和模型，计算贡献度，并返回两个DataFrame：一个为原始贡献度，一个为分组贡献度。
    """
    print("--- 开始计算特征贡献度 ---")
    try:
        print(f"加载模型: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print(f"加载数据: {TRAIN_DATA_PATH}")
        data = pd.read_csv(TRAIN_DATA_PATH).dropna(subset=[TARGET_VARIABLE])
    except Exception as e:
        print(f"!!! 文件加载失败: {e}\n请确保路径正确且文件存在。")
        return None, None

    X = data[ALL_FEATURES].fillna(data[ALL_FEATURES].median())
    y = data[TARGET_VARIABLE]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 存储每个特征的原始重要性
    individual_importances = pd.DataFrame(index=X.columns)

    # --- 逐一计算 ---
    print("  - [1/6] 计算 SHAP...")
    explainer = shap.TreeExplainer(model)
    individual_importances['SHAP'] = np.abs(explainer.shap_values(X)).mean(0)

    print("  - [2/6] 计算 Permutation Importance...")
    perm_result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    individual_importances['Permutation'] = perm_result.importances_mean

    print("  - [3/6] 计算 RFE...")
    rfe = RFE(estimator=RandomForestRegressor(random_state=42), n_features_to_select=1).fit(X, y)
    individual_importances['RFE'] = 1 / rfe.ranking_

    print("  - [4/6] 计算 Elasticity (Lasso Coef)...")
    lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42).fit(X_scaled, y)
    individual_importances['Elasticity'] = np.abs(lasso.coef_)

    print("  - [5/6] 计算 Bootstrap...")
    bootstrap_imps = np.array([
        RandomForestRegressor(random_state=i).fit(X.iloc[idx], y.iloc[idx]).feature_importances_
        for i, idx in enumerate([data.sample(frac=1, replace=True, random_state=j).index for j in range(100)])
    ])
    individual_importances['Bootstrap'] = bootstrap_imps.mean(axis=0)

    print("  - [6/6] 计算 ANOVA F-value...")
    f_values, _ = f_regression(X, y)
    individual_importances['ANOVA'] = f_values

    individual_importances.fillna(0, inplace=True)
    print("--- ✅ 所有方法的特征重要性计算完成 ---")

    # --- 计算分组百分比贡献度 ---
    grouped_contributions = {}
    for method in individual_importances.columns:
        df = individual_importances[[method]].rename(columns={method: 'importance'})
        total_importance = df['importance'].sum()

        contributions = {}
        for group_name, features in FEATURE_GROUPS.items():
            valid_features = [f for f in features if f in df.index]
            group_sum = df.loc[valid_features, 'importance'].sum()
            contributions[group_name] = (group_sum / total_importance) * 100 if total_importance > 0 else 0

        grouped_contributions[method] = contributions

    grouped_df = pd.DataFrame(grouped_contributions)
    print("--- ✅ 分组贡献度计算完成 ---")

    return individual_importances, grouped_df


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    individual_df, grouped_df = calculate_contributions()

    if individual_df is not None and grouped_df is not None:
        # 保存两个CSV文件
        individual_df.to_csv(INDIVIDUAL_CSV_PATH)
        print(f"\n✅ 单个特征贡献度数据已保存至:\n   {INDIVIDUAL_CSV_PATH}")

        grouped_df.to_csv(GROUPED_CSV_PATH)
        print(f"\n✅ 分组百分比贡献度数据已保存至:\n   {GROUPED_CSV_PATH}")

        print("\n--- 第1阶段完成 ---")
    else:
        print("\n--- 任务因错误而终止 ---")