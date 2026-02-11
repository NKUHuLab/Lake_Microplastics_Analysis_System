import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import os

# --- 1. 加载数据 ---
# 确保文件路径正确
fishery_path = r"E:\lake-MP-W\dataset\FAO\2022年渔业.csv"
lakes_path = r"E:\lake-MP-W\dataset\FAO\国家微塑料数据\湖泊微塑料总数据.csv"

# 检查文件是否存在
if not os.path.exists(fishery_path) or not os.path.exists(lakes_path):
    print("错误：一个或两个数据文件未找到，请检查路径。")
    exit()

# --- *** 这里是根据您的更新所做的修改 *** ---
# 既然文件已是UTF-8，我们使用 'utf-8-sig' 来读取
try:
    df_fishery = pd.read_csv(fishery_path, encoding='utf-8-sig')
    df_lakes = pd.read_csv(lakes_path, encoding='utf-8-sig')
    print("数据加载成功 (使用UTF-8模式)。")
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# --- 2. 根据最终、正确的逻辑进行数据聚合 ---
print("正在根据最终、正确的逻辑分离和聚合渔业数据...")

# a. 内陆捕捞 (Inland Capture)
# 逻辑：PRODUCTION_SOURCE_DET.CODE 列的值是 'CAPTURE'
df_capture = df_fishery[df_fishery['PRODUCTION_SOURCE_DET.CODE'] == 'CAPTURE'].copy()
df_capture_agg = df_capture.groupby('country')['VALUE'].sum().reset_index().rename(columns={'VALUE': 'Total_Inland_Capture'})
print(f"内陆捕捞数据聚合完成，共 {df_capture_agg.shape[0]} 个国家。")

# b. 内陆水产养殖 (Inland Aquaculture)
# 逻辑：PRODUCTION_SOURCE_DET.CODE 列的值是 '淡水' 或 'Brackishwater'
aquaculture_conditions = ['FRESHWATER', 'BRACKISHWATER']
df_aquaculture = df_fishery[df_fishery['PRODUCTION_SOURCE_DET.CODE'].isin(aquaculture_conditions)].copy()
df_aquaculture_agg = df_aquaculture.groupby('country')['VALUE'].sum().reset_index().rename(columns={'VALUE': 'Total_Inland_Aquaculture'})
print(f"内陆水产养殖数据聚合完成，共 {df_aquaculture_agg.shape[0]} 个国家。")

# c. 聚合湖泊数据 (与之前相同)
df_lakes_agg = df_lakes.groupby('country').agg(
    Mean_MP_Prediction=('prediction', 'mean'),
    Mean_Total_POP_SERVED=('Total_POP_SERVED', 'mean'),
    Mean_Advanced_Discharge=('Advanced_Waste_Discharge', 'mean'),
    income=('income', 'first')
).reset_index()

# --- 3. 数据融合 ---
print("正在融合捕捞、养殖和湖泊数据...")
# 使用外连接合并捕捞和养殖数据，以包含所有进行任一活动的国家
df_fishery_combined = pd.merge(df_capture_agg, df_aquaculture_agg, on='country', how='outer')

# 将合并后的渔业数据与湖泊数据进行内连接
df_final = pd.merge(df_fishery_combined, df_lakes_agg, on='country', how='inner')

# 外连接产生的缺失值(NaN)意味着该国该项活动产量为0，进行填充
df_final[['Total_Inland_Capture', 'Total_Inland_Aquaculture']] = df_final[['Total_Inland_Capture', 'Total_Inland_Aquaculture']].fillna(0)
print(f"数据融合完成，最终数据集包含 {df_final.shape[0]} 个国家。")

# --- 4. 特征工程与数据清洗 ---
print("正在进行特征工程...")
# 使用 log1p (log(x+1)) 进行对数转换，可以优雅地处理0值
df_final['log_Total_Inland_Capture'] = np.log1p(df_final['Total_Inland_Capture'])
df_final['log_Total_Inland_Aquaculture'] = np.log1p(df_final['Total_Inland_Aquaculture'])
df_final['log_Mean_MP_Prediction'] = np.log1p(df_final['Mean_MP_Prediction'])

# 设定收入分类顺序
income_levels = ['LIC', 'LMIC', 'UMIC', 'HIC']
df_final['income'] = pd.Categorical(df_final['income'], categories=income_levels, ordered=True)

# 移除回归分析中包含缺失值的行
df_final.dropna(subset=[
    'log_Mean_MP_Prediction', 'log_Total_Inland_Capture', 'log_Total_Inland_Aquaculture',
    'Mean_Total_POP_SERVED', 'Mean_Advanced_Discharge', 'income'
], inplace=True)
print(f"清洗后，用于回归分析的数据集包含 {df_final.shape[0]} 个国家。")


# --- 5. 多元线性回归分析 ---
# 检查最终数据集是否为空
if df_final.empty:
    print("\n错误：经过处理和清洗后，没有足够的数据进行回归分析。请检查您的原始数据和筛选条件。")
else:
    print("\n--- SQA 多元线性回归分析结果 ---")
    # 构建模型，同时包含捕捞和养殖作为自变量
    model_formula = 'log_Mean_MP_Prediction ~ log_Total_Inland_Capture + log_Total_Inland_Aquaculture + Mean_Total_POP_SERVED + Mean_Advanced_Discharge + C(income)'
    model = smf.ols(formula=model_formula, data=df_final)
    results = model.fit()
    print(results.summary())


    # --- 6. 关键系数比较检验 (Wald Test) ---
    print("\n--- 关键假设检验：捕捞 vs. 养殖的系数是否存在差异？ ---")
    # 我们的假设是 H0: log_Total_Inland_Capture 的系数 = log_Total_Inland_Aquaculture 的系数
    # 这等价于 H0: log_Total_Inland_Capture - log_Total_Inland_Aquaculture = 0
    try:
        wald_test = results.t_test('log_Total_Inland_Capture - log_Total_Inland_Aquaculture = 0')
        print(wald_test)
    except Exception as e:
        print(f"无法执行Wald检验，可能是模型中的变量名问题: {e}")

# --- 7. 保存最终数据供R使用 ---
output_path_sqa = r"E:\lake-MP-W\dataset\FAO\analysis_data_sqa.csv"
if not df_final.empty:
    df_final.to_csv(output_path_sqa, index=False, encoding='utf-8-sig')
    print(f"\nSQA分析完成。处理后的数据已保存至:\n{output_path_sqa}")
else:
    print("\n由于没有生成有效数据，未保存CSV文件。")