import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import os

print("--- 开始SQB分析：自动化处理所有共同物种 (最终修正版) ---")

# --- 1. 加载和准备数据 ---
fishery_path = r"E:\lake-MP-W\dataset\FAO\2022年渔业.csv"
lakes_path = r"E:\lake-MP-W\dataset\FAO\国家微塑料数据\湖泊微塑料总数据.csv"

try:
    df_fishery = pd.read_csv(fishery_path, encoding='utf-8-sig')
    df_lakes = pd.read_csv(lakes_path, encoding='utf-8-sig')
    print("数据加载成功。")
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# --- 2. 数据清洗与准备 ---
df_fishery['scientific name'] = df_fishery['scientific name'].str.strip()
df_fishery['PRODUCTION_SOURCE_DET.CODE'] = df_fishery['PRODUCTION_SOURCE_DET.CODE'].str.strip()

df_capture_raw = df_fishery[df_fishery['PRODUCTION_SOURCE_DET.CODE'] == 'CAPTURE'].copy()
aquaculture_conditions = ['FRESHWATER', 'BRACKISHWATER']
df_aquaculture_raw = df_fishery[df_fishery['PRODUCTION_SOURCE_DET.CODE'].isin(aquaculture_conditions)].copy()

captured_species = set(df_capture_raw['scientific name'].dropna())
aquacultured_species = set(df_aquaculture_raw['scientific name'].dropna())
common_species = list(captured_species.intersection(aquacultured_species))
print(f"找到了 {len(common_species)} 种共同物种，将开始循环分析...")

df_lakes_agg = df_lakes.groupby('country').agg(Mean_MP_Prediction=('prediction', 'mean'),
                                               income=('income', 'first')).reset_index()

# --- 3. 循环分析所有共同物种 ---
results_list = []

for species_name in common_species:
    try:
        sanitized_name = species_name.replace(' ', '_').replace('.', '')

        capture_col = f'Capture_{sanitized_name}'
        aqua_col = f'Aquaculture_{sanitized_name}'

        species_capture_agg = df_capture_raw[df_capture_raw['scientific name'] == species_name].groupby('country')[
            'VALUE'].sum().reset_index().rename(columns={'VALUE': capture_col})
        species_aquaculture_agg = \
        df_aquaculture_raw[df_aquaculture_raw['scientific name'] == species_name].groupby('country')[
            'VALUE'].sum().reset_index().rename(columns={'VALUE': aqua_col})

        df_species_combined = pd.merge(species_capture_agg, species_aquaculture_agg, on='country', how='outer')
        df_final_species = pd.merge(df_species_combined, df_lakes_agg, on='country', how='inner')
        df_final_species.fillna(0, inplace=True)

        log_capture_col = f'log_{capture_col}'
        log_aqua_col = f'log_{aqua_col}'

        df_final_species[log_capture_col] = np.log1p(df_final_species[capture_col])
        df_final_species[log_aqua_col] = np.log1p(df_final_species[aqua_col])
        df_final_species['log_Mean_MP_Prediction'] = np.log1p(df_final_species['Mean_MP_Prediction'])

        df_analysis = df_final_species[(df_final_species[capture_col] > 0) | (df_final_species[aqua_col] > 0)].copy()
        df_analysis.dropna(subset=[log_capture_col, log_aqua_col, 'log_Mean_MP_Prediction', 'income'], inplace=True)

        if df_analysis.shape[0] < 15:
            print(f"物种 '{species_name}' 数据点不足 ({df_analysis.shape[0]} 个)，已跳过。")
            continue

        # --- *** 这里是修正的地方：确保公式定义和模型运行在同一逻辑块内 *** ---
        model_formula = f'log_Mean_MP_Prediction ~ {log_capture_col} + {log_aqua_col} + C(income)'
        model_results = smf.ols(formula=model_formula, data=df_analysis).fit()

        results_list.append({
            'species': species_name,
            'n_obs': model_results.nobs,
            'r_squared': model_results.rsquared,
            'capture_coef': model_results.params[log_capture_col],
            'capture_stderr': model_results.bse[log_capture_col],
            'capture_pvalue': model_results.pvalues[log_capture_col],
            'aquaculture_coef': model_results.params[log_aqua_col],
            'aquaculture_stderr': model_results.bse[log_aqua_col],
            'aquaculture_pvalue': model_results.pvalues[log_aqua_col]
        })
        print(f"物种 '{species_name}' 分析完成。")

    except Exception as e:
        print(f"处理物种 '{species_name}' 时发生错误: {e}")

# --- 4. 汇总并保存所有结果 ---
if results_list:
    df_all_species_results = pd.DataFrame(results_list)
    df_all_species_results['capture_significant'] = df_all_species_results['capture_pvalue'] < 0.05
    df_all_species_results['aquaculture_significant'] = df_all_species_results['aquaculture_pvalue'] < 0.05
    df_all_species_results = df_all_species_results.sort_values(by='aquaculture_pvalue', ascending=True)

    output_path = r"E:\lake-MP-W\dataset\FAO\analysis_results_all_species.csv"
    df_all_species_results.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print("自动化分析全部完成！")
    print(f"共分析了 {len(results_list)} 个物种。")
    print(f"最终汇总结果已保存至: {output_path}")
    print("\n部分结果预览：")
    print(df_all_species_results.head())
    print("=" * 50)
else:
    print("\n自动化分析完成，但没有足够的物种数据来生成任何有效的回归模型。")