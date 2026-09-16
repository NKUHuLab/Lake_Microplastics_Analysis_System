# -*- coding: utf-8 -*-
"""
Geographic subset sensitivity: Pearson correlation and held-out predictive R2
"""
import os, sys, pickle, warnings
from pathlib import Path
import numpy as np; import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensitivity_common as U

OUT = U.OUT_DIR
FEATURES = U.FEATURE_LIST
RF_P = dict(bootstrap=True, max_depth=10, max_features=0.5, min_samples_leaf=1,
             min_samples_split=10, n_estimators=500, random_state=42, n_jobs=-1, oob_score=True)

def compute_corr_r2(model_class_params, X, y):
    """Repeated holdout metrics with imputation fitted within each training split."""
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    corrs = []
    r2_scores = []
    for tr, te in cv.split(X):
        imputer = SimpleImputer(strategy='median')
        X_tr = imputer.fit_transform(X.iloc[tr])
        X_te = imputer.transform(X.iloc[te])
        m = RandomForestRegressor(**model_class_params)
        m.fit(X_tr, y[tr])
        pred = m.predict(X_te)
        r = np.corrcoef(y[te], pred)[0, 1]
        corrs.append(r)
        r2_scores.append(r2_score(y[te], pred))
    mean_r = np.mean(corrs)
    return mean_r, np.asarray(corrs), np.mean(r2_scores), np.asarray(r2_scores)


def fit_final_model(model_params, X, y):
    """Fit one median imputer and model on the final training subset."""
    imputer = SimpleImputer(strategy='median')
    model = RandomForestRegressor(**model_params)
    model.fit(imputer.fit_transform(X), y)
    model.training_feature_medians_ = pd.Series(imputer.statistics_, index=X.columns)
    return model, imputer

print("="*60)
print("Geographic subset sensitivity — Pearson r and held-out R2")
print("="*60)

# 1. Load full training data
print("\n1. 加载数据...")
df_train = U.load_train_with_metadata()
X_full = df_train[FEATURES].copy()
y_full = df_train['ln'].to_numpy()
print(f"  全量训练数据: {len(df_train)} 样本")

# 2. 地理平衡子采样 (分层比例)
print("\n2. 地理平衡子采样...")
idx_geo = U.subsample_geographic_balance(df_train, min_per_continent=25, frac_per_continent=0.45)
df_geo = df_train.loc[idx_geo]
X_geo = df_geo[FEATURES].copy(); y_geo = df_geo['ln'].to_numpy()
print(f"  地理平衡子集: {len(df_geo)} 样本")

# 3. 其他子集
idx_rec = U.subsample_recent_years(df_train)
idx_hq = U.subsample_high_quality(df_train)
X_rec = df_train.loc[idx_rec, FEATURES].copy(); y_rec = df_train.loc[idx_rec, 'ln'].to_numpy()
X_hq = df_train.loc[idx_hq, FEATURES].copy(); y_hq = df_train.loc[idx_hq, 'ln'].to_numpy()

# 4. 训练所有模型 + 计算相关性R²
print("\n3. 训练模型 + 相关性R²...")
model_params = {k:v for k,v in RF_P.items() if k != 'oob_score'}
results = []

for name, X, y in [
    ('Main model (full data)', X_full, y_full),
    ('Geographic balance', X_geo, y_geo),
    ('Recent 10 years', X_rec, y_rec),
    ('High-quality methods', X_hq, y_hq),
]:
    r, corrs, r2_score_mean, r2_scores = compute_corr_r2(model_params, X, y)
    r_squared = r**2
    oob = np.nan
    try:
        m, _ = fit_final_model(RF_P, X, y); oob = m.oob_score_
    except: pass
    results.append({
        'Model': name, 'n': len(y),
        'Corr_r': r, 'R2_corr': r_squared,
        'Corr_std': np.std(corrs),
        'R2_score': r2_score_mean,
        'R2_score_std': np.std(r2_scores),
        'OOB_R2': oob,
    })
    print(f"  {name:30s} n={len(y):>4d}  r={r:.4f}  r²={r_squared:.4f}  R²={r2_score_mean:.4f}  ±{np.std(corrs):.4f}")

# 5. 训练地理平衡最终模型 + 全球预测
print("\n4. 训练地理平衡最终模型 + 全球预测...")
m_geo, geo_imputer = fit_final_model(RF_P, X_geo, y_geo)
fa = pd.read_csv(U.PREDICT_CSV); fa.columns = fa.columns.str.strip()
fa_geo = fa.copy()
fa_geo[FEATURES] = geo_imputer.transform(fa_geo[FEATURES])
geo_predict_csv = os.path.join(OUT, 'feature_2022_geo_training_median_imputed.csv')
fa_geo.to_csv(geo_predict_csv, index=False)
# The legacy helper returns an unweighted concentration sum and tree spread;
# these are not a physical reservoir estimate or confidence interval.
concentration_sum, tree_spread_low, tree_spread_high, _ = U.compute_concentration_sum_tree_spread(
    m_geo, geo_predict_csv
)
shp_path = os.path.join(OUT, '敏感性_地理平衡_lakes.shp')
merged_gdf, features_df = U.predict_and_save_shp(m_geo, shp_path, geo_predict_csv)
U.export_prediction_xlsx_from_shp(merged_gdf, os.path.join(OUT, '敏感性_地理平衡_predictions.xlsx'))
U.save_sensitivity_cache('geo_balanced', m_geo, merged_gdf, concentration_sum, tree_spread_low, tree_spread_high)

# 6. 全球预测对比 (相关性R²)
print("\n5. 全局预测对比...")
m_full, full_imputer = fit_final_model(RF_P, X_full, y_full)
pred_full = m_full.predict(full_imputer.transform(fa[FEATURES]))
pred_geo = m_geo.predict(geo_imputer.transform(fa[FEATURES]))

r_global = np.corrcoef(pred_full, pred_geo)[0, 1]
diff_mean = (pred_geo - pred_full).mean()
diff_std = (pred_geo - pred_full).std()
pct_diff = (np.exp(pred_geo).mean() - np.exp(pred_full).mean()) / np.exp(pred_full).mean() * 100

print(f"  全局预测相关性 r = {r_global:.4f} (r² = {r_global**2:.4f})")
print(f"  全局预测差异: mean = {diff_mean:.4f} ln ({pct_diff:.1f}%)")
print(f"  全局预测差异 std = {diff_std:.4f}")

# 7. 绘制地图
print("\n6. 绘制地理平衡地图...")
U.plot_sensitivity_map(merged_gdf, 'viridis',
    f'Geographically Balanced Subset (Stratified Proportional, n={len(df_geo)})',
    os.path.join(OUT, '敏感性_地理平衡_map'))

# 8. 保存未加权浓度和树间离散度诊断
with open(os.path.join(OUT, '敏感性_地理平衡_concentration_sum.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Geographically Balanced v2 (Stratified Proportional, n={len(df_geo)})\n")
    f.write(f"OOB R²: {m_geo.oob_score_:.4f}\n")
    f.write(f"Correlation r: {results[1]['Corr_r']:.4f} (r²={results[1]['R2_corr']:.4f})\n")
    f.write(f"Unweighted concentration sum: {concentration_sum:.4e}\n")
    f.write(f"Tree spread (not a confidence interval): [{tree_spread_low:.4e}, {tree_spread_high:.4e}]\n")
    f.write(f"Global prediction correlation vs main: r={r_global:.4f}\n")
    f.write(f"Global mean diff: {diff_mean:.4f} ln ({pct_diff:.1f}%)\n")

# 9. 三线表 (相关性R²)
print("\n7. 生成三线表...")
tbl_rows = []
for res in results:
    tbl_rows.append({
        'Model': res['Model'],
        'n (training)': res['n'],
        'Correlation_r': round(res['Corr_r'],4),
        'R2_corr_based': round(res['R2_corr'],4),
        'Corr_Std': round(res['Corr_std'],4),
        'R2_score': round(res['R2_score'],4),
        'R2_score_std': round(res['R2_score_std'],4),
        'OOB_R2': round(res['OOB_R2'],4) if not np.isnan(res['OOB_R2']) else None,
    })
tbl_df = pd.DataFrame(tbl_rows)
tbl_df.to_csv(os.path.join(OUT, '模型对比_相关性R2_三线表.csv'), index=False, encoding='utf-8-sig')
tbl_df.to_excel(os.path.join(OUT, '模型对比_相关性R2_三线表.xlsx'), index=False)
print(tbl_df.to_string(index=False))

# 10. 保存完整结果JSON
import json
final_res = {
    'models': [{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in results],
    'global_prediction': {
        'correlation_r': float(r_global),
        'r_squared': float(r_global**2),
        'mean_diff_ln': float(diff_mean),
        'mean_diff_pct': float(pct_diff),
        'std_diff_ln': float(diff_std),
    },
    'geo_balanced': {
        'n': len(df_geo),
        'oob_r2': float(m_geo.oob_score_),
        'corr_r': float(results[1]['Corr_r']),
        'corr_r2': float(results[1]['R2_corr']),
        'concentration_sum_unweighted': float(concentration_sum),
        'tree_spread_low_not_ci': float(tree_spread_low),
        'tree_spread_high_not_ci': float(tree_spread_high),
    }
}
with open(os.path.join(OUT, 'TrainingSensitivity_geo_balanced_results.json'), 'w') as f:
    json.dump(final_res, f, indent=2)

print("\n" + "="*60)
print("DONE — 地理平衡重跑完成")
print(f"输出: {OUT}")
print("="*60)
