# -*- coding: utf-8 -*-
"""Training subset sensitivity — 高质量方法子集 (polygon面染色版)"""
import os, sys, warnings
from pathlib import Path
import numpy as np; import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensitivity_common as U
OUT = U.OUT_DIR
FEATURES = U.FEATURE_LIST

def main():
    print("="*60); print("高质量方法子集"); print("="*60)
    df_train = U.load_train_with_metadata()
    idx = U.subsample_high_quality(df_train)
    df_sub = df_train.loc[idx]
    X = df_sub[FEATURES].values; y = df_sub['ln'].values
    model = U.train_rf_subsample(X, y)

    R_point, ci_low, ci_high, _ = U.compute_concentration_sum_tree_spread(model)

    shp_path = os.path.join(OUT, '敏感性_高质量方法_lakes.shp')
    merged_gdf, features_df = U.predict_and_save_shp(model, shp_path)
    U.export_prediction_xlsx_from_shp(merged_gdf, os.path.join(OUT, '敏感性_高质量方法_predictions.xlsx'))
    U.save_sensitivity_cache('high_quality', model, merged_gdf, R_point, ci_low, ci_high)

    U.plot_sensitivity_map(merged_gdf, 'GnBu',
        f'High-Quality Methods Only (n={len(df_sub)})',
        os.path.join(OUT, '敏感性_高质量方法_map'))

    with open(os.path.join(OUT,'敏感性_高质量方法_concentration_sum_diagnostic.txt'),'w',encoding='utf-8') as f:
        f.write(f"High-Quality Methods (n={len(df_sub)})\nOOB R²={model.oob_score_:.4f}\n"
                f"Unweighted concentration sum (tree-resampling 95% range): {R_point:.4e} [{ci_low:.4e}, {ci_high:.4e}]\n")
    print("="*60); print("完成!")

if __name__ == '__main__': main()
