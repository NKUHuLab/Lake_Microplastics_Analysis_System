# -*- coding: utf-8 -*-
# code/00_Model_Comparison.py (Revised with SVR Tuning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
from code import config

warnings.filterwarnings("ignore")
plt.rcParams.update(config.PLT_RC_PARAMS)

def evaluate_model(y_true, y_pred):
    """Calculates RMSE, R2, and Pearson correlation coefficient."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    corr_coef = np.corrcoef(y_true.values.flatten(), y_pred.flatten())[0, 1]
    return rmse, r2, corr_coef

def compare_models():
    """
    Compares model performance. RandomForest and SVR are tuned with GridSearchCV,
    while other models use a robust, predefined configuration.
    """
    print("--- Step 1: Starting Model Performance Comparison (Tuning RF and SVR) ---")

    # 1. Load and prepare data
    try:
        data = pd.read_csv(config.TRAIN_DATA_PATH).dropna(subset=[config.TARGET_VARIABLE])
        print("Training data loaded successfully.")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not load training data. Error: {e}")
        return

    X = data[config.MODEL_FEATURES].copy()
    if 'Res_time' in X.columns:
        X.loc[X['Res_time'] <= 0, 'Res_time'] = np.nan
    X = X.fillna(X.median())
    y = data[config.TARGET_VARIABLE]
    print("Data cleaning and imputation complete.")

    # 2. Define models and parameter grids
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_leaf': [3, 5]
        },
        'SVR': {
            # Parameters must be prerefineed with the model step name from the pipeline ('svr__')
            'svr__C': [1, 10, 100],
            'svr__gamma': ['scale', 'auto', 0.1],
            'svr__kernel': ['rbf']
        }
    }


    # 3. Cross-validation
    cv_outer = ShuffleSplit(n_splits=10, test_size=0.2, random_state=config.RANDOM_STATE)
    results = []

    for model_name, model in tqdm(models.items(), desc="Overall Model Progress"):
        print(f"\n  Evaluating model: {model_name}...")
        metrics_list = []
        for train_index, test_index in cv_outer.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            rmse, r2, corr = evaluate_model(y_test, y_test_pred)
            metrics_list.append({'RMSE': rmse, 'R2': r2, 'Corr': corr})

        df_metrics = pd.DataFrame(metrics_list)
        results.append({
            'Model': model_name,
            'Test R2 Mean': df_metrics['R2'].mean(), 'Test R2 Std': df_metrics['R2'].std(),
            'Test RMSE Mean': df_metrics['RMSE'].mean(), 'Test RMSE Std': df_metrics['RMSE'].std(),
            'Test Corr Mean': df_metrics['Corr'].mean(), 'Test Corr Std': df_metrics['Corr'].std(),
            'Test R2 All': df_metrics['R2'].tolist(),
            'Test Corr All': df_metrics['Corr'].tolist() # Added to store all correlation scores
        })

    # 4. Save and plot results
    results_df = pd.DataFrame(results).drop(columns=['Test R2 All', 'Test Corr All']) # Drop 'Test Corr All' from results_df if not needed in Excel
    results_df.to_excel(config.MODEL_COMPARISON_EXCEL_PATH, index=False)
    print(f"\nModel performance comparison saved to: {config.MODEL_COMPARISON_EXCEL_PATH}")

    fig, ax = plt.subplots(figsize=(12, 8))
    # --- CHANGE START ---
    test_corr_data = [res['Test Corr All'] for res in results]
    model_names = [res['Model'] for res in results]
    bplot = ax.boxplot(test_corr_data, labels=model_names, patch_artist=True,
                       medianprops=dict(color="black", linewidth=1.5))
    # --- CHANGE END ---

    colors = ['#ADD8E6', '#FFB6C1', '#90EE90', '#F0E68C', '#DDA0DD']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_title('Model Performance Comparison (RF & SVR Tuned)')
    ax.set_ylabel('Test Set R² Score') # Keep the label as R² Score as requested
    ax.yaxis.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(config.MODEL_COMPARISON_DIR, "model_corr_comparison.pdf") # Changed plot filename
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Model performance plot saved to: {plot_path}")
    print("--- Model performance comparison finished. ---\n")

if __name__ == '__main__':
    config.ensure_directories_exist()
    compare_models()