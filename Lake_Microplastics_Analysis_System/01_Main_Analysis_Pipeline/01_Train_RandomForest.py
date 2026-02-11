# -*- coding: utf-8 -*-
# code/01_train_model.py (English Text)

import pandas as pd
import numpy as np
import joblib
import warnings
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, permutation_test_score, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import config

warnings.filterwarnings("ignore")
plt.rcParams.update(config.PLT_RC_PARAMS)


def plot_permutation_test(model_score, permutation_scores, p_value):
    """
    Plots the results of the permutation test with English labels.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(permutation_scores, bins=25, density=True, label='Permutation Scores (Shuffled Data)',
            color='skyblue', edgecolor='grey', alpha=0.8)
    ax.axvline(model_score, ls="--", color="crimson", lw=2.5,
               label=f'Actual Model Score: {model_score:.3f}\n(p-value = {p_value:.4f})')

    ax.set_title("Model Significance Permutation Test", fontsize=16)
    ax.set_xlabel(f"Model Performance ({config.SCORING_METRIC})", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if p_value < 0.05:
        ax.text(0.95, 0.90, 'Result: Statistically Significant', transform=ax.transAxes,
                ha='right', va='top', fontsize=12, color='green', weight='bold')
    else:
        ax.text(0.95, 0.90, 'Result: Not Statistically Significant', transform=ax.transAxes,
                ha='right', va='top', fontsize=12, color='red', weight='bold')

    perm_plot_path = os.path.join(config.EVAL_DIR, "permutation_test_analysis.pdf")
    fig.savefig(perm_plot_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Permutation test analysis plot saved to: {perm_plot_path}")


def train_and_evaluate_final_model():
    """
    Trains, evaluates, and analyzes the final model using pre-set best parameters.
    """
    print("--- Starting final model training and evaluation with preset parameters ---")

    # 1. Load Data
    data = pd.read_csv(config.TRAIN_DATA_PATH).dropna(subset=[config.TARGET_VARIABLE])
    X = data[config.MODEL_FEATURES].fillna(data[config.MODEL_FEATURES].median())
    y = data[config.TARGET_VARIABLE]
    print(f"Data loaded: {len(X)} samples, {len(X.columns)} features.")

    # 2. Define Best Model directly
    print("\n[INFO] Skipping grid search. Using preset best hyperparameters.")
    best_params = {
        'bootstrap': True, 'max_depth': 10, 'max_features': 0.5,
        'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 500
    }
    print("\n[INFO] Using best hyperparameters:", best_params)

    best_model = RandomForestRegressor(
        **best_params, random_state=config.RANDOM_STATE,
        n_jobs=-1, oob_score=True
    )

    # 3. Train Model
    print("\n[INFO] Training the final model...")
    best_model.fit(X, y)
    print("Model training complete.")

    # 4. Overfitting Analysis and Final Evaluation
    print("\n[INFO] Overfitting Analysis and Final Performance Evaluation:")
    cv_splitter = ShuffleSplit(n_splits=10, test_size=0.2, random_state=config.RANDOM_STATE)

    scoring = {'R2': 'r2', 'neg_MSE': 'neg_mean_squared_error'}
    cv_results = cross_validate(best_model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)

    full_train_score = best_model.score(X, y)
    mean_cv_r2 = np.mean(cv_results['test_R2'])
    mean_cv_rmse = np.sqrt(-np.mean(cv_results['test_neg_MSE']))
    std_cv_rmse = np.std(np.sqrt(-cv_results['test_neg_MSE']))
    oob_score = best_model.oob_score_

    print(f"Score on FULL Training Data (R²): {full_train_score:.4f}")
    print(f"Mean Cross-Validation Score (R²): {mean_cv_r2:.4f}")
    print(f"Out-of-Bag (OOB) Score (R²):      {oob_score:.4f}")

    overfit_gap = full_train_score - mean_cv_r2
    print(f"--> Gap between Training and CV Score: {overfit_gap:.4f}")
    if overfit_gap > 0.15:
        print("WARNING: A significant gap exists, suggesting potential overfitting.")
    else:
        print("SUCCESS: The gap is small, indicating good generalization.")

    print("\n[INFO] Final Model Performance (10-fold Cross-Validation):")
    print(f"Average R²:   {mean_cv_r2:.4f} (± {np.std(cv_results['test_R2']):.4f})")
    print(f"Average RMSE: {mean_cv_rmse:.4f} (± {std_cv_rmse:.4f})")

    # 5. Permutation Test for Model Significance
    print("\n[INFO] Performing Permutation Test to check model significance... (This may take some time)")
    score, permutation_scores, p_value = permutation_test_score(
        best_model, X, y, n_permutations=100, cv=cv_splitter,
        scoring=config.SCORING_METRIC, n_jobs=-1
    )

    print(f"Permutation Test Result: p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("SUCCESS: The model is statistically significant (p < 0.05).")
    else:
        print("WARNING: The model is NOT statistically significant (p >= 0.05).")

    plot_permutation_test(score, permutation_scores, p_value)

    # 6. Save Model
    print("\n[INFO] Saving final model...")
    joblib.dump(best_model, config.MODEL_PATH)
    print(f"Final model saved to: {config.MODEL_PATH}")
    print("\n--- SCRIPT FINISHED ---")


if __name__ == '__main__':
    config.ensure_directories_exist()
    train_and_evaluate_final_model()