# 13_mlp_evaluation.py (Final Revision for Robustness)

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# It's good practice to have the config in a separate, importable file
try:
    from code import config
except (ImportError, ModuleNotFoundError):
    print("Warning: Could not import 'config' from 'code' directory.")
    print("Please ensure this script is run from the project's root directory or adjust the path.")


    # Define fallback config values if config.py is not found
    class FallbackConfig:
        TRAIN_DATA_PATH = 'data/train_data.csv'  # Example path
        MODEL_FEATURES = []  # Must be defined in your actual config
        TARGET_VARIABLE = ''  # Must be defined in your actual config
        RANDOM_STATE = 42


    config = FallbackConfig()

warnings.filterwarnings("ignore", category=RuntimeWarning)


def evaluate_metrics(y_true, y_pred):
    """Calculates RMSE, R2, and Pearson correlation coefficient."""
    if y_pred.ndim > 1: y_pred = y_pred.flatten()
    y_true_flat = y_true.to_numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred))
    r2 = r2_score(y_true_flat, y_pred)

    if np.std(y_true_flat) == 0 or np.std(y_pred) == 0:
        corr_coef = np.nan
    else:
        corr_coef = np.corrcoef(y_true_flat, y_pred)[0, 1]
    return rmse, r2, corr_coef


def evaluate_mlp_model():
    """
    Builds and evaluates a robust MLP Regressor using a PowerTransformer
    to handle skewed data with negative values correctly.
    """
    print("--- Starting FINAL MLP Model Evaluation ---")

    # 1. Load data
    try:
        data = pd.read_csv(config.TRAIN_DATA_PATH).dropna(subset=[config.TARGET_VARIABLE])
        print("Training data loaded successfully.")
    except (FileNotFoundError, KeyError) as e:
        print(f"FATAL: Could not load data. Check path/names in config.py. Error: {e}")
        return

    if not config.MODEL_FEATURES or not config.TARGET_VARIABLE:
        print("FATAL: MODEL_FEATURES or TARGET_VARIABLE not defined in config.py.")
        return

    # 2. Initial Data Cleaning (Imputation only)
    X = data[config.MODEL_FEATURES].copy()
    y = data[config.TARGET_VARIABLE]

    # Clean specific known problematic values if necessary
    if 'Res_time' in X.columns and (X['Res_time'] <= 0).any():
        print("Cleaning 'Res_time': replacing non-positive values with NaN...")
        X.loc[X['Res_time'] <= 0, 'Res_time'] = np.nan

    print("Imputing missing values with column medians...")
    X = X.fillna(X.median())

    # The incorrect np.log1p step is now removed.
    print("\n--- Data Cleaning Complete ---")

    # 3. Define a robust model pipeline
    # This pipeline will first transform the data to be more Gaussian, then scale it.
    mlp_pipeline = Pipeline([
        # FIX: Use PowerTransformer to handle skewness and negative values correctly.
        ('power_transform', PowerTransformer(method='yeo-johnson')),
        ('scaler', RobustScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='tanh',
            solver='adam',
            alpha=0.1,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=config.RANDOM_STATE
        ))
    ])

    # 4. Wrap the model to also transform the target variable 'y'
    # This is critical if the target variable is also skewed.
    print("Wrapping model in TransformedTargetRegressor...")
    final_model = TransformedTargetRegressor(
        regressor=mlp_pipeline,
        # The Yeo-Johnson method works for positive and negative values in y
        transformer=PowerTransformer(method='yeo-johnson')
    )

    # 5. Set up and run cross-validation
    cv_splitter = ShuffleSplit(n_splits=10, test_size=0.2, random_state=config.RANDOM_STATE)
    print(f"Using ShuffleSplit for {cv_splitter.get_n_splits()}-fold cross-validation...")

    metrics_list = []
    for train_index, test_index in tqdm(cv_splitter.split(X), total=cv_splitter.get_n_splits(), desc="CV Folds"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        metrics = evaluate_metrics(y_test, y_pred)
        metrics_list.append({'RMSE': metrics[0], 'R2': metrics[1], 'Corr': metrics[2]})

    # 6. Aggregate and display results
    metrics_df = pd.DataFrame(metrics_list).dropna()

    summary_data = {
        'Model': 'MLP (Final)',
        'Test R2 Mean': metrics_df['R2'].mean(),
        'Test R2 Std': metrics_df['R2'].std(),
        'Test RMSE Mean': metrics_df['RMSE'].mean(),
        'Test RMSE Std': metrics_df['RMSE'].std(),
        'Test Corr Mean': metrics_df['Corr'].mean(),
        'Test Corr Std': metrics_df['Corr'].std()
    }

    summary_df = pd.DataFrame([summary_data]).round(4)

    print("\n--- MLP Model Performance Summary ---")
    print(summary_df.to_string(index=False))
    print("\n--- MLP evaluation process finished. ---")


if __name__ == '__main__':
    evaluate_mlp_model()