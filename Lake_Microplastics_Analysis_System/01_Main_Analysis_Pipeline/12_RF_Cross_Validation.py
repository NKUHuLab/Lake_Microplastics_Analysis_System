# 12_rf_cross_validation.py

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from code import config

warnings.filterwarnings("ignore")

def evaluate_metrics(y_true, y_pred):
    """Calculates RMSE, R2, and Pearson correlation coefficient."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Ensure inputs are numpy arrays for correlation calculation
    corr_coef = np.corrcoef(y_true.to_numpy().flatten(), y_pred.flatten())[0, 1]
    return rmse, r2, corr_coef

def perform_rf_cross_validation():
    """
    Performs a 10-fold cross-validation on the training data using the
    final RandomForest model's configuration.
    """
    print("--- Starting 10-Fold Cross-Validation for RandomForest Model ---")

    # 1. Load data
    try:
        data = pd.read_csv(config.TRAIN_DATA_PATH).dropna(subset=[config.TARGET_VARIABLE])
        print("Training data loaded successfully.")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not load training data. Please check the path in config.py. Error: {e}")
        return

    X = data[config.MODEL_FEATURES].fillna(data[config.MODEL_FEATURES].median())
    y = data[config.TARGET_VARIABLE]

    # 2. Define the model with the same hyperparameters as the final trained model
    # Note: Cross-validation re-trains the model on each fold.
    # We define the model architecture here, we do not load the .pkl file.
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    print("RandomForest model configured for cross-validation.")

    # 3. Set up the 10-fold cross-validation
    cv_splitter = ShuffleSplit(n_splits=10, test_size=0.2, random_state=config.RANDOM_STATE)
    print(f"Using ShuffleSplit for {cv_splitter.get_n_splits()}-fold cross-validation.")

    # 4. Perform cross-validation and collect metrics
    metrics_list = []
    for train_index, test_index in tqdm(cv_splitter.split(X), total=cv_splitter.get_n_splits(), desc="CV Folds"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model on the training fold
        model.fit(X_train, y_train)

        # Predict on the test fold
        y_pred = model.predict(X_test)

        # Evaluate and store metrics
        rmse, r2, corr = evaluate_metrics(y_test, y_pred)
        metrics_list.append({'RMSE': rmse, 'R2': r2, 'Corr': corr})

    # 5. Aggregate and display results
    results_df = pd.DataFrame(metrics_list)
    print("\n--- Cross-Validation Results ---")
    print(results_df.to_string())

    print("\n--- Aggregated Performance Metrics ---")
    mean_r2 = results_df['R2'].mean()
    std_r2 = results_df['R2'].std()
    mean_rmse = results_df['RMSE'].mean()
    std_rmse = results_df['RMSE'].std()
    mean_corr = results_df['Corr'].mean()
    std_corr = results_df['Corr'].std()

    print(f"Average R²: {mean_r2:.4f} (± {std_r2:.4f})")
    print(f"Average RMSE: {mean_rmse:.4f} (± {std_rmse:.4f})")
    print(f"Average Corr: {mean_corr:.4f} (± {std_corr:.4f})")
    print("\n--- Cross-validation process finished. ---")

if __name__ == '__main__':
    # Ensure config.py can be found if running as a script
    # This might need adjustment based on your project structure
    try:
        from code import config
    except ImportError:
        import sys
        # Assuming the script is run from the project root
        sys.path.append('.')
        from code import config

    perform_rf_cross_validation()