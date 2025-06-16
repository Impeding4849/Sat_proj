import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

# --- 1. Load Data ---
# Define base column names and constants (mirroring baseline_optimizer.py)
FEATURE_COLS_BASE = [
    'altitude_plane_{}_km', 
    'inclination_plane_{}_deg',
    'high_q_plane_{}',
    'med_q_plane_{}',
    'low_q_plane_{}'
]
GLOBAL_FEATURES_ML = ['f_phasing'] # Add f_phasing as a global feature
TARGET_COLS_ML = ['mean_revisit_time_hr', 'mean_achieved_quality'] # Modified for ML
N_TOTAL_PLANES = 10 # The total number of planes the model is trained on
N_FEATURES_PER_PLANE = len(FEATURE_COLS_BASE)


# Load the training data from the CSV file.
try:
    df = pd.read_csv('constellation_training_data_cleaned.csv')
    print("Successfully loaded dataset.")
    print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: 'constellation_training_data.csv' not found.")
    print("Please ensure the dataset is in the same directory as the script.")
    df = None # Set to None if file not found


# --- 2. Prepare Data for Multi-Target Regression ---
if df is not None:
    # Define all possible feature columns the model expects (mirroring baseline_optimizer.py)
    all_plane_feature_cols = [pattern.format(i) for i in range(1, N_TOTAL_PLANES + 1) for pattern in FEATURE_COLS_BASE]
    all_model_feature_cols = all_plane_feature_cols + GLOBAL_FEATURES_ML

    # Check if ML target columns and feature columns exist in the dataframe
    # Note: total_system_cost_usd is no longer a direct ML target here
    if all(col in df.columns for col in TARGET_COLS_ML) and \
       all(col in df.columns for col in all_model_feature_cols):
        y = df[TARGET_COLS_ML].copy() # Targets for the ML model
        X = df[all_model_feature_cols].copy()

        # Fill NaN values with -1, as per the optimizer's training strategy
        X.fillna(-1, inplace=True) # f_phasing should not have NaNs, but good practice for other features
    else:
        print("Error: One or more target or feature columns not found in the DataFrame.")
        print(f"Expected targets: {TARGET_COLS_ML}")
        print(f"Expected features (examples): {all_model_feature_cols[:N_FEATURES_PER_PLANE*2]} and {GLOBAL_FEATURES_ML}")
        print(f"Found: {list(df.columns)}")
        X, y = None, None # Invalidate X and y
else:
    X, y = None, None

if X is not None and y is not None:
    # --- 2.5 Scale Features ---
    feature_scaler = StandardScaler()
    # Fit on the entire X dataset before splitting to ensure consistency with optimizer's potential full data usage
    # However, for a pure ML evaluation, fitting only on X_train is standard.
    # For this specific use case (matching optimizer), we fit on all available X.
    X_scaled_full = feature_scaler.fit_transform(X)
    print("\nFeatures (X) scaled using StandardScaler (fitted on full X dataset).")

    # --- 3. Split Data ---
    # Split the data into training and testing sets.
    # The random_state ensures the split is reproducible.
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_scaled_full, y, test_size=0.2, random_state=4309 # Use scaled features for split
    )
    print(f"Training set size: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
    print(f"Test set size: {X_test_scaled.shape[0]} samples, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # --- 3.5 Scale Target Variables (y_train) ---
    target_scaler = StandardScaler()
    y_train_scaled_values = target_scaler.fit_transform(y_train) # y_train is a DataFrame
    print("\nTarget variables (y_train) scaled using StandardScaler.")
    print(f"  Original y_train mean: {np.mean(y_train.values, axis=0)}")
    print(f"  Scaled y_train_scaled_values mean: {np.mean(y_train_scaled_values, axis=0)}")
    print(f"  Scaled y_train_scaled_values std: {np.std(y_train_scaled_values, axis=0)}")



    # --- 4. Initialize and Train the Multi-Output Model ---
    # First, create a standard XGBoost regressor instance. This will be our base estimator.
    # Match parameters from baseline_optimizer.py
    xgb_regressor = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150, 
        learning_rate=0.1,
        max_depth=5, 
        random_state=4309, 
        n_jobs=-1, 
        eval_metric='rmse'
    )

    # Now, wrap the XGBoost regressor with MultiOutputRegressor.
    # This wrapper will train one independent XGBoost model for each target column.
    multi_output_model = MultiOutputRegressor(estimator=xgb_regressor)
    print("\nTraining a separate XGBoost model for each target (using scaled features and targets)...")
    multi_output_model.fit(X_train_scaled, y_train_scaled_values) # Train on scaled X and scaled y
    print("Training completed.")
    
    # --- 5. Make Predictions ---
    # The model will output predictions for two targets.
    print("\nMaking predictions on the test set (using scaled features)...")
    y_pred_scaled = multi_output_model.predict(X_test_scaled)

    # --- Unscale the predictions ---
    y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled)
    print(f"Predictions unscaled using the fitted target_scaler.")

    # --- 6. Evaluate the Model for Each Target ---
    print("\n--- Multi-Target Model Performance ---")
    performance_metrics = []
    # Evaluate performance for each target variable separately.
    for i, target_name in enumerate(TARGET_COLS_ML): # Use ML-specific targets
        # Select the true values and predicted values for the current target
        true_values = y_test.iloc[:, i]       # y_test is on original scale
        predicted_values = y_pred_unscaled[:, i] # Use unscaled predictions

        mse = mean_squared_error(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)

        print(f"\nTarget: '{target_name}'")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  R-squared (R²): {r2:.4f}")
        performance_metrics.append({
            'model_name': 'XGBoost',
            'target_variable': target_name,
            'mse': mse,
            'r2': r2
        })
    print("\n------------------------------------")

    # --- 7. Save the Model and Scaler ---
    model_filename = 'multi_output_xgb_model.joblib'
    feature_scaler_filename_xgb = 'feature_scaler_xgb.joblib'
    target_scaler_filename_xgb = 'target_scaler_xgb.joblib'

    joblib.dump(multi_output_model, model_filename)
    joblib.dump(feature_scaler, feature_scaler_filename_xgb)
    joblib.dump(target_scaler, target_scaler_filename_xgb)

    print(f"\nTrained model saved as: {model_filename}")
    print(f"Feature scaler saved as: {feature_scaler_filename_xgb}")
    print(f"Target scaler saved as: {target_scaler_filename_xgb}")

    # --- 8. Save Performance Metrics ---
    metrics_df = pd.DataFrame(performance_metrics)
    metrics_filename = 'model_performance_metrics.csv'
    model_id_for_metrics = 'XGBoost'

    try:
        if os.path.exists(metrics_filename):
            existing_metrics_df = pd.read_csv(metrics_filename)
            # Remove old entries for this model
            existing_metrics_df = existing_metrics_df[existing_metrics_df['model_name'] != model_id_for_metrics]
            # Append new metrics
            updated_metrics_df = pd.concat([existing_metrics_df, metrics_df], ignore_index=True)
        else:
            updated_metrics_df = metrics_df
        updated_metrics_df.to_csv(metrics_filename, index=False)
        print(f"Performance metrics saved to {metrics_filename}")
    except Exception as e:
        print(f"Error saving performance metrics: {e}")
else:
    print("\nSkipping training and saving due to data loading/preparation issues.")
