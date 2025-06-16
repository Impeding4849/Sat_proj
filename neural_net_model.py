import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# --- Try to import TensorFlow and Keras ---
try:
    import tensorflow as tf
    from tensorflow.keras.optimizers import AdamW
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    print(f"TensorFlow version: {tf.__version__}")
    TF_KERAS_AVAILABLE = True
    try:
        import keras_tuner as kt
        print(f"KerasTuner version: {kt.__version__}")
        KERAS_TUNER_AVAILABLE = True
    except ImportError:
        print("Error: KerasTuner library not found. Please install it using 'pip install keras-tuner'")
        KERAS_TUNER_AVAILABLE = False

except ImportError:
    print("Error: TensorFlow/Keras library not found. Please install it using 'pip install tensorflow'")
    TF_KERAS_AVAILABLE = False

# --- 1. Define Constants (mirroring baseline_ml_model.py) ---
FEATURE_COLS_BASE = [
    'altitude_plane_{}_km',
    'inclination_plane_{}_deg',
    'high_q_plane_{}',
    'med_q_plane_{}',
    'low_q_plane_{}'
]
GLOBAL_FEATURES_ML = ['f_phasing']
TARGET_COLS_ML = ['mean_revisit_time_hr', 'mean_achieved_quality']
N_TOTAL_PLANES = 10  # The total number of planes the model is trained on
N_FEATURES_PER_PLANE = len(FEATURE_COLS_BASE)

# --- 2. Load Data ---
try:
    df = pd.read_csv('constellation_training_data_cleaned.csv')
    print("Successfully loaded dataset.")
    print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: 'constellation_training_data_cleaned.csv' not found.")
    print("Please ensure the dataset is in the same directory as the script.")
    df = None

# --- 3. Prepare Data for Multi-Target Regression ---
X, y = None, None
if df is not None:
    all_plane_feature_cols = [pattern.format(i) for i in range(1, N_TOTAL_PLANES + 1) for pattern in FEATURE_COLS_BASE]
    all_model_feature_cols = all_plane_feature_cols + GLOBAL_FEATURES_ML

    if all(col in df.columns for col in TARGET_COLS_ML) and \
       all(col in df.columns for col in all_model_feature_cols):
        y = df[TARGET_COLS_ML].copy()
        X = df[all_model_feature_cols].copy()
        X.fillna(-1, inplace=True)
    else:
        print("Error: One or more target or feature columns not found in the DataFrame.")
        print(f"Expected targets: {TARGET_COLS_ML}")
        print(f"Expected features (examples): {all_model_feature_cols[:N_FEATURES_PER_PLANE*2]} and {GLOBAL_FEATURES_ML}")
        print(f"Found: {list(df.columns)}")

if X is not None and y is not None:
    # --- 3.5 Scale Features ---
    feature_scaler = StandardScaler()
    # Fit on the entire X dataset before splitting, similar to baseline_ml_model.py
    # This scaler will be saved.
    X_scaled_full = feature_scaler.fit_transform(X)

    # --- 4. Split Data ---
    # Split the scaled data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_full, y, test_size=0.2, random_state=4309 # Using scaled data
    )
    print(f"\nTraining set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set size: {X_test.shape[0]} samples, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # --- 4.5 Scale Target Variables (y_train) ---
    # y_test remains on the original scale for evaluation.
    # y_train is a DataFrame. fit_transform returns a NumPy array.
    target_scaler = StandardScaler()
    y_train_scaled_values = target_scaler.fit_transform(y_train)
    print("\nTarget variables (y_train) scaled using StandardScaler.")
    print(f"  Original y_train mean: {np.mean(y_train.values, axis=0)}")
    print(f"  Scaled y_train_scaled_values mean: {np.mean(y_train_scaled_values, axis=0)}")
    print(f"  Scaled y_train_scaled_values std: {np.std(y_train_scaled_values, axis=0)}")

else:
    y_train_scaled_values = None # Ensure it's defined if data loading fails
    print("Skipping model training due to data loading or preparation issues.")

# --- 5. Neural Network Model Definition, Training, and Evaluation ---
if TF_KERAS_AVAILABLE and KERAS_TUNER_AVAILABLE and X is not None and y is not None and X_train is not None:

    def build_model_for_tuner(hp):
        """Builds a Keras Sequential model with tunable hyperparameters."""
        input_shape = X_train.shape[1]
        num_outputs = y_train.shape[1]

        # Define batch_size as a hyperparameter.
        # KerasTuner will pass this to the model.fit() method during the search.
        hp.Choice('batch_size', values=[16, 32, 64, 128])

        model = Sequential([
            Dense(
                units=hp.Int('units_layer_1', min_value=32, max_value=256, step=32),
                input_shape=(input_shape,)
            ),
            BatchNormalization(),
            Activation(hp.Choice('activation_1', values=['relu', 'tanh'])),
            Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)),
            Dense(
                units=hp.Int('units_layer_2', min_value=32, max_value=128, step=32)
            ),
            BatchNormalization(),
            Activation(hp.Choice('activation_2', values=['relu', 'tanh'])),
            Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)),
            Dense(
                units=hp.Int('units_layer_3', min_value=16, max_value=64, step=16)
            ),
            BatchNormalization(),
            Activation(hp.Choice('activation_3', values=['relu', 'tanh'])),
            # No dropout typically after the last hidden layer before the output layer
            Dense(num_outputs, activation='linear')  # Linear activation for regression
        ])

        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        model.compile(optimizer=AdamW(learning_rate=learning_rate),
                      loss='mean_squared_error', 
                      metrics=['mae', 'mse'])
        return model

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] # y_train is a DataFrame here
    
    # Instantiate the tuner
    tuner = kt.Hyperband(
        build_model_for_tuner,
        objective='val_loss',
        max_epochs=50, # Max epochs to train specific model versions during search
        factor=3,
        directory='keras_tuner_dir',
        project_name='constellation_nn_tuning'
    )

    print("\nStarting hyperparameter search with KerasTuner...")
    # Define a simple early stopping for the search phase
    search_early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    tuner.search(X_train, y_train_scaled_values, 
                 epochs=50, # This is also passed to Hyperband's max_epochs logic
                 validation_split=0.2, 
                 callbacks=[search_early_stopping],
                 verbose=1)
    
    print("\nHyperparameter search completed.")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nBest hyperparameters found: {best_hps.values}")

    # Save the best hyperparameters for the NN model
    best_nn_hps_filename = 'best_nn_hps.joblib'
    joblib.dump(best_hps.values, best_nn_hps_filename)
    print(f"Best NN hyperparameters saved to: {best_nn_hps_filename}")

    # Build the model with the best hyperparameters
    nn_model = build_model_for_tuner(best_hps)
    print("\n--- Neural Network Model Summary ---")
    nn_model.summary()

    # Callbacks
    # Path for the best model checkpoint during training
    best_model_checkpoint_path = 'neural_net_model_checkpoint.keras'
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001, verbose=1)

    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True), # Increased patience for final training
        ModelCheckpoint(filepath=best_model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0),
        reduce_lr
    ]

    print("\nTraining the Neural Network model...")
    history = nn_model.fit(
        X_train, y_train_scaled_values, # Pass scaled y_train values
        epochs=300,
        batch_size=best_hps.get('batch_size') or 32, # Optionally tune batch_size or use a fixed one
        validation_split=0.2, # Use 20% of training data for validation
        callbacks=callbacks_list,
        verbose=1
    )
    print("Training completed.")
    # nn_model now has the best weights due to EarlyStopping(restore_best_weights=True)

    # --- 6. Make Predictions ---
    print("\nMaking predictions on the test set (using the best model weights)...")
    y_pred_nn_scaled = nn_model.predict(X_test)

    # --- Unscale the predictions ---
    y_pred_nn_unscaled = target_scaler.inverse_transform(y_pred_nn_scaled)
    print(f"\nPredictions unscaled using the fitted target_scaler.")

    # --- 7. Evaluate the Model for Each Target ---
    print("\n--- Neural Network Model Performance ---")
    performance_metrics_nn = []
    for i, target_name in enumerate(TARGET_COLS_ML):
        true_values = y_test.iloc[:, i]       # y_test is a DataFrame, on original scale
        predicted_values = y_pred_nn_unscaled[:, i] # Use unscaled predictions

        mse = mean_squared_error(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)

        print(f"\nTarget: '{target_name}'")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  R-squared (R²): {r2:.4f}")
        performance_metrics_nn.append({
            'model_name': 'NeuralNetwork',
            'target_variable': target_name,
            'mse': mse,
            'r2': r2
        })
    print("\n------------------------------------")

    # --- 8. Save the Model and Scaler ---
    final_model_filename_nn = 'neural_net_model.keras'
    feature_scaler_filename_nn = 'feature_scaler_nn.joblib' # Scaler for X
    target_scaler_filename_nn = 'target_scaler_nn.joblib'   # Scaler for y

    nn_model.save(final_model_filename_nn)
    joblib.dump(feature_scaler, feature_scaler_filename_nn)
    joblib.dump(target_scaler, target_scaler_filename_nn)

    print(f"\nTrained Neural Network model saved as: {final_model_filename_nn}")
    print(f"Feature scaler saved as: {feature_scaler_filename_nn}")
    print(f"Target scaler saved as: {target_scaler_filename_nn}")

    if os.path.exists(best_model_checkpoint_path):
        print(f"Note: A training checkpoint was also saved at {best_model_checkpoint_path}")
        # You might want to remove it if it's different from the final model and not needed:
        # if final_model_filename_nn != best_model_checkpoint_path:
        #     os.remove(best_model_checkpoint_path)
    
    # --- 9. Save Performance Metrics ---
    if performance_metrics_nn: # Check if list is not empty
        metrics_df_nn = pd.DataFrame(performance_metrics_nn)
        metrics_filename = 'model_performance_metrics.csv'
        model_id_for_metrics_nn = 'NeuralNetwork'

        try:
            if os.path.exists(metrics_filename):
                existing_metrics_df = pd.read_csv(metrics_filename)
                existing_metrics_df = existing_metrics_df[existing_metrics_df['model_name'] != model_id_for_metrics_nn]
                updated_metrics_df = pd.concat([existing_metrics_df, metrics_df_nn], ignore_index=True)
            else:
                updated_metrics_df = metrics_df_nn
            updated_metrics_df.to_csv(metrics_filename, index=False)
            print(f"Neural Network performance metrics saved to {metrics_filename}")
        except Exception as e:
            print(f"Error saving Neural Network performance metrics: {e}")

elif not TF_KERAS_AVAILABLE or not KERAS_TUNER_AVAILABLE:
    print("\nSkipping Neural Network training because TensorFlow/Keras is not available.")
else:
    # This case is already covered by the print statement after data loading/preparation
    pass
