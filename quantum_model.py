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
    from tensorflow.keras.models import Model # Sequential not used, Model is
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    print(f"TensorFlow version: {tf.__version__}")
    try:
        import keras_tuner as kt
        print(f"KerasTuner version: {kt.__version__}")
        KERAS_TUNER_AVAILABLE = True
    except ImportError:
        print("Error: KerasTuner library not found. Please install it using 'pip install keras-tuner'")
        KERAS_TUNER_AVAILABLE = False
    TF_KERAS_AVAILABLE = True
except ImportError:
    print("Error: TensorFlow/Keras library not found. Please install it using 'pip install tensorflow'")
    TF_KERAS_AVAILABLE = False
# --- Try to import PennyLane ---
try:
    import pennylane as qml
    print(f"PennyLane version: {qml.__version__}")
    PENNYLANE_AVAILABLE = True
except ImportError:
    print("Error: PennyLane library not found. Please install it using 'pip install pennylane'")
    PENNYLANE_AVAILABLE = False

# --- 1. Define Constants (mirroring neural_net_model.py) ---
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

# --- 3. Prepare Data ---
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

if X is not None and y is not None:
    # --- 3.5 Scale Features ---
    feature_scaler = StandardScaler()
    X_scaled_full = feature_scaler.fit_transform(X)

    # --- 4. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_full, y, test_size=0.3, random_state=4309
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

# --- Load Best Hyperparameters from Neural Network Model ---
BEST_NN_HPS_FILENAME = 'best_nn_hps.joblib'
best_nn_hps_loaded = None
NN_HPS_LOADED_SUCCESSFULLY = False
if os.path.exists(BEST_NN_HPS_FILENAME):
    try:
        best_nn_hps_loaded = joblib.load(BEST_NN_HPS_FILENAME)
        print(f"Successfully loaded best hyperparameters from Neural Network model: {BEST_NN_HPS_FILENAME}")
        NN_HPS_LOADED_SUCCESSFULLY = True
    except Exception as e:
        print(f"Error loading best NN hyperparameters from {BEST_NN_HPS_FILENAME}: {e}")
        print("Classical part of QML model will not use pre-tuned NN hyperparameters.")
else:
    print(f"Warning: Best NN hyperparameters file '{BEST_NN_HPS_FILENAME}' not found.")
    print("Classical part of QML model will not use pre-tuned NN hyperparameters.")
    print("Ensure 'neural_net_model.py' has been run and saved its HPs for optimal QML model configuration.")


# --- 5. Quantum Model Definition, Training, and Evaluation ---
if TF_KERAS_AVAILABLE and PENNYLANE_AVAILABLE and KERAS_TUNER_AVAILABLE and X_train is not None and y_train is not None:
    if not NN_HPS_LOADED_SUCCESSFULLY:
        print("\nCRITICAL ERROR: Neural Network hyperparameters were not loaded.")
        print("The QML model relies on these for its classical part. Please run 'neural_net_model.py' first.")
        print("Skipping QML model training.")
    else:
        def build_model_for_tuner(hp, nn_hps):
            """Builds a Keras Model with tunable QML and fixed classical hyperparameters."""
            # Tunable N_QUBITS and N_QLAYERS
            n_qubits_hp = hp.Int('n_qubits', min_value=2, max_value=8, step=1)
            n_qlayers_hp = hp.Int('n_qlayers', min_value=1, max_value=4, step=1)

            dev_dynamic = qml.device("default.qubit", wires=n_qubits_hp)

            @qml.qnode(dev_dynamic, interface='tf', diff_method='backprop')
            def quantum_circuit_dynamic(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits_hp))
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits_hp))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_hp)]

            weight_shapes = {"weights": (n_qlayers_hp, n_qubits_hp, 3)}
            qlayer = qml.qnn.KerasLayer(quantum_circuit_dynamic, weight_shapes, output_dim=n_qubits_hp)

            input_dim_classical = X_train.shape[1]
            num_outputs_final = y_train.shape[1]

            inputs_classical = tf.keras.Input(shape=(input_dim_classical,))

            # --- Classical Pre-processing Layers (using FIXED HPs from nn_hps) ---
            # These names ('units_layer_1', 'activation_1', etc.) must match those saved by neural_net_model.py
            x = Dense(units=nn_hps['units_layer_1'])(inputs_classical)
            x = BatchNormalization()(x)
            x = Activation(nn_hps['activation_1'])(x)
            x = Dropout(nn_hps['dropout_1'])(x)

            x = Dense(units=nn_hps['units_layer_2'])(x)
            x = BatchNormalization()(x)
            x = Activation(nn_hps['activation_2'])(x)
            x = Dropout(nn_hps['dropout_2'])(x)
            
            x = Dense(units=nn_hps['units_layer_3'])(x)
            x = BatchNormalization()(x)
            x = Activation(nn_hps['activation_3'])(x)
            # No dropout after last pre-processing hidden layer

            # Quantum input preparation layer
            x_quantum_prep = Dense(n_qubits_hp)(x) # Output size matches n_qubits_hp
            x_quantum_prep = BatchNormalization()(x_quantum_prep)
            quantum_inputs = Activation('tanh')(x_quantum_prep)
            
            quantum_outputs = qlayer(quantum_inputs)
            
            # --- Classical Post-processing Layer (using HPs from NN's 3rd layer as an example) ---
            y = Dense(units=nn_hps['units_layer_3'])(quantum_outputs)
            y = BatchNormalization()(y)
            y = Activation(nn_hps['activation_3'])(y)
            # No dropout before the final output layer

            outputs_final = Dense(num_outputs_final, activation='linear')(y)
            model = Model(inputs=inputs_classical, outputs=outputs_final)

            # Use learning rate from loaded NN HPs
            learning_rate_fixed = nn_hps['learning_rate']
            model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate_fixed),
                          loss='mean_squared_error',
                          metrics=['mae', 'mse'])
            return model

        # Instantiate the tuner
        tuner = kt.Hyperband(
            lambda hp: build_model_for_tuner(hp, nn_hps=best_nn_hps_loaded), # Pass loaded NN HPs
            objective='val_loss',
            max_epochs=60, 
            factor=3,
            directory='keras_tuner_qml_dir',
            project_name='constellation_qml_tuning_quantum_only' # Updated project name
        )

        print("\nStarting hyperparameter search (N_QUBITS, N_QLAYERS) for QML model...")
        search_early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        tuner.search(X_train, y_train_scaled_values,
                     epochs=60, 
                     validation_split=0.2,
                     callbacks=[search_early_stopping],
                     verbose=1)

        print("\nHyperparameter search completed.")
        best_qml_specific_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"\nBest QML-specific hyperparameters found: {best_qml_specific_hps.values}")

        # Build the model with the best quantum HPs and fixed classical HPs from NN
        qml_model = build_model_for_tuner(best_qml_specific_hps, nn_hps=best_nn_hps_loaded)
        # The line below is redundant as qml_model already has its input layer defined.
        # input_dim_classical was local to build_model_for_tuner.
        print("\n--- Quantum-Hybrid Model Summary (Tuned QML, Fixed Classical) ---")
        qml_model.summary()

        best_model_checkpoint_path = 'quantum_model_checkpoint.keras'
        # Learning rate is fixed from NN HPs, so ReduceLROnPlateau might be less impactful
        # but can still reduce if the fixed LR is above its min_lr.
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001, verbose=1)
        callbacks_list = [
            EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True),
            ModelCheckpoint(filepath=best_model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0),
            reduce_lr
        ]

        print("\nTraining the Quantum-Hybrid model with tuned QML HPs and fixed classical HPs...")
        # Use batch_size from loaded NN HPs for the final training
        batch_size_fixed = best_nn_hps_loaded.get('batch_size', 32) # Default if not in HPs

        history = qml_model.fit(
            X_train, y_train_scaled_values,
            epochs=200,
            batch_size=batch_size_fixed, # Use fixed batch size from NN HPs
            validation_split=0.2,
            callbacks=callbacks_list,
            verbose=1
        )
        print("Training completed.")

        print("\nMaking predictions on the test set (using the best model weights)...")
        y_pred_qml_scaled = qml_model.predict(X_test)

        y_pred_qml_unscaled = target_scaler.inverse_transform(y_pred_qml_scaled)
        print(f"\nPredictions unscaled using the fitted target_scaler.")

        print("\n--- Quantum-Hybrid Model Performance ---")
        performance_metrics_qml = []
        for i, target_name in enumerate(TARGET_COLS_ML):
            true_values = y_test.iloc[:, i]
            predicted_values = y_pred_qml_unscaled[:, i]
            mse = mean_squared_error(true_values, predicted_values)
            r2 = r2_score(true_values, predicted_values)
            print(f"\nTarget: '{target_name}'\n  MSE: {mse:.4f}\n  R²: {r2:.4f}")
            performance_metrics_qml.append({
                'model_name': 'QuantumHybrid',
                'target_variable': target_name,
                'mse': mse,
                'r2': r2
            })
        print("\n------------------------------------")

        # Save the best QML-specific hyperparameters (n_qubits, n_qlayers)
        best_qml_only_hps_filename = 'best_qml_only_hps.joblib'
        joblib.dump(best_qml_specific_hps.values, best_qml_only_hps_filename)
        print(f"Best QML-only hyperparameters saved to: {best_qml_only_hps_filename}")

        final_model_weights_filename_qml = 'quantum_model.weights.h5'
        feature_scaler_filename_qml = 'feature_scaler_qml.joblib'
        target_scaler_filename_qml = 'target_scaler_qml.joblib'

        qml_model.save_weights(final_model_weights_filename_qml)
        joblib.dump(feature_scaler, feature_scaler_filename_qml)
        joblib.dump(target_scaler, target_scaler_filename_qml)
        print(f"\nTrained Quantum-Hybrid model weights saved as: {final_model_weights_filename_qml}")
        print(f"Feature scaler saved as: {feature_scaler_filename_qml}")
        print(f"Target scaler saved as: {target_scaler_filename_qml}")

        if performance_metrics_qml:
            metrics_df_qml = pd.DataFrame(performance_metrics_qml)
            metrics_filename = 'model_performance_metrics.csv'
            model_id_for_metrics_qml = 'QuantumHybrid'
            try:
                if os.path.exists(metrics_filename):
                    existing_metrics_df = pd.read_csv(metrics_filename)
                    existing_metrics_df = existing_metrics_df[existing_metrics_df['model_name'] != model_id_for_metrics_qml]
                    updated_metrics_df = pd.concat([existing_metrics_df, metrics_df_qml], ignore_index=True)
                else:
                    updated_metrics_df = metrics_df_qml
                updated_metrics_df.to_csv(metrics_filename, index=False)
                print(f"Quantum-Hybrid performance metrics saved to {metrics_filename}")
            except Exception as e:
                print(f"Error saving Quantum-Hybrid performance metrics: {e}")

elif not TF_KERAS_AVAILABLE or not PENNYLANE_AVAILABLE or not KERAS_TUNER_AVAILABLE:
    print("\nSkipping QML model training because TensorFlow/Keras, PennyLane, or KerasTuner is not available.")
else:
    pass # Data loading issues already handled