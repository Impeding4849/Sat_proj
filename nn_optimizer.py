import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem # Changed from ElementwiseProblem
from pymoo.optimize import minimize
from sklearn.preprocessing import StandardScaler
# from sklearn.multioutput import MultiOutputRegressor # Not needed for Keras model
import warnings
import plotly.graph_objects as go
import joblib
import plotly.io as pio

# Quieter TensorFlow logging (optional, set before TF import)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

# Set default plotly template for better aesthetics
pio.templates.default = "plotly_dark"

# --- Try to import TensorFlow and Keras ---
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    TF_KERAS_AVAILABLE = True
except ImportError:
    print("Error: TensorFlow/Keras library not found. Please install it using 'pip install tensorflow'")
    TF_KERAS_AVAILABLE = False


# Base column names (mirroring neural_net_model.py and baseline_optimizer.py)
FEATURE_COLS_BASE = [
    'altitude_plane_{}_km',
    'inclination_plane_{}_deg',
    'high_q_plane_{}',
    'med_q_plane_{}',
    'low_q_plane_{}'
]
# TARGET_COLS_ML from neural_net_model.py are ['mean_revisit_time_hr', 'mean_achieved_quality']
# Cost is calculated manually, so the optimizer objectives are:
TARGET_COLS_OPTIMIZER = ['total_system_cost_usd', 'mean_revisit_time_hr', 'mean_achieved_quality']
N_TOTAL_PLANES = 10 # The total number of planes the model is trained on
F_PHASING_BOUNDS = [0, 3] # Min and Max for f_phasing, based on training data generation
N_FEATURES_PER_PLANE = len(FEATURE_COLS_BASE) # Should be 5

# Constants for cost calculation (mirroring baseline_optimizer.py)
SIM_SATELLITE_OPTIONS = {
    "low_quality": {
        "cost": 1.0e6,
        "sensor_quality": 0.5,
        "instrument_fov_deg": 15.0,
        "max_slew_angle_deg": 20.0,
        "mass": 110 #kg
    },
    "medium_quality": {
        "cost": 3.0e6,
        "sensor_quality": 0.7,
        "instrument_fov_deg": 5.0,
        "max_slew_angle_deg": 40.0,
        "mass": 225 #kg
    },
    "high_quality": {
        "cost": 8.0e6,
        "sensor_quality": 0.9,
        "instrument_fov_deg": 1.5,
        "max_slew_angle_deg": 50.0,
        "mass": 450 #kg
    }
}
LAUNCH_COST_PER_KG_USD = 5000.0
OPTIMIZER_TO_SIM_KEYS = {
    'high_q': 'high_quality',
    'med_q':  'medium_quality',
    'low_q':  'low_quality'
}
SATELLITE_TYPE_KEYS = ['high_q', 'med_q', 'low_q']

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')


def get_nn_model_and_scalers():
    """
    Loads a pre-trained Keras model and its associated feature and target scalers from disk.
    """
    model_filename = 'neural_net_model.keras'
    feature_scaler_filename = 'feature_scaler_nn.joblib'
    target_scaler_filename = 'target_scaler_nn.joblib'

    if not TF_KERAS_AVAILABLE:
        print("TensorFlow/Keras not imported. Cannot load Keras model.")
        return None, None, None

    try:
        model = tf.keras.models.load_model(model_filename)
        feature_scaler = joblib.load(feature_scaler_filename)
        target_scaler = joblib.load(target_scaler_filename)
        print(f"Successfully loaded pre-trained Keras model from: {model_filename}")
        print(f"Successfully loaded feature scaler from: {feature_scaler_filename}")
        print(f"Successfully loaded target scaler from: {target_scaler_filename}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_filename}', feature scaler '{feature_scaler_filename}', or target scaler '{target_scaler_filename}' not found.")
        print("Please run 'neural_net_model.py' first to train and save the model and scaler.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while loading the Keras model or scaler: {e}")
        return None, None, None
    return model, feature_scaler, target_scaler



def calculate_system_costs(satellite_mix: dict) -> dict:
    """
    Calculates total hardware, mass, launch, and system costs.
    (Identical to baseline_optimizer.py)
    """
    hardware_cost = 0.0
    total_mass_kg = 0.0
    for sat_type, count in satellite_mix.items():
        sim_sat_key = OPTIMIZER_TO_SIM_KEYS.get(sat_type)
        if not sim_sat_key:
            raise KeyError(f"Satellite type key '{sat_type}' used by optimizer not found in OPTIMIZER_TO_SIM_KEYS mapping.")
        try:
            hardware_cost += SIM_SATELLITE_OPTIONS[sim_sat_key]['cost'] * count
            total_mass_kg += SIM_SATELLITE_OPTIONS[sim_sat_key]['mass'] * count
        except KeyError as e:
            print(f"[ERROR] The key '{e}' was not found for sim_sat_key '{sim_sat_key}' (mapped from '{sat_type}') in SIM_SATELLITE_OPTIONS.")
            print("Please ensure 'cost' and 'mass' are defined for all satellite types in constellation_sim.SATELLITE_OPTIONS.")
            raise

    launch_cost = total_mass_kg * LAUNCH_COST_PER_KG_USD
    total_system_cost_variable = hardware_cost + launch_cost

    return {
        "hardware_cost": hardware_cost,
        "total_mass_kg": total_mass_kg,
        "launch_cost": launch_cost,
        "total_system_cost_variable": total_system_cost_variable
    }

class ConstellationProblem(Problem): # Changed from ElementwiseProblem
    """
    Defines the optimization problem using the Neural Network model.
    """
    def __init__(self, model, feature_scaler, target_scaler, n_planes_to_optimize):
        self.model = model
        self.feature_scaler = feature_scaler
        self.n_planes_to_optimize = n_planes_to_optimize
        self.target_scaler = target_scaler
        self.n_model_plane_features = N_TOTAL_PLANES * N_FEATURES_PER_PLANE
        self.n_model_total_features = self.n_model_plane_features + 1 # +1 for f_phasing

        n_var = n_planes_to_optimize * N_FEATURES_PER_PLANE + 1 # +1 for f_phasing

        plane_xl = [300, 25.0, 0, 0, 0]
        plane_xu = [800, 98.0, 20, 20, 20]

        xl = np.array(plane_xl * n_planes_to_optimize + [F_PHASING_BOUNDS[0]])
        xu = np.array(plane_xu * n_planes_to_optimize + [F_PHASING_BOUNDS[1]])

        n_total_constraints = 4 + (2 * self.n_planes_to_optimize)
        # For Problem, vtype is not directly set here but handled by problem definition
        super().__init__(n_var=n_var, n_obj=3, n_constr=n_total_constraints, xl=xl, xu=xu)

    def _calculate_manual_cost(self, x_problem_vars):
        satellite_counts = {key: 0 for key in SATELLITE_TYPE_KEYS}
        for p_idx in range(self.n_planes_to_optimize):
            base_idx = p_idx * N_FEATURES_PER_PLANE
            satellite_counts['high_q'] += round(x_problem_vars[base_idx + 2])
            satellite_counts['med_q']  += round(x_problem_vars[base_idx + 3])
            satellite_counts['low_q']  += round(x_problem_vars[base_idx + 4])

        cost_components = calculate_system_costs(satellite_counts)
        variable_cost = cost_components['total_system_cost_variable']
        total_cost = variable_cost + self.n_planes_to_optimize # Per-plane overhead
        return total_cost

    def _evaluate(self, X_batch, out, *args, **kwargs): # X_batch is now a 2D array of solutions
        n_solutions = X_batch.shape[0]

        # Prepare batch of inputs for the neural network
        model_inputs_batch = np.full((n_solutions, self.n_model_total_features), -1.0)

        for i in range(n_solutions):
            x_individual = X_batch[i, :]
            x_plane_vars_individual = x_individual[:-1]
            f_phasing_val_individual = round(x_individual[-1])

            model_inputs_batch[i, :len(x_plane_vars_individual)] = x_plane_vars_individual
            model_inputs_batch[i, self.n_model_plane_features] = f_phasing_val_individual

        # Scale the entire batch
        x_scaled_batch = self.feature_scaler.transform(model_inputs_batch)

        # Predict for the entire batch
        # Keras model predicts 2 values: [mean_revisit_time_hr, mean_achieved_quality]
        predictions_batch_scaled = self.model.predict(x_scaled_batch, verbose=0)

        # Unscale the predictions
        predictions_batch_unscaled = self.target_scaler.inverse_transform(predictions_batch_scaled)

        # Initialize arrays for objectives and constraints
        F_batch = np.zeros((n_solutions, self.n_obj))
        G_batch = np.zeros((n_solutions, self.n_constr))

        for i in range(n_solutions):
            x_individual = X_batch[i, :]
            x_plane_vars_individual = x_individual[:-1]

            # Use unscaled predictions
            revisit_time, quality = predictions_batch_unscaled[i, 0], predictions_batch_unscaled[i, 1]
            cost = self._calculate_manual_cost(x_plane_vars_individual)

            F_batch[i, 0] = cost
            F_batch[i, 1] = revisit_time
            F_batch[i, 2] = -quality  # Negate quality to maximize

            # Constraints (g(x) <= 0)
            current_constraints = []
            current_constraints.append(1.0 - cost)                  # cost >= 1.0
            current_constraints.append(0.001 - revisit_time)        # revisit_time >= 0.001 hr
            current_constraints.append(0.0 - quality)               # quality >= 0.0
            current_constraints.append(quality - 1.0)               # quality <= 1.0

            for p_idx in range(self.n_planes_to_optimize):
                base_idx = p_idx * N_FEATURES_PER_PLANE
                high_q_sats = round(x_plane_vars_individual[base_idx + 2])
                med_q_sats = round(x_plane_vars_individual[base_idx + 3])
                low_q_sats = round(x_plane_vars_individual[base_idx + 4])
                sum_sats_this_plane = high_q_sats + med_q_sats + low_q_sats
                current_constraints.append(2.0 - sum_sats_this_plane)   # sum_sats_this_plane >= 2
                current_constraints.append(sum_sats_this_plane - 20.0) # sum_sats_this_plane <= 20
            G_batch[i, :] = np.array(current_constraints)

        out["F"] = F_batch
        out["G"] = G_batch


def run_optimization(n_planes, model, feature_scaler, target_scaler):
    """
    Runs the NSGA-II optimization for a specific number of planes using the NN model.
    """
    print(f"\n--- Optimizing for {n_planes} Orbital Plane(s) using Neural Network Model ---")
    problem = ConstellationProblem(model, feature_scaler, target_scaler, n_planes)
    algorithm = NSGA2(pop_size=128, eliminate_duplicates=True)
    res = minimize(problem, algorithm, ('n_gen', 312), seed=1, verbose=False) # Generations can be tuned
    print("Optimization complete.")
    return res.X, res.F


def find_overall_best_compromise(all_results):
    """
    Finds the best compromise solution from ALL Pareto fronts combined.
    (Identical to baseline_optimizer.py)
    """
    all_objectives = np.vstack([res['F'] for res in all_results.values()])
    normalized_obj = np.zeros_like(all_objectives, dtype=float)

    min_cost, max_cost = all_objectives[:, 0].min(), all_objectives[:, 0].max()
    range_cost = max_cost - min_cost if (max_cost - min_cost) > 1e-9 else 1.0
    normalized_obj[:, 0] = (all_objectives[:, 0] - min_cost) / range_cost

    min_revisit, max_revisit = all_objectives[:, 1].min(), all_objectives[:, 1].max()
    range_revisit = max_revisit - min_revisit if (max_revisit - min_revisit) > 1e-9 else 1.0
    normalized_obj[:, 1] = (all_objectives[:, 1] - min_revisit) / range_revisit

    min_quality, max_quality = all_objectives[:, 2].min(), all_objectives[:, 2].max()
    range_quality = max_quality - min_quality if (max_quality - min_quality) > 1e-9 else 1.0
    normalized_obj[:, 2] = (max_quality - all_objectives[:, 2]) / range_quality

    distances = np.linalg.norm(normalized_obj, axis=1)
    best_global_idx = np.argmin(distances)

    point_counter = 0
    for n_planes, results in all_results.items():
        num_points_in_run = len(results['F'])
        if point_counter + num_points_in_run > best_global_idx:
            best_n_planes = n_planes
            best_local_idx = best_global_idx - point_counter
            break
        point_counter += num_points_in_run
    return best_n_planes, best_local_idx

def create_interactive_plot(all_results, best_n_planes, best_solution_idx):
    """
    Generates an interactive 3D plot of all Pareto fronts.
    (Identical to baseline_optimizer.py, but title can be updated)
    """
    fig = go.Figure()
    colors = pio.templates['plotly_dark'].layout.colorway

    for n_planes, results in all_results.items():
        objectives = results['F']
        params = results['X']
        hover_texts = []
        for i in range(len(objectives)):
            cost, revisit, quality = objectives[i]
            param_set = params[i]
            f_phasing_val = round(param_set[-1])
            text = f"<b>Configuration: {n_planes} Planes</b><br>"
            text += "--------------------<br>"
            text += f"Cost: ${cost:,.0f}<br>"
            text += f"Revisit Time: {revisit:.2f} hr<br>"
            text += f"Quality: {quality:.4f}<br>"
            text += f"F-Phasing: {f_phasing_val}<br>"
            text += "--------------------<br><b>Parameters:</b><br>"
            for p_idx in range(n_planes):
                base_idx = p_idx * N_FEATURES_PER_PLANE
                altitude = param_set[base_idx + 0]
                inclination = param_set[base_idx + 1]
                high_q = param_set[base_idx + 2]
                med_q = param_set[base_idx + 3]
                low_q = param_set[base_idx + 4]
                text += (f"  Plane {p_idx+1}: {altitude:.1f} km, {inclination:.1f}°, "
                         f"H:{round(high_q)}, M:{round(med_q)}, L:{round(low_q)}<br>")
            hover_texts.append(text)

        is_optimal_front = (n_planes == best_n_planes)
        fig.add_trace(go.Scatter3d(
            x=objectives[:, 0], y=objectives[:, 1], z=objectives[:, 2],
            mode='markers',
            marker=dict(
                size=5 if is_optimal_front else 3.5,
                color=colors[(n_planes - 1) % len(colors)],
                opacity=1.0 if is_optimal_front else 0.5,
                symbol='diamond' if is_optimal_front else 'circle'
            ),
            name=f'{n_planes} Planes' + (' (Optimal Front)' if is_optimal_front else ''),
            hovertext=hover_texts, hoverinfo='text'
        ))

    best_front_results = all_results[best_n_planes]
    best_obj = best_front_results['F'][best_solution_idx]
    fig.add_trace(go.Scatter3d(
        x=[best_obj[0]], y=[best_obj[1]], z=[best_obj[2]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='cross', line=dict(width=1, color='black')),
        name='Overall Best Solution (NN Model)', hoverinfo='none'
    ))

    fig.update_layout(
        title_text='<b>Pareto Fronts: NN Model Constellation Optimization</b><br><sup>Cost vs. Revisit Time vs. Quality</sup>',
        scene=dict(xaxis_title='Total System Cost (USD)', yaxis_title='Mean Revisit Time (hr)', zaxis_title='Mean Achieved Quality'),
        legend_title_text='Number of Planes', margin=dict(l=0, r=0, b=0, t=50)
    )
    print("\nDisplaying interactive plot...")
    fig.show()

def save_results_to_csv(all_results_dict, filename="nn_optimizer_pareto_fronts.csv"):
    """
    Saves the aggregated optimization results to a CSV file.
    (Identical to baseline_optimizer.py, but default filename changed)
    """
    data_for_df = []
    for n_planes_in_run, results in all_results_dict.items():
        params_array = results['X']
        objectives_array = results['F']
        for i in range(params_array.shape[0]):
            solution_params = params_array[i]
            solution_objectives = objectives_array[i]
            row_data = {
                'num_planes': n_planes_in_run,
                'cost': solution_objectives[0],
                'revisit_time': solution_objectives[1],
                'quality': solution_objectives[2],
                'f_phasing': round(solution_params[-1])
            }
            for p_idx in range(n_planes_in_run):
                base_idx = p_idx * N_FEATURES_PER_PLANE
                row_data[f'plane_{p_idx+1}_alt'] = solution_params[base_idx + 0]
                row_data[f'plane_{p_idx+1}_inc'] = solution_params[base_idx + 1]
                row_data[f'plane_{p_idx+1}_high_q'] = round(solution_params[base_idx + 2])
                row_data[f'plane_{p_idx+1}_med_q'] = round(solution_params[base_idx + 3])
                row_data[f'plane_{p_idx+1}_low_q'] = round(solution_params[base_idx + 4])
            data_for_df.append(row_data)

    if not data_for_df:
        print("No data to save to CSV.")
        return
    df = pd.DataFrame(data_for_df)
    df.to_csv(filename, index=False)
    print(f"\nOptimization results saved to '{filename}'")

def main():
    """
    Main function to run the optimization using the Neural Network model.
    """
    if not TF_KERAS_AVAILABLE:
        print("\nAborting execution due to TensorFlow/Keras library not being available.")
        return

    model, feature_scaler, target_scaler = get_nn_model_and_scalers()
    if model is None or feature_scaler is None or target_scaler is None:
        print("Keras Model and/or scalers could not be loaded. Aborting optimization.")
        return

    all_results = {}
    # Optimize for 2 to N_TOTAL_PLANES (e.g., 2 to 10)
    for i in range(2, N_TOTAL_PLANES + 1): # Start from 2 planes
        X, F = run_optimization(i, model, feature_scaler, target_scaler)
        if F is not None and F.shape[0] > 0:
            F[:, 2] *= -1 # Quality was negated for maximization, flip back for storage/plotting
            all_results[i] = {'X': X, 'F': F}

    if not all_results:
        print("\nNo optimizations could be run or yielded results.")
        return

    best_n_planes, best_compromise_idx = find_overall_best_compromise(all_results)
    optimal_solution_params = all_results[best_n_planes]['X'][best_compromise_idx]
    optimal_solution_objectives = all_results[best_n_planes]['F'][best_compromise_idx]
    optimal_f_phasing = round(optimal_solution_params[-1])

    print("\n\n======================================================================")
    print(f" Final Recommended Design ({best_n_planes}-Plane Constellation) - NN Model ")
    print("======================================================================")
    print("\n--- Predicted Performance (Best Compromise) ---")
    print(f"  - Total System Cost:     ${optimal_solution_objectives[0]:,.2f} USD")
    print(f"  - Mean Revisit Time:     {optimal_solution_objectives[1]:.2f} hours")
    print(f"  - Mean Achieved Quality: {optimal_solution_objectives[2]:.4f}")
    print(f"  - Global F-Phasing:      {optimal_f_phasing}")

    print("\n--- Recommended Orbital Parameters ---")
    for i in range(best_n_planes):
        base_idx = i * N_FEATURES_PER_PLANE
        altitude = optimal_solution_params[base_idx + 0]
        inclination = optimal_solution_params[base_idx + 1]
        high_q = optimal_solution_params[base_idx + 2]
        med_q = optimal_solution_params[base_idx + 3]
        low_q = optimal_solution_params[base_idx + 4]
        print(f"  Plane {i+1}: Alt: {altitude:.1f} km, Inc: {inclination:.1f}°, "
              f"HighQ: {round(high_q)}, MedQ: {round(med_q)}, LowQ: {round(low_q)}")
    print("======================================================================")

    create_interactive_plot(all_results, best_n_planes, best_compromise_idx)
    save_results_to_csv(all_results, filename="nn_optimizer_pareto_fronts.csv")

if __name__ == '__main__':
    main()