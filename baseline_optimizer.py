import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import warnings
import plotly.graph_objects as go
import joblib
import plotly.io as pio

# Set default plotly template for better aesthetics
pio.templates.default = "plotly_dark"

# Try to import XGBoost and provide a helpful message if it's not installed.
try:
    import xgboost as xgb
except ImportError:
    print("Error: XGBoost library not found. Please install it using 'pip install xgboost'")
    xgb = None


# Base column names
FEATURE_COLS_BASE = [
    'altitude_plane_{}_km', 
    'inclination_plane_{}_deg',
    'high_q_plane_{}',
    'med_q_plane_{}',
    'low_q_plane_{}'
]
TARGET_COLS = ['total_system_cost_usd', 'mean_revisit_time_hr', 'mean_achieved_quality']
N_TOTAL_PLANES = 10 # The total number of planes the model is trained on
F_PHASING_BOUNDS = [0, 3] # Min and Max for f_phasing, based on training data generation
N_FEATURES_PER_PLANE = len(FEATURE_COLS_BASE) # Should be 5

SIM_SATELLITE_OPTIONS = {
    "low_quality": {
        "cost": 1.0e6,  # Revised: $2M
        "sensor_quality": 0.5,
        "instrument_fov_deg": 15.0, # Instrument's own FOV
        "max_slew_angle_deg": 20.0,  # Max angle sensor can point off-nadir
        "mass": 110 #kg
    },
    "medium_quality": {
        "cost": 3.0e6,  # Revised: $7.5M
        "sensor_quality": 0.7,
        "instrument_fov_deg": 5.0,
        "max_slew_angle_deg": 40.0,
        "mass": 225 #kg
    },
    "high_quality": {
        "cost": 8.0e6, # Revised: $15M
        "sensor_quality": 0.9,
        "instrument_fov_deg": 1.5,
        "max_slew_angle_deg": 50.0,
        "mass": 450 #kg
    }
}

LAUNCH_COST_PER_KG_USD = 5000.0 

# Mapping from optimizer's internal satellite type keys to keys in SIM_SATELLITE_OPTIONS
OPTIMIZER_TO_SIM_KEYS = {
    'high_q': 'high_quality',
    'med_q':  'medium_quality',
    'low_q':  'low_quality'
}

# Ensure satellite type keys here match those in SATELLITE_OPTIONS
SATELLITE_TYPE_KEYS = ['high_q', 'med_q', 'low_q'] 
# Corresponds to indices 2, 3, 4 in the problem variables for each plane's features


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def get_xgb_model_and_scalers():
    """
    Loads a pre-trained XGBoost model and its associated feature and target scalers from disk.
    """
    model_filename = 'multi_output_xgb_model.joblib'
    feature_scaler_filename = 'feature_scaler_xgb.joblib' # Updated name
    target_scaler_filename = 'target_scaler_xgb.joblib'   # New target scaler
    
    if xgb is None: # XGBoost is a dependency for the model
        print("XGBoost not imported. Cannot load model.")
        return None, None, None
        
    try:
        model = joblib.load(model_filename)
        feature_scaler = joblib.load(feature_scaler_filename)
        target_scaler = joblib.load(target_scaler_filename)
        print(f"Successfully loaded pre-trained model from: {model_filename}")
        print(f"Successfully loaded feature scaler from: {feature_scaler_filename}")
        print(f"Successfully loaded target scaler from: {target_scaler_filename}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_filename}', feature scaler '{feature_scaler_filename}', or target scaler '{target_scaler_filename}' not found.")
        print("Please run 'baseline_ml_model.py' first to train and save the model and scaler.")
        return None, None, None
    return model, feature_scaler, target_scaler

def calculate_system_costs(satellite_mix: dict) -> dict:
    """
    Calculates total hardware, mass, launch, and system costs.

    Args:
        satellite_mix (dict): A dictionary with the counts of each satellite type.
                              Keys should match SATELLITE_TYPE_KEYS.

    Returns:
        dict: A dictionary containing 'hardware_cost', 'total_mass_kg', 
              'launch_cost', and 'total_system_cost'.
    """
    hardware_cost = 0.0
    total_mass_kg = 0.0
    for sat_type, count in satellite_mix.items():
        sim_sat_key = OPTIMIZER_TO_SIM_KEYS.get(sat_type)
        if not sim_sat_key:
            raise KeyError(f"Satellite type key '{sat_type}' used by optimizer not found in OPTIMIZER_TO_SIM_KEYS mapping.")
        try:
            hardware_cost += SIM_SATELLITE_OPTIONS[sim_sat_key]['cost'] * count
            total_mass_kg += SIM_SATELLITE_OPTIONS[sim_sat_key]['mass'] * count # Using 'mass' as per constellation_sim.py
        except KeyError as e:
            print(f"[ERROR] The key '{e}' was not found for sim_sat_key '{sim_sat_key}' (mapped from '{sat_type}') in SIM_SATELLITE_OPTIONS.")
            print("Please ensure 'cost' and 'mass' are defined for all satellite types in constellation_sim.SATELLITE_OPTIONS.")
            raise # Re-raise the exception to stop execution if config is bad

    launch_cost = total_mass_kg * LAUNCH_COST_PER_KG_USD
    total_system_cost_variable = hardware_cost + launch_cost # Variable part of the cost
    
    return {
        "hardware_cost": hardware_cost,
        "total_mass_kg": total_mass_kg,
        "launch_cost": launch_cost,
        "total_system_cost_variable": total_system_cost_variable # Renamed for clarity
    }

class ConstellationProblem(ElementwiseProblem):
    """
    Defines the optimization problem. It adapts the input to match
    the unified model's 10-plane (20-feature) input requirement.
    """
    def __init__(self, model, feature_scaler, target_scaler, n_planes_to_optimize):
        self.model = model
        self.feature_scaler = feature_scaler
        self.n_planes_to_optimize = n_planes_to_optimize
        self.target_scaler = target_scaler
        # Total plane-specific features the ML model expects
        self.n_model_plane_features = N_TOTAL_PLANES * N_FEATURES_PER_PLANE
        # Total features ML model expects (plane features + 1 for f_phasing)
        self.n_model_total_features = self.n_model_plane_features + 1
        
        # n_var includes plane features + 1 for f_phasing
        n_var = n_planes_to_optimize * N_FEATURES_PER_PLANE + 1

        # Bounds: [alt, inc, high_q, med_q, low_q]
        # Alt: 300-800 km, Inc: 25-98 deg (as per new constraints)
        # Q-values: 0-20 (based on CSV observation, can be adjusted)
        plane_xl = [300, 25.0, 0, 0, 0]  # Updated inclination lower bound
        plane_xu = [800, 98.0, 20, 20, 20] # Updated inclination upper bound

        # Add bounds for f_phasing
        xl = np.array(plane_xl * n_planes_to_optimize + [F_PHASING_BOUNDS[0]])
        xu = np.array(plane_xu * n_planes_to_optimize + [F_PHASING_BOUNDS[1]])

        # Define constraints: 4 base constraints
        # + 2 constraints per active plane (sum_sats >= 1 and sum_sats <= 20)
        n_total_constraints = 4 + (2 * self.n_planes_to_optimize)
        super().__init__(n_var=n_var, n_obj=3, n_constr=n_total_constraints, xl=xl, xu=xu, vtype=float) # f_phasing is float, will be rounded


    def _calculate_manual_cost(self, x_problem_vars):
        """
        Calculates the system cost manually based on the number of satellites and planes.
        x_problem_vars: The decision variables for the current number of planes being optimized.
        """
        # x_problem_vars here are only the plane-specific variables, f_phasing is excluded
        # Aggregate satellite counts from all optimized planes
        satellite_counts = {key: 0 for key in SATELLITE_TYPE_KEYS}

        for p_idx in range(self.n_planes_to_optimize):
            base_idx = p_idx * N_FEATURES_PER_PLANE
            # These are the direct outputs from the optimizer for the current n_planes_to_optimize
            # Indices 2, 3, 4 correspond to high_q, med_q, low_q counts respectively
            satellite_counts['high_q'] += round(x_problem_vars[base_idx + 2])
            satellite_counts['med_q']  += round(x_problem_vars[base_idx + 3])
            satellite_counts['low_q']  += round(x_problem_vars[base_idx + 4]) # Corrected from x_problem_vars
        
        # Calculate variable costs (hardware + launch) using the provided function
        cost_components = calculate_system_costs(satellite_counts)
        variable_cost = cost_components['total_system_cost_variable']

        # Add fixed and per-plane overhead costs
        total_cost = variable_cost
        total_cost += self.n_planes_to_optimize
        
        return total_cost

    def _evaluate(self, x, out, *args, **kwargs):
        # x contains plane variables and f_phasing as the last element
        x_plane_vars = x[:-1]  # Plane-specific variables
        f_phasing_val = round(x[-1]) # Extract and round f_phasing

        # Create a full-size feature vector for the ML model, initialized with -1
        # This vector includes all N_TOTAL_PLANES plane features + 1 for f_phasing
        model_input_features = np.full(self.n_model_total_features, -1.0)
        
        # Place the optimizer's active plane variables into the start of the vector
        model_input_features[:len(x_plane_vars)] = x_plane_vars
        # Place the f_phasing value at its designated position (the last feature)
        model_input_features[self.n_model_plane_features] = f_phasing_val
        
        # Scale the full vector and predict revisit time and quality
        x_scaled = self.feature_scaler.transform(model_input_features.reshape(1, -1))
        predictions_scaled = self.model.predict(x_scaled) # Shape (1, n_targets)

        # Unscale the predictions
        predictions_unscaled = self.target_scaler.inverse_transform(predictions_scaled)
        revisit_time, quality = predictions_unscaled[0, 0], predictions_unscaled[0, 1]

        # Calculate cost manually
        cost = self._calculate_manual_cost(x_plane_vars) # Pass only plane variables to cost function

        # Pymoo minimizes objectives, so we negate quality to maximize it
        out["F"] = np.array([cost, revisit_time, -quality])

        # Constraints (g(x) <= 0)
        constraints = []

        # cost should be > 0. Using a small positive threshold for robustness.
        constraints.append(1.0 - cost)                  # cost >= 1.0
        constraints.append(0.001 - revisit_time)        # revisit_time >= 0.001 hr
        constraints.append(0.0 - quality)               # quality >= 0.0
        constraints.append(quality - 1.0)               # quality <= 1.0

        # Add constraint for each active plane: sum of satellites >= 1
        for p_idx in range(self.n_planes_to_optimize):
            base_idx = p_idx * N_FEATURES_PER_PLANE
            # x_plane_vars contains variables for active planes only
            high_q_sats = round(x_plane_vars[base_idx + 2])
            med_q_sats = round(x_plane_vars[base_idx + 3])
            low_q_sats = round(x_plane_vars[base_idx + 4])
            sum_sats_this_plane = high_q_sats + med_q_sats + low_q_sats
            constraints.append(2.0 - sum_sats_this_plane)   # sum_sats_this_plane >= 2  (becomes 2.0 - sum_sats_this_plane <= 0)
            constraints.append(sum_sats_this_plane - 20.0) # sum_sats_this_plane <= 20 (becomes sum_sats_this_plane - 20.0 <= 0)
        out["G"] = np.array(constraints)


def run_optimization(n_planes, model, feature_scaler, target_scaler):
    """
    Runs the NSGA-II optimization for a specific number of planes.
    """
    print(f"\n--- Optimizing for {n_planes} Orbital Plane(s) ---")
    problem = ConstellationProblem(model, feature_scaler, target_scaler, n_planes)
    algorithm = NSGA2(pop_size=128, eliminate_duplicates=True)
    res = minimize(problem, algorithm, ('n_gen', 312), seed=1, verbose=False)
    print("Optimization complete.")
    return res.X, res.F
    # Note: res.F will have [calculated_cost, predicted_revisit_time, -predicted_quality]
    # The quality is already negated by the problem definition for maximization.

    
def find_overall_best_compromise(all_results):
    """
    Finds the best compromise solution from ALL Pareto fronts combined.
    This helps identify which number of planes yields the most balanced design.
    """
    # Combine all objectives from all runs into a single array
    all_objectives = np.vstack([res['F'] for res in all_results.values()])
    # all_objectives columns are: [cost, revisit_time, quality (already positive)]
    
    normalized_obj = np.zeros_like(all_objectives, dtype=float)

    # Normalize Cost (minimize): (value - min) / (max - min)
    min_cost, max_cost = all_objectives[:, 0].min(), all_objectives[:, 0].max()
    range_cost = max_cost - min_cost if (max_cost - min_cost) > 1e-9 else 1.0 # Avoid division by zero
    normalized_obj[:, 0] = (all_objectives[:, 0] - min_cost) / range_cost

    # Normalize Revisit Time (minimize): (value - min) / (max - min)
    min_revisit, max_revisit = all_objectives[:, 1].min(), all_objectives[:, 1].max()
    range_revisit = max_revisit - min_revisit if (max_revisit - min_revisit) > 1e-9 else 1.0
    normalized_obj[:, 1] = (all_objectives[:, 1] - min_revisit) / range_revisit

    # Normalize Quality (maximize). We want to minimize (1 - normalized_positive_quality)
    # or equivalently, minimize (max_quality - value) / range_quality
    min_quality, max_quality = all_objectives[:, 2].min(), all_objectives[:, 2].max()
    range_quality = max_quality - min_quality if (max_quality - min_quality) > 1e-9 else 1.0
    normalized_obj[:, 2] = (max_quality - all_objectives[:, 2]) / range_quality
    
    # Calculate Euclidean distance to the ideal point (0,0,0) in normalized space
    distances = np.linalg.norm(normalized_obj, axis=1)
    best_global_idx = np.argmin(distances)
    
    # Identify which run the best point belongs to
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
    Generates an interactive 3D plot of all Pareto fronts using Plotly.
    """
    fig = go.Figure()

    # Define a color scale for the number of planes
    colors = pio.templates['plotly_dark'].layout.colorway

    # Add traces for each number of planes 
    for n_planes, results in all_results.items():
        objectives = results['F']
        params = results['X']
        # objectives columns are already: [cost, revisit_time, quality (positive)]
        # Create detailed hover text for each point
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
            x=objectives[:, 0],
            y=objectives[:, 1],
            z=objectives[:, 2],
            mode='markers',
            marker=dict(
                size=5 if is_optimal_front else 3.5,
                color=colors[(n_planes - 1) % len(colors)], # n_planes is 1-indexed
                opacity=1.0 if is_optimal_front else 0.5,
                symbol='diamond' if is_optimal_front else 'circle'
            ),
            name=f'{n_planes} Planes' + (' (Optimal Front)' if is_optimal_front else ''),
            hovertext=hover_texts,
            hoverinfo='text'
        ))

    # Highlight the single best compromise solution
    best_front_results = all_results[best_n_planes]
    best_obj = best_front_results['F'][best_solution_idx]
    
    fig.add_trace(go.Scatter3d(
        x=[best_obj[0]], y=[best_obj[1]], z=[best_obj[2]],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='cross',
            line=dict(width=1, color='black')
        ),
        name='Overall Best Solution',
        hoverinfo='none' # Hover info is already on the point from its original trace
    ))
    
    # Update layout for a clean, professional look
    fig.update_layout(
        title_text='<b>Pareto Fronts: Constellation Design Optimization</b><br><sup>Cost vs. Revisit Time vs. Quality</sup>',
        scene=dict(
            xaxis_title='Total System Cost (USD)',
            yaxis_title='Mean Revisit Time (hr)',
            zaxis_title='Mean Achieved Quality'
        ),
        legend_title_text='Number of Planes',
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    print("\nDisplaying interactive plot...")
    fig.show()

def save_results_to_csv(all_results_dict, filename="optimizer_pareto_fronts.csv"):
    """
    Saves the aggregated optimization results to a CSV file.

    Args:
        all_results_dict (dict): Dictionary containing results from all optimization runs.
                                 Keys are n_planes, values are dicts {'X': params, 'F': objectives}.
        filename (str): The name of the CSV file to save.
    """
    data_for_df = []
    max_planes_overall = N_TOTAL_PLANES # Used to ensure all potential plane columns are considered

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
                'quality': solution_objectives[2], # Assumes quality is already positive
                'f_phasing': round(solution_params[-1]) # Add f_phasing
            }

            # Add parameters for each plane in the current solution
            # solution_params[:-1] are the plane-specific params
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
    Main function to run the cumulative optimization and visualization.
    """
    if xgb is None: # Check if XGBoost was imported successfully
        print("\nAborting execution due to XGBoost library not being available.")
        return
        
    model, feature_scaler, target_scaler = get_xgb_model_and_scalers() # Load model and scalers
    if model is None or feature_scaler is None or target_scaler is None:
        print("Model and/or scalers could not be loaded. Aborting optimization.")
        return
        
    all_results = {}
    # Optimize for 2 to N_TOTAL_PLANES (e.g., 2 to 10)
    for i in range(2, N_TOTAL_PLANES + 1):
        X, F = run_optimization(i, model, feature_scaler, target_scaler)
        if F is not None and F.shape[0] > 0:
            # Cost (F[:,0]) is already calculated manually and positive.
            # Quality (F[:,2]) was -quality from problem, so flip it back for storage/plotting.
            F[:, 2] *= -1
            all_results[i] = {'X': X, 'F': F}

    if not all_results:
        print("\nNo optimizations could be run.")
        return

    # Find the single best solution across all runs
    best_n_planes, best_compromise_idx = find_overall_best_compromise(all_results)
    
    # Extract details of the optimal solution
    optimal_solution_params = all_results[best_n_planes]['X'][best_compromise_idx]
    optimal_solution_objectives = all_results[best_n_planes]['F'][best_compromise_idx]
    optimal_f_phasing = round(optimal_solution_params[-1])

    # --- Final Recommendation ---
    print("\n\n==========================================================")
    print(f"      Final Recommended Design ({best_n_planes}-Plane Constellation)      ")
    print("==========================================================")
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
              # Rounding Q-values as they represent counts of satellites


    print("==========================================================")
    
    # --- Interactive Visualization ---
    create_interactive_plot(all_results, best_n_planes, best_compromise_idx)

    # --- Save all results to CSV ---
    save_results_to_csv(all_results)


if __name__ == '__main__':
    main()
