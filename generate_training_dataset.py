import csv
import numpy as np
import random
from astropy import units as u
from astropy.coordinates import EarthLocation
import sys
import os
from contextlib import contextmanager
from tqdm import tqdm # For progress bar
import multiprocessing
from functools import partial
import time

# IMPORTANT: Before running this script, you must rename your original file
# from 'python-qec-deo.py' to 'constellation_sim.py' and ensure it contains
# a 'mass' key for each entry in SATELLITE_OPTIONS.
try:
    import constellation_sim as sim
except ImportError:
    print("Error: Could not import 'constellation_sim'.")
    print("Please make sure you have renamed 'python-qec-deo.py' to 'constellation_sim.py' and it is in the same directory.")
    exit()

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull."""
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# --- Cost Model Configuration ---
# You can adjust this value to reflect different launch market prices.
# This is a key assumption in your model.
LAUNCH_COST_PER_KG_USD = 5000.0 

def generate_random_satellite_mix(total_sats: int) -> dict:
    """
    Generates a random distribution of satellite types for a given total number.
    Ensures that the sum of satellite types equals the total number of satellites.

    Args:
        total_sats (int): The total number of satellites in the constellation.

    Returns:
        dict: A dictionary with the counts for "high_quality", "medium_quality",
              and "low_quality" satellites.
    """
    if total_sats <= 0:
        return {"high_quality": 0, "medium_quality": 0, "low_quality": 0}

    # Generate three random numbers
    r1, r2, r3 = random.random(), random.random(), random.random()
    
    # Normalize them to sum to 1
    total_r = r1 + r2 + r3
    p1, p2, p3 = r1 / total_r, r2 / total_r, r3 / total_r
    
    # Allocate satellites based on proportions
    counts = {
        "high_quality": int(np.floor(p1 * total_sats)),
        "medium_quality": int(np.floor(p2 * total_sats)),
        "low_quality": 0 # The rest go to low_quality
    }

    # Assign remaining satellites due to flooring to low_quality
    assigned_sats = counts["high_quality"] + counts["medium_quality"]
    counts["low_quality"] = total_sats - assigned_sats
    
    # A final check to ensure the sum is correct
    final_sum = sum(counts.values())
    if final_sum != total_sats:
        # Adjust the largest category if there's a small discrepancy
        diff = total_sats - final_sum
        max_key = max(counts, key=counts.get)
        counts[max_key] += diff

    return counts

def calculate_system_costs(satellite_mix: dict) -> dict:
    """
    Calculates total hardware, mass, launch, and system costs.

    Args:
        satellite_mix (dict): A dictionary with the counts of each satellite type.

    Returns:
        dict: A dictionary containing 'hardware_cost', 'total_mass', 
              'launch_cost', and 'total_system_cost'.
    """
    hardware_cost = 0.0
    total_mass = 0.0
    for sat_type, count in satellite_mix.items():
        try:
            hardware_cost += sim.SATELLITE_OPTIONS[sat_type]['cost'] * count
            total_mass += sim.SATELLITE_OPTIONS[sat_type]['mass'] * count
        except KeyError as e:
            print(f"[ERROR] The key '{e}' was not found in SATELLITE_OPTIONS for type '{sat_type}'.")
            print("Please ensure 'cost' and 'mass' are defined for all satellite types in constellation_sim.py.")
            raise
            
    launch_cost = total_mass * LAUNCH_COST_PER_KG_USD
    total_system_cost = hardware_cost + launch_cost
    
    return {
        "hardware_cost": hardware_cost,
        "total_mass_kg": total_mass,
        "launch_cost": launch_cost,
        "total_system_cost": total_system_cost
    }

def generate_single_sample_data(sample_index: int, param_ranges: dict, ground_grid_params: dict, max_planes: int, header_len: int):
    """
    Generates a single row of training data.
    This function is designed to be called by a multiprocessing worker.
    """
    try:
        # 1. Randomly select constellation parameters
        num_planes = random.randint(*param_ranges["num_planes"])
        f_phasing = random.randint(*param_ranges["f_phasing"]) 

        altitudes_km_list = [random.uniform(*param_ranges["altitude_km"]) for _ in range(num_planes)]
        inclinations_deg_list = [random.uniform(*param_ranges["inclination_deg"]) for _ in range(num_planes)]
        sats_per_plane_list = [random.randint(*param_ranges["sats_per_plane"]) for _ in range(num_planes)] # Ensure sats_per_plane > 0
        
        # 2. Build the plane definitions and determine per-plane satellite mixes
        plane_defs = []
        per_plane_satellite_distributions = [] # List of dicts, one for each plane
        aggregated_sat_mix_for_costs = {"high_quality": 0, "medium_quality": 0, "low_quality": 0}
        total_sats_constellation = 0

        for i_plane in range(num_planes):
            sats_this_plane_count = sats_per_plane_list[i_plane]
            if sats_this_plane_count == 0: # Should not happen if param_ranges["sats_per_plane"][0] > 0
                # Handle case of zero satellites in a plane if it's possible by design
                # For now, assume sats_per_plane_list will always have > 0 satellites
                pass
            total_sats_constellation += sats_this_plane_count
            
            sat_mix_for_this_plane = generate_random_satellite_mix(sats_this_plane_count)
            per_plane_satellite_distributions.append(sat_mix_for_this_plane)

            for sat_type, count in sat_mix_for_this_plane.items():
                aggregated_sat_mix_for_costs[sat_type] += count
            
            plane_defs.append({
                "altitude": altitudes_km_list[i_plane] * u.km, 
                "inclination": inclinations_deg_list[i_plane] * u.deg, 
                "sats_in_plane": sats_this_plane_count, # For validation or direct use
                "satellite_counts_for_plane": sat_mix_for_this_plane
                # "phase_value" per plane removed
            })
        costs = calculate_system_costs(aggregated_sat_mix_for_costs)

        # 3. Run the simulation
        # Pass the global f_phasing to the simulation
        constellation = sim.build_custom_constellation(plane_definitions=plane_defs, global_f_phasing=f_phasing)
        
        # Recreate ground_grid for this worker (or ensure it's picklable if passed directly)
        # For simplicity, recreating here based on params.
        lats_g = np.linspace(*ground_grid_params["lats_range"], ground_grid_params["lats_points"]) * u.deg
        lons_g = np.linspace(*ground_grid_params["lons_range"], ground_grid_params["lons_points"], endpoint=ground_grid_params["lons_endpoint"]) * u.deg
        current_ground_grid = [EarthLocation(lon=lon, lat=lat) for lat in lats_g for lon in lons_g]

        with suppress_stdout_stderr():
            mean_revisit_hr, achieved_qualities, num_accessed_gp = sim.calculate_mean_revisit_time(
                constellation, 
                current_ground_grid, 
                duration_days=2,
                time_step_minutes=15,
                min_achieved_image_quality=0.001
            )

        # Check if all ground points were accessed
        required_accessed_points = 0.8 * len(current_ground_grid)
        if num_accessed_gp < required_accessed_points:
            # print(f"[INFO] Worker {sample_index}: Not enough ground points accessed ({num_accessed_gp}/{len(current_ground_grid)}). Required: {required_accessed_points:.0f}. Marking as invalid.")
            # This print can be noisy if many samples fail this check and is commented out for large runs.
            # It's useful for initial debugging but can be commented out for large runs.
            return [-1] * header_len # Return an error row, sample is invalid

        mean_revisit_val = mean_revisit_hr.to_value(u.hr) if mean_revisit_hr is not np.nan and not np.isnan(mean_revisit_hr.value) else -1.0
        mean_quality_val = np.mean(achieved_qualities) if achieved_qualities else 0.0
        
        plane_specific_data_row = []
        for i_plane in range(max_planes):
            if i_plane < num_planes:
                # Use the generated per-plane distribution for CSV
                counts_this_plane_csv = per_plane_satellite_distributions[i_plane]
                plane_specific_data_row.extend([
                    round(altitudes_km_list[i_plane], 2),
                    round(inclinations_deg_list[i_plane], 2),
                    sats_per_plane_list[i_plane], # sats_plane_i
                    # plane_defs[i_plane]['phase_value'], # Removed per-plane phase value
                    counts_this_plane_csv.get("high_quality", 0),
                    counts_this_plane_csv.get("medium_quality", 0),
                    counts_this_plane_csv.get("low_quality", 0)
                ])
            else:
                plane_specific_data_row.extend([-1.0, -1.0, -1, -1, -1, -1]) # Placeholder for non-existent planes

        base_data_row_part1 = [total_sats_constellation, num_planes, f_phasing] 
        base_data_row_part2 = [
            aggregated_sat_mix_for_costs['high_quality'], 
            aggregated_sat_mix_for_costs['medium_quality'], aggregated_sat_mix_for_costs['low_quality'],
            round(costs['total_mass_kg'], 2), round(costs['hardware_cost'], 2), # costs are from aggregated_sat_mix_for_costs
            round(costs['launch_cost'], 2), round(costs['total_system_cost'], 2), # costs are from aggregated_sat_mix_for_costs
            round(mean_revisit_val, 4), round(mean_quality_val, 4)
        ]
        return base_data_row_part1 + plane_specific_data_row + base_data_row_part2
    except Exception as e:
        print(f"[ERROR] Worker {sample_index}: An error occurred: {e}") # Log worker errors
        import traceback
        traceback.print_exc() # This will print the full traceback
        return [-1] * header_len # Return an error row


def generate_training_data(num_samples: int, output_filename="constellation_training_data.csv"):
    """
    Generates a CSV file with training data for constellation optimization.

    Args:
        num_samples (int): The number of random constellation configurations to generate.
        output_filename (str): The name of the output CSV file.
    """
    # Define the parameter space for random generation
    # Note: PARAM_RANGES["num_planes"][1] will define the max columns for per-plane params
    PARAM_RANGES = {
        "num_planes": (2, 10),           # Range for random integer
        "sats_per_plane": (2, 20),       # Range for random integer (ensure > 0)
        "altitude_km": (300.0, 800.0),   # Range for random real number (float)
        "inclination_deg": (25.0, 98.0), # Range for random real number (float)
        "f_phasing": (0, 3)              # Global F phasing parameter (0 to MAX_PLANES-1 typically)
    }
    MAX_PLANES = PARAM_RANGES["num_planes"][1]

    # Define a fixed grid of ground points for consistent analysis across samples
    # Store parameters for recreating ground_grid in workers
    ground_grid_params = {
        "lats_range": (-60.0, 60.0),
        "lats_points": 6, # Number of latitude points
        "lons_range": (-180.0, 180.0),
        "lons_points": 10,  # Number of longitude points
        "lons_endpoint": False
    }
    # Note: The actual ground_grid object is not passed to workers to avoid pickling issues with astropy objects.
    # It will be recreated within each worker based on ground_grid_params.

    # --- CSV File Setup ---
    base_header = [
        # Input Features (Constellation Design)
        'total_sats', 'num_planes',
        'f_phasing', # Added global f_phasing
        'num_high_quality', 'num_medium_quality', 'num_low_quality',
        # Input Features (Cost Drivers)
        'total_mass_kg', 'hardware_cost_usd', 'launch_cost_usd',
        # Output Targets
        'total_system_cost_usd', 'mean_revisit_time_hr', 'mean_achieved_quality'
    ]

    plane_specific_columns_header = []
    for i in range(1, MAX_PLANES + 1):
        plane_specific_columns_header.append(f'altitude_plane_{i}_km')
        plane_specific_columns_header.append(f'inclination_plane_{i}_deg')
        plane_specific_columns_header.append(f'sats_plane_{i}')
        # plane_specific_columns_header.append(f'phase_val_plane_{i}') # Removed per-plane phase
        plane_specific_columns_header.append(f'high_q_plane_{i}')
        plane_specific_columns_header.append(f'med_q_plane_{i}')
        plane_specific_columns_header.append(f'low_q_plane_{i}')
        
    # Insert plane-specific columns after 'num_planes' and before 'num_high_quality'
    # base_header[:2] gives ['total_sats', 'num_planes']
    # base_header[2:] gives ['num_high_quality', ..., 'mean_achieved_quality']
    header = base_header[:3] + plane_specific_columns_header + base_header[3:] # Adjusted index for f_phasing
    
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # --- Parallel Data Generation ---
        # Determine number of processes (e.g., number of CPU cores - 1, or a fixed number)
        # num_processes = multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
        num_processes = multiprocessing.cpu_count() # Use all available CPU cores
        print(f"Using {num_processes} processes for data generation (all available cores).")

        # Create a partial function to pass fixed arguments to the worker
        worker_func = partial(generate_single_sample_data, 
                              param_ranges=PARAM_RANGES, 
                              ground_grid_params=ground_grid_params, 
                              max_planes=MAX_PLANES,
                              header_len=len(header))

        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use imap_unordered to get results as they complete, good for progress bars
            # The first argument to imap_unordered is the function to call,
            # the second is an iterable of arguments for each call (here, just sample indices).
            results = pool.imap_unordered(worker_func, range(num_samples))
            
            for data_row in tqdm(results, total=num_samples, desc="Generating Samples", unit="sample"):
                writer.writerow(data_row)
            
    print(f"\n\n{'='*20} DATA GENERATION COMPLETE {'='*20}")
    print(f"Successfully created '{output_filename}' with {num_samples} samples.")


if __name__ == '__main__':
    # --- Configuration ---
    # For a robust ML model, you might need hundreds or thousands of samples,
    # which can take a very long time to generate.
    NUMBER_OF_SAMPLES_TO_GENERATE = 15_000 # Adjust as needed

    generate_training_data(NUMBER_OF_SAMPLES_TO_GENERATE)
