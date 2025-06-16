import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, GCRS, ITRS
from astropy.time import Time, TimeDelta
from astropy.coordinates import Angle # Import Angle
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.core.propagation import func_twobody # Import func_twobody
from poliastro.spheroid_location import SpheroidLocation

# --- 1. Define Satellite Hardware Options ---
# This dictionary holds the trade-space for our satellite hardware.
# FOV is the full cone angle of the sensor in degrees.
SATELLITE_OPTIONS = {
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

# --- Constants for Quality Calculation ---
REFERENCE_ALTITUDE_FOR_QUALITY_KM = 300.0 # km, sensor quality is defined at this altitude


def build_custom_constellation(plane_definitions: list, global_f_phasing: int):
    """
    Builds a custom constellation based on per-plane definitions.
    
    Args:
        plane_definitions (list): A list of dicts, where each dict defines a plane.
                                  Each dict must include "altitude", "inclination", 
                                  "satellite_counts_for_plane" (e.g., {"high_quality": 2, ...}).
                                  Optionally, "sats_in_plane" can be provided for validation.
        global_f_phasing (int): The global phasing parameter (F in T/P/F Walker notation)
                                which determines the phase shift between planes.

    Returns:
        list: A list of tuples, where each tuple is 
              (poliastro.twobody.Orbit, dict_of_properties).
    """
    constellation = []
    num_planes = len(plane_definitions)
    
    # Calculate RAAN (Right Ascension of the Ascending Node) for each plane
    plane_raans = np.linspace(0, 360, num_planes, endpoint=False) * u.deg

    total_sats_in_constellation = 0
    for plane_def_check in plane_definitions:
        if "satellite_counts_for_plane" not in plane_def_check:
            raise ValueError("Each plane definition must include 'phase_value'.")
        sats_in_this_plane_from_counts = sum(plane_def_check["satellite_counts_for_plane"].values())
        if plane_def_check.get('sats_in_plane') is not None and plane_def_check['sats_in_plane'] != sats_in_this_plane_from_counts:
            raise ValueError("ERROR")
        total_sats_in_constellation += sats_in_this_plane_from_counts

    for i, plane_def in enumerate(plane_definitions):
        plane_index = i # 0-indexed plane number
        raan = plane_raans[i]
        altitude = plane_def['altitude']
        inclination = plane_def['inclination']
        sats_dist_for_this_plane = plane_def['satellite_counts_for_plane']
        sats_in_this_plane_count = sum(sats_dist_for_this_plane.values())
        if sats_in_this_plane_count == 0:
            continue # Skip plane if no satellites

        # Create satellite type list for THIS plane with specific ordering
        satellite_type_list_for_plane = []
        
        # Make copies of counts to decrement
        current_counts = sats_dist_for_this_plane.copy()
        
        # Define the order of preference for each "round" of placement
        placement_cycle = ["high_quality", "medium_quality", "low_quality"]
        
        # Keep track of which type to try next in the cycle
        cycle_idx = 0
        
        for _ in range(sats_in_this_plane_count):
            placed_satellite_for_slot = False
            # Attempt to place a satellite for the current slot
            # Iterate through the placement_cycle (potentially multiple times if types are exhausted)
            # to find an available satellite type.
            for _attempt in range(len(placement_cycle)):
                sat_type_to_try = placement_cycle[cycle_idx]
                if current_counts.get(sat_type_to_try, 0) > 0:
                    satellite_type_list_for_plane.append(sat_type_to_try)
                    current_counts[sat_type_to_try] -= 1
                    placed_satellite_for_slot = True
                    cycle_idx = (cycle_idx + 1) % len(placement_cycle) # Move to next type for *next* slot
                    break # Satellite placed for this slot
                else:
                    # This type is exhausted, try the next type in the cycle for *this* slot
                    cycle_idx = (cycle_idx + 1) % len(placement_cycle)
            
            if not placed_satellite_for_slot:
                # This should not be reached if sats_in_this_plane_count is the sum
                # of initial satellite counts, as the logic above should always find a satellite
                # if any are remaining to be placed.
                # Consider raising an error or logging if this state is unexpectedly hit.
                print(f"[ERROR] Could not place a satellite for a slot in plane {i}, though {sats_in_this_plane_count - len(satellite_type_list_for_plane)} should remain.")
                break # Avoid infinite loop if something is wrong with counts

        # Calculate true anomaly for each satellite in the plane
        mean_anomalies = np.linspace(0, 360, sats_in_this_plane_count, endpoint=False) * u.deg
        
        for j in range(sats_in_this_plane_count):
            # True anomaly for j-th satellite in i-th plane:
            # nu_ij = M_j + (i * F * 360/T)
            # M_j is the mean anomaly of the j-th satellite in its plane (mean_anomalies[j])
            # i is plane_index
            # F is global_f_phasing
            # T is total_sats_in_constellation
            inter_plane_phasing_offset = (plane_index * global_f_phasing * 360 / total_sats_in_constellation) * u.deg if total_sats_in_constellation > 0 else 0 * u.deg
            calculated_nu_val = mean_anomalies[j] + inter_plane_phasing_offset
            nu_angle = Angle(calculated_nu_val).wrap_at(180 * u.deg)

            orbit = Orbit.from_classical(
                attractor=Earth,
                a=Earth.R + altitude,
                ecc=0 * u.one,
                inc=inclination,
                raan=raan,
                argp=0 * u.deg,
                nu=nu_angle # Use the wrapped angle
            )
            
            current_sat_type_for_orbit = satellite_type_list_for_plane[j]
            # Make a copy to avoid modifying the global SATELLITE_OPTIONS
            satellite_properties = SATELLITE_OPTIONS[current_sat_type_for_orbit].copy()
            satellite_properties['name'] = current_sat_type_for_orbit # Add name for logging
            
            constellation.append((orbit, satellite_properties))
            
    return constellation

def calculate_mean_revisit_time(constellation: list, ground_points: list, 
                                duration_days=1, time_step_minutes=5, 
                                min_achieved_image_quality=0.0):
    """
    Calculates the mean revisit time for a constellation over a grid of ground points.
    considering a minimum achieved image quality for valid accesses.
    Image quality is degraded by the cosine of the off-nadir viewing angle.
    """
    print("Starting Mean Revisit Time calculation for HETEROGENEOUS constellation...")
    print(f"Constellation size: {len(constellation)} satellites")
    print(f"Ground points to check: {len(ground_points)}")
    print(f"Simulation duration: {duration_days} days, Time step: {time_step_minutes} minutes")
    print(f"Minimum Achieved Image Quality required: {min_achieved_image_quality}")

    start_time = Time("2025-01-01 00:00:00", scale="utc")
    end_time = start_time + TimeDelta(duration_days * u.day)
    
    # Define time steps for propagation
    total_duration_seconds = (end_time - start_time).to(u.s)
    time_step_seconds = (time_step_minutes * u.min).to(u.s)
    num_time_steps = int(total_duration_seconds / time_step_seconds) + 1
    times_jd = np.linspace(start_time.jd, end_time.jd, num_time_steps)
    propagation_times = Time(times_jd, format='jd', scale='utc')

    all_revisit_gaps = []
    all_achieved_qualities = [] # New list to store achieved image qualities
    accessed_ground_point_indices = set() # To track unique accessed ground points

    # Pre-propagate all satellite positions and transform to ITRS
    print("Pre-propagating satellite orbits and transforming to ITRS...")
    sat_itrs_positions_list = []  # List of arrays, each (N_times, 3) for a satellite
    # The print statement for each satellite propagation is removed for cleaner output,
    # especially when this function is called by workers.
    for orbit, sat_props in constellation:
        # Manual propagation loop instead of orbit.sample()
        r_gcrs_list_for_orbit_km = []
        for t_epoch in propagation_times:
            # Calculate time delta from the orbit's own epoch
            time_delta_from_orbit_epoch = t_epoch - orbit.epoch
            # Propagate the orbit to this specific time
            propagated_state_at_t = orbit.propagate(time_delta_from_orbit_epoch)
            # Get the position vector (x,y,z) in km; .r is a CartesianRepresentation
            r_gcrs_km_at_t = propagated_state_at_t.r.to_value(u.km)
            r_gcrs_list_for_orbit_km.append(r_gcrs_km_at_t)
        
        # Convert list of [x,y,z] arrays to a single (N_points, 3) numpy array
        positions_gcrs_xyz_km = np.array(r_gcrs_list_for_orbit_km)

        coords_gcrs = SkyCoord(x=positions_gcrs_xyz_km[:, 0] * u.km,
                               y=positions_gcrs_xyz_km[:, 1] * u.km,
                               z=positions_gcrs_xyz_km[:, 2] * u.km,
                               frame=GCRS(obstime=propagation_times), # Use the array of times
                               representation_type='cartesian')
        coords_itrs = coords_gcrs.transform_to(ITRS(obstime=propagation_times))
        sat_itrs_positions_list.append(coords_itrs.cartesian.xyz.to_value(u.m).T)

    earth_R_m = Earth.R.to_value(u.m)
    # Pass reference altitude as a scalar to JIT function
    reference_altitude_km_val = REFERENCE_ALTITUDE_FOR_QUALITY_KM  # Ensure this is defined for JIT call

    for i_gp, ground_point_loc in enumerate(ground_points):  # ground_point_loc is EarthLocation
        print(f"  - Calculating accesses for ground point {i_gp+1}/{len(ground_points)} ({ground_point_loc.lat:.1f}, {ground_point_loc.lon:.1f})...")
        
        gp_itrs_cartesian = ground_point_loc.to_geocentric()
        gp_pos_itrs_m = np.array([gp_itrs_cartesian[0].to_value(u.m),
                                  gp_itrs_cartesian[1].to_value(u.m),
                                  gp_itrs_cartesian[2].to_value(u.m)])
        gp_norm_itrs_m = np.linalg.norm(gp_pos_itrs_m)
        if gp_norm_itrs_m == 0: continue # Should not happen for valid EarthLocation
        gp_local_zenith_vec = gp_pos_itrs_m / gp_norm_itrs_m

        point_access_times_jd = []

        for t_idx, current_time in enumerate(propagation_times):
            accessed_at_this_step = False
            for sat_idx, (orbit, sat_props) in enumerate(constellation):
                if accessed_at_this_step:
                    break

                sat_pos_itrs_m = sat_itrs_positions_list[sat_idx][t_idx, :]

                # --- Visibility Check 1: Satellite above ground point's horizon ---
                vec_gp_to_sat_m = sat_pos_itrs_m - gp_pos_itrs_m
                if np.dot(vec_gp_to_sat_m, gp_local_zenith_vec) <= 0: # Using 0 deg elevation threshold
                    continue # Satellite is below or at horizon for this GP

                # --- Visibility Check 2: Ground point within satellite's slewing capability ---
                vec_sat_to_gp_m = gp_pos_itrs_m - sat_pos_itrs_m
                norm_vec_sat_to_gp_m = np.linalg.norm(vec_sat_to_gp_m)
                if norm_vec_sat_to_gp_m < 1e-9: continue # Satellite is at the ground point (very unlikely)

                vec_sat_nadir_approx_m = -sat_pos_itrs_m 
                norm_vec_sat_nadir_approx_m = np.linalg.norm(vec_sat_nadir_approx_m)
                if norm_vec_sat_nadir_approx_m < 1e-9: continue # Satellite at Earth's center (impossible)

                cos_angle_to_nadir = np.dot(vec_sat_to_gp_m, vec_sat_nadir_approx_m) / \
                                     (norm_vec_sat_to_gp_m * norm_vec_sat_nadir_approx_m)
                
                # Clip cos_angle_to_nadir to avoid domain errors with arccos
                clipped_cos_angle_val = np.clip(cos_angle_to_nadir, -1.0, 1.0)
                angle_to_nadir_rad = np.arccos(clipped_cos_angle_val)

                max_slew_angle_rad = (sat_props['max_slew_angle_deg'] * u.deg).to_value(u.rad)

                if angle_to_nadir_rad <= max_slew_angle_rad:
                    # --- Calculate Achieved Image Quality (GSD-based Model) ---
                    base_sensor_quality = sat_props['sensor_quality']
                    current_radius_m = np.linalg.norm(sat_pos_itrs_m)
                    current_altitude_m = current_radius_m - earth_R_m
                    
                    altitude_scaling_factor = 0.01 # Default for problematic altitude
                    if current_altitude_m > 1e-3: # Ensure altitude is positive
                        altitude_scaling_factor = (REFERENCE_ALTITUDE_FOR_QUALITY_KM * 1000.0) / current_altitude_m
                    
                    off_nadir_degradation = np.cos(angle_to_nadir_rad) # Cosine of the off-nadir angle
                    
                    achieved_image_quality = base_sensor_quality * altitude_scaling_factor * off_nadir_degradation

                    if achieved_image_quality >= min_achieved_image_quality:
                        accessed_at_this_step = True
                        all_achieved_qualities.append(achieved_image_quality) # Store quality
                        point_access_times_jd.append(current_time.jd)
                        accessed_ground_point_indices.add(i_gp) # Mark this ground point as accessed
                        break # One valid access is enough for this time step

        
        if len(point_access_times_jd) > 1:
            access_time_objects = Time(np.sort(list(set(point_access_times_jd))), format='jd', scale='utc') # Ensure sorted unique times
            delta_t_values = np.diff(access_time_objects)

            if isinstance(delta_t_values, TimeDelta):
                # This is the most common and expected case for modern Astropy versions,
                # where np.diff on a Time array returns a single TimeDelta object.
                gaps_quantity = delta_t_values
            elif isinstance(delta_t_values, u.Quantity):
                # Fallback if it's some other Quantity type (less likely from np.diff(Time))
                gaps_quantity = delta_t_values
            elif isinstance(delta_t_values, np.ndarray) and delta_t_values.dtype == object:
                # This case is reported by you as frequent.
                # The elements are expected to be TimeDelta objects.
                # We need to explicitly handle this list/array of TimeDelta objects
                # by ensuring they are consistently formatted for the Quantity constructor.
                print("[INFO] np.diff(Time) returned a NumPy array of TimeDelta objects. Processing element-wise.")
                if len(delta_t_values) > 0:
                    # Ensure elements are indeed TimeDelta before processing
                    if all(isinstance(td, TimeDelta) for td in delta_t_values):
                        # Convert each TimeDelta to a common unit (e.g., days).
                        # This results in a list of TimeDelta objects, all in the same unit,
                        # from which u.Quantity can reliably construct a single Quantity array.
                        gaps_quantity = u.Quantity([td.to(u.day) for td in delta_t_values])
                    else:
                        raise TypeError(
                            "NumPy array (dtype=object) from np.diff(Time) "
                            "does not contain TimeDelta objects as expected."
                        )
                else:
                    gaps_quantity = u.Quantity([], unit=u.day) # Handle empty array case
            elif isinstance(delta_t_values, np.ndarray):
                # Numeric dtype: assumes raw numbers are differences in days (from JD)
                print("[INFO] np.diff(Time) returned a NumPy array of numbers. Assuming units of days.")
                gaps_quantity = u.Quantity(delta_t_values, unit=u.day)
            else:
                # Fallback for unexpected types
                raise TypeError(
                    f"Unexpected type for time differences: {type(delta_t_values)}. "
                    "Expected Astropy TimeDelta, Quantity, or NumPy array."
                )
            
            gaps_hr_quantity = gaps_quantity.to(u.hr)
            all_revisit_gaps.extend(gaps_hr_quantity.value)
        elif len(point_access_times_jd) == 1:
            print(f"    Note: Ground point {i_gp+1} accessed only once. No revisit gap calculated for this point.")
        else:
            print(f"    Note: Ground point {i_gp+1} never accessed.")

    num_unique_ground_points_accessed = len(accessed_ground_point_indices)

    if not all_revisit_gaps:
        print("\n[WARNING] No revisit gaps were calculated. This could mean no ground point was accessed more than once, or no accesses at all.")
        return np.nan * u.hr, all_achieved_qualities, num_unique_ground_points_accessed

    mean_revisit_hr_val = np.mean(all_revisit_gaps)
    median_revisit_hr_val = np.median(all_revisit_gaps)
    max_revisit_hr_val = np.max(all_revisit_gaps)
    print(f"\n--- Revisit Statistics ({len(all_revisit_gaps)} gaps from {len(all_achieved_qualities)} valid accesses) ---")
    print(f"Mean Revisit Time: {mean_revisit_hr_val:.2f} hr")
    print(f"Median Revisit Time: {median_revisit_hr_val:.2f} hr")
    print(f"Max Revisit Time: {max_revisit_hr_val:.2f} hr")

    if all_achieved_qualities:
        mean_img_quality = np.mean(all_achieved_qualities)
        median_img_quality = np.median(all_achieved_qualities)
        min_img_quality = np.min(all_achieved_qualities)
        max_img_quality = np.max(all_achieved_qualities)
        print(f"\n--- Achieved Image Quality Statistics (for {len(all_achieved_qualities)} valid accesses) ---")
        print(f"Mean Achieved Image Quality: {mean_img_quality:.3f}")
        print(f"Median Achieved Image Quality: {median_img_quality:.3f}")
        print(f"Min Achieved Image Quality: {min_img_quality:.3f}")
        print(f"Max Achieved Image Quality: {max_img_quality:.3f}")
    print(f"Number of unique ground points accessed: {num_unique_ground_points_accessed} / {len(ground_points)}")

    return mean_revisit_hr_val * u.hr, all_achieved_qualities, num_unique_ground_points_accessed


if __name__ == '__main__':
    # --- Example Usage for a Custom Mixed Constellation ---
    
   # Define the specific parameters for each orbital plane
    plane_params = [
        # Plane 1
        {"altitude": 650 * u.km, "inclination": 75 * u.deg, "sats_in_plane": 10,
         "satellite_counts_for_plane": {"high_quality": 2, "medium_quality": 5, "low_quality": 3}},
        # Plane 2
        {"altitude": 550 * u.km, "inclination": 97.6 * u.deg, "sats_in_plane": 15,
         "satellite_counts_for_plane": {"high_quality": 5, "medium_quality": 5, "low_quality": 5}},
        # Plane 3
        {"altitude": 750 * u.km, "inclination": 55 * u.deg, "sats_in_plane": 8,
         "satellite_counts_for_plane": {"high_quality": 3, "medium_quality": 3, "low_quality": 2}},
        # Plane 4
        {"altitude": 450 * u.km, "inclination": 25 * u.deg, "sats_in_plane": 12,
         "satellite_counts_for_plane": {"high_quality": 2, "medium_quality": 4, "low_quality": 6}},
        # Plane 5
        {"altitude": 600 * u.km, "inclination": 40 * u.deg, "sats_in_plane": 10,
         "satellite_counts_for_plane": {"high_quality": 1, "medium_quality": 4, "low_quality": 5}},
        # Plane 6
        {"altitude": 700 * u.km, "inclination": 85 * u.deg, "sats_in_plane": 10,
         "satellite_counts_for_plane": {"high_quality": 2, "medium_quality": 4, "low_quality": 4}},
    ]
    # Total satellites = 10+15+8+12+10+10 = 65
    global_f_phasing_example = 1 # Example F parameter for Walker Delta

    # Build the constellation
    custom_constellation = build_custom_constellation(plane_definitions=plane_params, global_f_phasing=global_f_phasing_example)
 
     # Create a simple grid of ground points for analysis
    lats = np.linspace(-60, 60, 10) * u.deg
    lons = np.linspace(-180, 180, 10) * u.deg 
    ground_grid = [EarthLocation(lon=lon, lat=lat) for lat in lats for lon in lons]

    # Calculate the mean revisit time
    mrt, achieved_qualities, num_accessed_gp = calculate_mean_revisit_time(custom_constellation,
                                                                          ground_grid,
                                                                          duration_days=2,
                                                                          time_step_minutes=15, # Adjust for accuracy/speed
                                                                          min_achieved_image_quality=0.001) # Adjust as needed

    print(f"\n--- Results ---")
    print(f"Calculated Mean Revisit Time: {mrt:.2f}")
