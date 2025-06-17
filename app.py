import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import io
import contextlib # Keep this for suppress_stdout_stderr if used elsewhere, or for run_simulation
from mbse_utils import generate_requirements_csv, generate_xmi_model, generate_bdd_dot
import textwrap # For dedenting markdown strings
import os

# Attempt to import PIL and store the Image module if successful
_PIL_Image_module = None
try:
    from PIL import Image as _PIL_Image_module_temp
    _PIL_Image_module = _PIL_Image_module_temp
except ImportError:
    # This error is primarily for the user; the app will try to run without texture.
    st.error("Pillow library not found. Please install it: pip install Pillow")

# Set a default theme for the plots
pio.templates.default = "plotly_dark"

# Initialize SATELLITE_OPTIONS to None
SATELLITE_OPTIONS = None
# Attempt to import simulation modules
try:
    from constellation_sim import build_custom_constellation, calculate_mean_revisit_time, SATELLITE_OPTIONS as sim_SATELLITE_OPTIONS
    from poliastro.bodies import Earth as PoliastroEarth # For Earth radius and transformations
    from astropy import units as u
    from astropy.coordinates import EarthLocation, SkyCoord, GCRS, ITRS
    from astropy.time import Time # Still needed for propagation
    simulation_modules_available = True
    SATELLITE_OPTIONS = sim_SATELLITE_OPTIONS # Assign if import is successful
except ImportError as e:
    # This error will be shown in the Streamlit interface if imports fail
    simulation_modules_available = False
    simulation_import_error = e

st.set_page_config(layout="wide", page_title="Constellation Design Explorer")

# CSS for styling the button as a link is removed.

# Define satellite type colors globally for consistency
SAT_TYPE_COLORS_RGBA = {
    "low_quality": [120, 120, 120, 255],  # Grey
    "medium_quality": [0, 150, 255, 255], # Blue
    "high_quality": [255, 100, 0, 255],   # Orange
    "default": [200, 200, 200, 255]       # Default color
}

# Define marker symbols for satellite types (NEW)
SAT_TYPE_SYMBOLS = {
    "low_quality": "circle",
    "medium_quality": "square",
    "high_quality": "diamond",
    "default": "circle"
}

@st.cache_data
def load_data(file_path):
    """
    Loads data from the specified CSV file path.
    Using st.cache_data to prevent reloading on every interaction.
    """
    try:
        df = pd.read_csv(file_path)
        # Ensure quality is positive (as it's maximized)
        if 'quality' in df.columns and df['quality'].min() < 0:
            df['quality'] *= -1
        return df
    except FileNotFoundError:
        st.error(f"Error: The file `{file_path}` was not found. Please make sure it is in the same directory as your Streamlit script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the CSV file: {e}")
        return None

@st.cache_data
def load_performance_metrics(file_path):
    """
    Loads performance metrics from the specified CSV file path.
    Using st.cache_data to prevent reloading on every interaction.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The performance metrics file `{file_path}` was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the performance metrics CSV file: {e}")
        return None
def find_best_compromise(df_subset):
    """
    Finds the best compromise solution from a DataFrame subset using the
    normalized distance method. This is adapted from the baseline_optimizer.
    """
    if df_subset.empty:
        return None

    objectives = df_subset[['cost', 'revisit_time', 'quality']].values
    normalized_obj = np.zeros_like(objectives, dtype=float)

    # Normalize Cost (minimize)
    min_cost, max_cost = objectives[:, 0].min(), objectives[:, 0].max()
    range_cost = max_cost - min_cost if (max_cost - min_cost) > 1e-9 else 1.0
    normalized_obj[:, 0] = (objectives[:, 0] - min_cost) / range_cost

    # Normalize Revisit Time (minimize)
    min_revisit, max_revisit = objectives[:, 1].min(), objectives[:, 1].max()
    range_revisit = max_revisit - min_revisit if (max_revisit - min_revisit) > 1e-9 else 1.0
    normalized_obj[:, 1] = (objectives[:, 1] - min_revisit) / range_revisit

    # Normalize Quality (maximize)
    min_quality, max_quality = objectives[:, 2].min(), objectives[:, 2].max()
    range_quality = max_quality - min_quality if (max_quality - min_quality) > 1e-9 else 1.0
    normalized_obj[:, 2] = (max_quality - objectives[:, 2]) / range_quality
    
    # Calculate Euclidean distance to the ideal point (0,0,0)
    distances = np.linalg.norm(normalized_obj, axis=1)
    best_idx_in_subset = np.argmin(distances)
    
    return df_subset.iloc[best_idx_in_subset]


def create_interactive_plot(valid_points, invalid_points, best_solution):
    """
    Generates an interactive 3D plot of the valid, invalid, and best points.
    """
    fig = go.Figure()

    # 1. Add trace for INVALID points (greyed out)
    if not invalid_points.empty:
        fig.add_trace(go.Scatter3d(
            x=invalid_points['cost'] / 1_000_000,  # Convert to millions
            y=invalid_points['revisit_time'],
            z=invalid_points['quality'], # Quality is already positive
            mode='markers',
            marker=dict(size=4, color='grey', opacity=0.3),
            name='Invalid Solutions',  # Reverted name
            legendgroup="group1", # Assign to a legend group
            hoverinfo='skip'
        ))
    # 2. Add trace for VALID points
    if not valid_points.empty:
        hover_texts = []
        for _, row in valid_points.iterrows():
            f_phasing_text = f"F-Phasing: {int(row['f_phasing'])}<br>" if 'f_phasing' in row else ""
            hover_texts.append(f"<b>Configuration: {int(row['num_planes'])} Planes</b><br>"
                               f"--------------------<br>"
                               f"Cost: ${row['cost']/1_000_000:,.1f}M<br>"
                               f"Revisit Time: {row['revisit_time']:.2f} hr<br>"
                               f"Quality: {row['quality']:.4f}<br>"
                               f"{f_phasing_text}")

        fig.add_trace(go.Scatter3d(
            x=valid_points['cost'] / 1_000_000,  # Convert to millions
            y=valid_points['revisit_time'],
            z=valid_points['quality'],
            mode='markers',
            marker=dict(
                size=5,
                color=valid_points['num_planes'],
                colorscale='Viridis',
                colorbar=dict(title="Num. Planes"),
                opacity=0.9,
            ),
            name='Valid Solutions', # Reverted name
            legendgroup="group2", # Assign to a different legend group
            hovertext=hover_texts,
            hoverinfo='text'
        ))

    # 3. Highlight the single BEST compromise solution
    if best_solution is not None:
        # Prepare hover text for the best solution
        best_solution_f_phasing_text = ""
        if 'f_phasing' in best_solution and pd.notna(best_solution['f_phasing']):
            try:
                best_solution_f_phasing_text = f"F-Phasing: {int(float(best_solution['f_phasing']))}<br>"
            except ValueError:
                best_solution_f_phasing_text = f"F-Phasing: {best_solution['f_phasing']}<br>" # Display as is if not convertible

        best_solution_hover_text = (
            f"<b>Optimal Compromise: {int(best_solution['num_planes'])} Planes</b><br>"
            f"--------------------<br>"
            f"Cost: ${best_solution['cost']/1_000_000:,.1f}M<br>"
            f"Revisit Time: {best_solution['revisit_time']:.2f} hr<br>"
            f"Quality: {best_solution['quality']:.4f}<br>"
            f"{best_solution_f_phasing_text}"
        )
        fig.add_trace(go.Scatter3d(
            x=[best_solution['cost'] / 1_000_000], # Convert to millions
            y=[best_solution['revisit_time']],
            z=[best_solution['quality']],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='white')),
            name='Optimal Compromise', # This name will appear if legend is shown
            hovertext=[best_solution_hover_text], # Use list for hovertext
            hoverinfo='text' # Enable hover info
        ))
    
    fig.update_layout(
        #title_text='<b>Interactive Pareto Front Explorer</b><br><sup>Filter solutions to find your optimal design</sup>',
        scene=dict(
            xaxis_title='Total System Cost (Millions USD)', 
            yaxis_title='Mean Revisit Time (hr)', 
            zaxis_title='Mean Achieved Quality',
            yaxis=dict(autorange='reversed') # This line flips the Y-axis
        ),
        showlegend=False,  # This line hides the legend
        margin=dict(l=0, r=0, b=0, t=50),
        height=700,
    )
    # The legend dictionary has been removed as showlegend=False handles hiding it.
    
    return fig

def get_int_param_from_series(series, key, default_val=0):
    """Safely retrieves and converts a parameter to int from a pandas Series."""
    val = series.get(key)
    if pd.isna(val) or val is None:
        return default_val
    try:
        return int(float(val)) # Convert to float first to handle "10.0" then to int
    except ValueError:
        # This case might occur if data is unexpectedly non-numeric
        st.warning(f"Could not convert value '{val}' for key '{key}' to int. Using default {default_val}.")
        return default_val
        
# Add these imports at the top of your app.py file
import matplotlib.cbook as cbook
from PIL import Image

# Add these imports at the top of your app.py file if they aren't there
import matplotlib.cbook as cbook
from PIL import Image

# Make sure numpy is imported at the top of your file:
# import numpy as np

# Make sure numpy as np and from PIL import Image are at the top of your file

def create_plotly_globe_visualization(propagated_data):
    """
    Creates a 3D Plotly figure for orbits and satellite positions.
    - Renders one orbit path per plane.
    - Renders markers for each individual satellite.
    - Orbit paths are black.
    - Satellite markers are colored/shaped by type.
    """
    fig_3d_orbit = go.Figure()

    # --- Load and Prepare Earth Texture ---
    texture_data = None
    try:
        if _PIL_Image_module: # Check if Pillow's Image module was imported
            with _PIL_Image_module.open("earth_map.jpg") as img:
                max_texture_size = (1024, 768)
                img.thumbnail(max_texture_size, _PIL_Image_module.Resampling.LANCZOS)
                texture_data = np.asarray(img.convert('L'))
        else:
            st.warning("Pillow library not available. Earth texture cannot be loaded. Using fallback color.")
            texture_data = None
    except FileNotFoundError:
        st.warning("Earth texture file 'earth_map.jpg' not found. Using a fallback color.")
        texture_data = None
    except Exception as e:
        st.warning(f"Could not load or process Earth texture: {e}. Using fallback color.")
        texture_data = None

    def rgba_to_plotly_str(rgba_list):
        return f"rgba({rgba_list[0]},{rgba_list[1]},{rgba_list[2]},{rgba_list[3]/255.0})"

    R_earth_km = PoliastroEarth.R.to_value(u.km)
    
    # --- Earth Sphere (Textured or Fallback) ---
    if texture_data is not None:
        n_lat, n_lon = texture_data.shape
        phi, theta = np.linspace(0, np.pi, n_lat), np.linspace(0, 2 * np.pi, n_lon)
        x_sphere = R_earth_km * np.outer(np.sin(phi), np.cos(theta))
        y_sphere = R_earth_km * np.outer(np.sin(phi), np.sin(theta))
        z_sphere = R_earth_km * np.outer(np.cos(phi), np.ones(n_lon))
        earth_trace = go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            surfacecolor=np.roll(texture_data, shift=-(texture_data.shape[1] // 2), axis=1),
            colorscale=[[0, 'rgb(30,100,200)'], [0.4, 'rgb(60,150,60)'], [1.0, 'rgb(240,220,180)']],
            showscale=False, name="Earth", hoverinfo='skip'
        )
    else: # Fallback sphere
        n_sphere_points = 100
        phi, theta = np.linspace(0, np.pi, n_sphere_points), np.linspace(0, 2 * np.pi, n_sphere_points)
        x_sphere = R_earth_km * np.outer(np.sin(phi), np.cos(theta))
        y_sphere = R_earth_km * np.outer(np.sin(phi), np.sin(theta))
        z_sphere = R_earth_km * np.outer(np.cos(phi), np.ones(n_sphere_points))
        earth_trace = go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, 'rgb(30,100,200)'], [1, 'rgb(60,150,60)']],
            showscale=False, opacity=0.7, name="Earth", hoverinfo='skip'
        )
    fig_3d_orbit.add_trace(earth_trace)

    # --- Add dummy traces for a clean legend ---
    sorted_sat_types = sorted([key for key in SAT_TYPE_SYMBOLS.keys() if key != "default"])
    for sat_type_key in sorted_sat_types:
        symbol = SAT_TYPE_SYMBOLS[sat_type_key]
        color_rgba = SAT_TYPE_COLORS_RGBA.get(sat_type_key, SAT_TYPE_COLORS_RGBA["default"])
        plotly_color_str_legend = rgba_to_plotly_str(color_rgba)
        type_display_name = sat_type_key.replace('_', ' ').title()
        fig_3d_orbit.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers',
            marker=dict(size=8, color=plotly_color_str_legend, symbol=symbol),
            name=type_display_name, legendgroup=sat_type_key
        ))

    # --- Satellite Orbits and Initial Position Markers ---
    max_coord_val_overall = R_earth_km
    for sat_data in propagated_data:
        orbit_path_coords_list = sat_data.get("orbit_xyz_gcrs_km", [])
        coords = np.asarray(orbit_path_coords_list) if orbit_path_coords_list else np.array([])
        initial_pos_val = sat_data.get("initial_position_gcrs_km")
        initial_pos_km = np.asarray(initial_pos_val) if initial_pos_val is not None else np.array([])

        has_orbit_track = coords.ndim == 2 and coords.shape[0] > 0 and coords.shape[1] == 3
        has_initial_pos = initial_pos_km.ndim == 1 and initial_pos_km.shape[0] == 3
        
        if not has_orbit_track and not has_initial_pos:
            continue

        if has_orbit_track: max_coord_val_overall = max(max_coord_val_overall, np.max(np.abs(coords)))
        if has_initial_pos: max_coord_val_overall = max(max_coord_val_overall, np.max(np.abs(initial_pos_km)))

        sat_type = sat_data.get("type", "default")
        marker_color_rgba = SAT_TYPE_COLORS_RGBA.get(sat_type, SAT_TYPE_COLORS_RGBA["default"])
        marker_plotly_color_str = rgba_to_plotly_str(marker_color_rgba)
        sat_symbol = SAT_TYPE_SYMBOLS.get(sat_type, SAT_TYPE_SYMBOLS["default"])
        sat_label = sat_data.get("label", "Satellite")
        orbit_line_color_str = 'rgba(0, 0, 0, 1.0)'  # Set all orbital paths to be black

        # Orbit line trace (only plotted if data exists)
        if has_orbit_track:
            fig_3d_orbit.add_trace(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode='lines',
                line=dict(width=1, color=orbit_line_color_str), # Use fixed black color and width
                name="Orbit Path",
                legendgroup="orbit_paths",
                hoverinfo='skip',
                showlegend=False
            ))

        # Initial position marker trace
        if has_initial_pos:
            fig_3d_orbit.add_trace(go.Scatter3d(
                x=[initial_pos_km[0]], y=[initial_pos_km[1]], z=[initial_pos_km[2]],
                mode='markers',
                marker=dict(size=8, color=marker_plotly_color_str, symbol=sat_symbol),
                name=f"{sat_label} Initial",
                legendgroup=sat_type,
                showlegend=False,
                hovertext=f"<b>{sat_label} (Initial Position)</b><br>"
                          f"Type: {sat_type.replace('_', ' ').title()}<br>"
                          f"Position (km):<br>X: {initial_pos_km[0]:.1f}<br>Y: {initial_pos_km[1]:.1f}<br>Z: {initial_pos_km[2]:.1f}",
                hoverinfo='text'
            ))
    
    plot_range = max_coord_val_overall * 1.15
    fig_3d_orbit.update_layout(
        showlegend=True,
        legend_title_text='Satellite Types',
        legend=dict(traceorder='normal', itemsizing='constant', orientation='v',
                    yanchor='top', y=1, xanchor='right', x=1.05),
        scene=dict(
            xaxis=dict(title='X (km)', range=[-plot_range, plot_range], showbackground=False),
            yaxis=dict(title='Y (km)', range=[-plot_range, plot_range], showbackground=False),
            zaxis=dict(title='Z (km)', range=[-plot_range, plot_range], showbackground=False),
            aspectmode='cube',
            camera_eye=dict(x=-0.197, y=-1.405, z=1.108)
        ), 
        margin={"r":10,"t":60,"l":10,"b":10}, height=700
    )
    
    return fig_3d_orbit

def run_simulation_for_optimal_design(optimal_design_series):
    """
    Runs a detailed simulation for the given optimal design parameters.
    Prepares visualization data with one orbit track per plane.
    """
    if not simulation_modules_available:
        st.error(f"Simulation modules are not available. Cannot run detailed simulation. Error: {simulation_import_error}")
        return None, None, None, None, {}, None

    sim_params = {
        "duration_days": 2,
        "time_step_minutes": 15,
        "min_achieved_image_quality_threshold": 0.001
    }

    try:
        num_planes = int(optimal_design_series['num_planes'])
        f_phasing = int(float(optimal_design_series.get('f_phasing', 0)))

        plane_definitions = []
        for i in range(1, num_planes + 1):
            alt_val, inc_val = optimal_design_series.get(f"plane_{i}_alt"), optimal_design_series.get(f"plane_{i}_inc")
            hq, mq, lq = get_int_param_from_series(optimal_design_series, f"plane_{i}_high_q"), get_int_param_from_series(optimal_design_series, f"plane_{i}_med_q"), get_int_param_from_series(optimal_design_series, f"plane_{i}_low_q")

            if pd.isna(alt_val) or pd.isna(inc_val):
                st.warning(f"Missing altitude/inclination for plane {i}. Skipping this plane.")
                continue
            
            sats_in_this_plane = hq + mq + lq
            if sats_in_this_plane <= 0:
                st.info(f"Plane {i} has no satellites; it will be omitted.")
                continue

            plane_definitions.append({
                "altitude": float(alt_val) * u.km, "inclination": float(inc_val) * u.deg,
                "sats_in_plane": sats_in_this_plane,
                "satellite_counts_for_plane": {"high_quality": hq, "medium_quality": mq, "low_quality": lq}
            })
        
        if not plane_definitions:
            st.error("No valid plane definitions for simulation.")
            return None, None, None, None, sim_params, None

        constellation = build_custom_constellation(plane_definitions=plane_definitions, global_f_phasing=f_phasing)
        lats_g, lons_g = np.linspace(-60, 60, 5) * u.deg, np.linspace(-180, 180, 10, endpoint=False) * u.deg
        ground_grid = [EarthLocation(lon=lon, lat=lat) for lat in lats_g for lon in lons_g]

        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stdout_capture):
            mrt_val, qualities, num_accessed = calculate_mean_revisit_time(
                constellation, ground_grid,
                duration_days=sim_params["duration_days"], time_step_minutes=sim_params["time_step_minutes"],
                min_achieved_image_quality=sim_params["min_achieved_image_quality_threshold"]
            )

        sim_mrt = mrt_val.to_value(u.hr) if hasattr(mrt_val, 'unit') and not np.isnan(mrt_val.value) else np.nan
        sim_quality = np.mean(qualities) if qualities else 0.0

        # --- Plotly Globe Visualization Data Preparation ---
        propagated_data_for_plotly = []
        if constellation:
            try:
                prop_start_time = Time("2025-01-01 00:00:00", scale="utc")
                prop_duration = sim_params["duration_days"] * u.day
                
                # Visualization-specific propagation parameters
                viz_time_step_minutes = 2  # As requested: 2-minute time step for visualization
                viz_prop_time_step_seconds = viz_time_step_minutes * 60
                num_prop_points_viz = int(prop_duration.to_value(u.s) / viz_prop_time_step_seconds) + 1
                prop_epochs_viz = prop_start_time + np.linspace(0, prop_duration.to_value(u.s), num_prop_points_viz) * u.s

                # The orbit_track_plotted_for_viz flag is no longer needed
                # as we want to plot all orbit tracks.

                for i, (orbit, props) in enumerate(constellation):
                    # Propagate to get positions for visualization using viz_prop_epochs
                    r_gcrs_list_km_viz = [orbit.propagate(t - orbit.epoch).r.to_value(u.km) for t in prop_epochs_viz]
                    positions_gcrs_xyz_km_viz = np.array(r_gcrs_list_km_viz)
                    
                    # Ground track calculation (using visualization propagation for consistency)
                    skycoords_gcrs_viz = SkyCoord(
                        x=positions_gcrs_xyz_km_viz[:, 0] * u.km,
                        y=positions_gcrs_xyz_km_viz[:, 1] * u.km,
                        z=positions_gcrs_xyz_km_viz[:, 2] * u.km,
                        frame=GCRS(obstime=prop_epochs_viz),
                        representation_type='cartesian'
                    )
                    skycoords_itrs_viz = skycoords_gcrs_viz.transform_to(ITRS(obstime=prop_epochs_viz))
                    lons_deg_viz = skycoords_itrs_viz.represent_as('spherical').lon.wrap_at(180 * u.deg).deg
                    lats_deg_viz = skycoords_itrs_viz.represent_as('spherical').lat.deg
                    
                    sat_type_name = props.get('name', 'Sat')
                    sat_label = f"{sat_type_name.replace('_', ' ').title()} {i+1}"
                    
                    sat_viz_data = {
                        "label": sat_label,
                        "type": sat_type_name,
                        "ground_track_lonlat_deg": np.vstack((lons_deg_viz, lats_deg_viz)).T.tolist(),
                        "initial_position_gcrs_km": orbit.r.to_value(u.km).tolist()
                    }
                    
                    # Populate orbit track for all satellites for visualization
                    sat_viz_data["orbit_xyz_gcrs_km"] = positions_gcrs_xyz_km_viz.tolist()
                        
                    propagated_data_for_plotly.append(sat_viz_data)

            except Exception as plotly_data_e:
                st.warning(f"Could not generate data for Plotly visualization: {plotly_data_e}")
                propagated_data_for_plotly = []

        return (sim_mrt, sim_quality, num_accessed, len(ground_grid), sim_params, propagated_data_for_plotly)

    except Exception as e:
        st.error(f"An error occurred during simulation: {e}")
        return None, None, None, None, sim_params, None



# --- App Layout ---
st.title("🛰️ Constellation Design Trade-Space Explorer")

# The "Learn More" button previously styled as a link is removed from here.

# Introductory text moved here
st.markdown(
    """This application allows you to discover optimal imaging satellite constellation designs by leveraging pre-trained Machine Learning models and multi-objective optimization results along 3D pareto fronts.
    You can define your mission's Key Performance Parameters (KPPs), visualize trade-offs, simulate the performance of selected designs, and generate Model-Based Systems Engineering (MBSE) artifacts like requirements documents and system models."""
)


@st.dialog("About the Constellation Design Trade-Space Explorer", width="large")
def show_learn_more_dialog():
    # Define the markdown content as a separate variable
    # Ensure the content starts on a new line immediately after """ for textwrap.dedent
    st.markdown(r"""
        This tool is designed to help you navigate the complex process of designing an optimal Earth observation satellite constellation for imaging missions.

        ---
        **Motivation: Why is This Important?**

        Designing an imaging satellite constellation involves balancing numerous competing factors:
        * **Coverage & Revisit Time:** How often can your satellites see a point on Earth? This is crucial for timely data.
        * **Image Quality:** What level of detail and accuracy can your sensors provide? Higher quality often means more expensive and heavier satellites.
        * **System Cost:** The cost model primarily focuses on satellite hardware and launch expenses. For the scope of this application, detailed simulation and costing of the ground segment (e.g., control stations, data processing centers) and the user segment (e.g., end-user terminals, data dissemination networks) are considered simplified assumptions. However, these segments are conceptually represented in the generated MBSE architectural diagrams to provide a more complete system overview. Budgets remain a primary driver in the optimization.
        * **Complexity:** More planes and satellites can improve performance but also increase operational complexity and cost.
        Finding the "sweet spot" that meets mission objectives without exceeding budget or complexity is a significant engineering challenge. This tool leverages Machine Learning and simulation to explore this vast design space efficiently.

        ---
        **Understanding the Design Parameters:**

        The ML models and optimizers explore designs based on these key parameters:

        * **Altitude (300 km to 800 km):** This is the height of the satellite's orbit above Earth's surface. Lower altitudes can offer better image resolution but may have shorter satellite lifespans and smaller individual footprints. Higher altitudes cover more ground but may require more powerful sensors.
        * **Inclination (25° to 98°):** Inclination is the angle of the orbital plane relative to Earth's equator.
            * Low inclinations (e.g., 25°) focus coverage on equatorial regions.
            * High inclinations (e.g., 98°, a sun-synchronous orbit) provide coverage over polar regions and can offer consistent lighting conditions for imaging.
        * **Satellite Types (High, Medium, Low Quality):** Each plane can be populated with a mix of simplified satellite bus characteristics, differing in:
            * **Cost:** High-quality satellites are significantly more expensive.
            * **Sensor Quality:** A proxy for imaging performance (e.g., resolution, clarity).
            * **Mass:** Heavier satellites incur higher launch costs.
            * *(These characteristics are defined in `constellation_sim.py`'s `SATELLITE_OPTIONS`)*
        * **Global F-Phasing (0 to 3):** This parameter (from Walker Delta constellation patterns) controls the relative spacing (phase shift) of satellites between different orbital planes, helping to distribute coverage more evenly. It's used in calculating the true anomaly ($\nu_{ij}$) for each satellite:
        
            $$\nu_{ij} = M_j + \frac{p_i \cdot F \cdot 360^\circ}{T}$$

            Where: $\nu_{ij}$ is the true anomaly of the $j^{\text{th}}$ satellite in the $p_i^{\text{th}}$ plane, $M_j$ is the mean anomaly of the $j^{\text{th}}$ satellite within its plane, $p_i$ is the plane index (0-indexed), $F$ is the global F-phasing value, and $T$ is the total number of satellites in the constellation.

        * **Orbital Elements Context:** The parameters you can adjust (altitude, inclination, and F-phasing which determines true anomaly) define three of the six classical Keplerian orbital elements needed to fully describe an orbit (the others being eccentricity, longitude of the ascending node, and argument of periapsis). In the current simulation, eccentricity is assumed to be zero (circular orbits), and the longitude of the ascending node and argument of periapsis are set to default or distributed values. Future work could explore varying these remaining three elements to further optimize constellation designs.
        ---
        **How it Works (Application Flow):**

        1.  **Explore Designs:**
            * **Select Model:** Choose an ML model (XGBoost, Neural Network, or Quantum-Hybrid) from the sidebar. These models were pre-trained to predict constellation performance.
            * **View Pareto Front:** The 3D plot displays numerous potential designs, showing trade-offs between **Cost**, **Revisit Time**, and **Image Quality**.
            * **Define KPPs:** Use the **sidebar sliders** to set your mission's Key Performance Parameters (e.g., max cost, max revisit time, min quality). This filters the designs to show only those meeting your criteria.
        2.  **Discover Optimal Design:**
            * The tool automatically highlights the **Optimal Compromise** design within your KPPs on the 3D plot.
            * Detailed parameters for this optimal design (number of planes, altitude, inclination, satellite types per plane, F-phasing) are displayed in the right-hand panel.
        3.  **Simulate & Visualize:**
            * Click **"Run Simulation for This Design"** to perform a detailed physics-based simulation of the selected optimal design using Poliastro.
            * View the simulated performance metrics (revisit time, quality, coverage).
            * See the constellation's orbits and ground tracks visualized on an interactive 3D globe.
        4.  **Generate MBSE Artifacts:**
            * **Requirements:** Export a **Requirements Document (CSV)** compatible with tools like DOORS or JAMA.
            * **System Model:** Generate a **System Architecture Model (XMI)** for import into SysML modeling tools (e.g., Cameo, Papyrus).
            * **Diagram:** View an auto-generated **SysML Block Definition Diagram (BDD)** representing the system architecture.

        ---
        **Behind the Scenes (Code Workflow):**

        * **1. Data Generation (`generate_training_dataset.py`):**
            * Thousands of random constellation designs were simulated using `constellation_sim.py` (which wraps Poliastro for orbital mechanics).
            * Performance metrics (revisit time, quality) and system costs are calculated for each design. This extensive dataset is saved to `constellation_training_data_cleaned.csv`.
            * During this data generation, a key criterion for a design to be considered valid is its ability to access at least **80% of a predefined global grid of ground points**. This grid spans latitudes from **-60° to +60°** and longitudes from **-180° to +180°**, ensuring that the generated designs have a reasonable global coverage capability before being used for ML model training.
        * **2. ML Model Training (e.g., `neural_net_model.py`, `quantum_model.py`):**
            * The `constellation_training_data_cleaned.csv` dataset serves as the foundation for training several distinct machine learning models. The objective is to create models that can rapidly predict key performance indicators (`mean_revisit_time_hr` and `mean_achieved_quality`) based on the input constellation design parameters.
            * **Model Diversity:**
                * **XGBoost (`baseline_ml_model.py`):** A gradient boosting framework known for its efficiency, accuracy, and robustness, often performing well on structured/tabular data. It's used here as a strong classical baseline.
                * **Keras Neural Network (`neural_net_model.py`):** Deep learning models built with TensorFlow and Keras are employed to capture potentially complex, non-linear relationships within the design space. `KerasTuner` is utilized for automated hyperparameter optimization (e.g., number of layers, neurons per layer, activation functions, learning rate) to find the best performing neural network architecture.
                * **Quantum-Hybrid Model (`quantum_model.py`):** This explores a more novel approach by combining classical neural network layers with a quantum circuit implemented using PennyLane and TensorFlow Quantum. The classical layers pre-process the input data, which is then encoded into a quantum state. The quantum circuit, with tunable parameters (like the number of qubits and quantum layers, also optimized using `KerasTuner`), performs transformations, and its measurements are post-processed by further classical layers to produce the final predictions. This model investigates the potential of quantum machine learning techniques for this complex optimization problem.
            * **Training Process:** Each model type is trained to predict the two target variables. Feature scaling (using `StandardScaler`) is applied to the input features, and target variables are also scaled before training the neural network and quantum-hybrid models to aid in convergence and performance.
            * **Saved Artifacts:** The trained models, along with their specific feature and target scalers, and optimized hyperparameters (for Keras-based models) are saved to disk (e.g., `.keras` files for models, `.joblib` for scalers and hyperparameters). This allows the optimizer and this application to load and use them without retraining.
        * **3. Multi-Objective Optimization (e.g., `nn_optimizer.py`, `quantum_optimizer.py`):**
            * A pre-trained ML model is loaded. The NSGA-II genetic algorithm explores the design space (number of planes, altitudes, inclinations, satellite mix, F-phasing) to find **Pareto-optimal fronts**.
            * A Pareto front represents a set of solutions where no single objective (e.g., cost, revisit time, quality) can be improved without worsening at least one other objective. In essence, these are the "best possible" trade-offs.
            * The optimizer balances cost (to be minimized), revisit time (to be minimized), and quality (to be maximized) to identify these fronts. The "Optimal Compromise" point highlighted in the application is a single solution chosen from these fronts that offers a balanced performance across all objectives, especially after filtering by your KPPs.
            * These optimization results, containing many such Pareto-optimal solutions, are saved to CSV files (e.g., `nn_optimizer_pareto_fronts.csv`).
    """) # End of the st.markdown call within the dialog function

# --- Global Constants for Model Selection and Metrics ---
# Define the path to the performance metrics file
PERFORMANCE_METRICS_FILE = 'model_performance_metrics.csv'

MODEL_OPTIONS = {
    "XGBoost Trained Optimizer": "optimizer_pareto_fronts.csv",
    "Keras Neural Net Trained Optimizer": "nn_optimizer_pareto_fronts.csv",
    "Hybrid-Quantum ML Optimizer": "qml_optimizer_pareto_fronts.csv"
}
# Mapping from the display model names to the names used in model_performance_metrics.csv
MODEL_NAME_TO_METRICS_NAME = {
    "XGBoost Trained Optimizer": "XGBoost",
    "Keras Neural Net Trained Optimizer": "NeuralNetwork",
    "Hybrid-Quantum ML Optimizer": "QuantumHybrid" # Assuming this is the name in your CSV for QML
}
model_names_list = list(MODEL_OPTIONS.keys())

# Initialize selected model in session state if not already set
# This needs to be done BEFORE data_file_path is accessed.
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = model_names_list[0] # Default to the first model


# --- Load Data ---
data_file_path = MODEL_OPTIONS[st.session_state.selected_model_name]
all_results_df = None
if data_file_path:
    all_results_df = load_data(data_file_path)
else:
    # This case might be less likely now if selected_model_name always has a default
    st.warning(f"The '{st.session_state.selected_model_name}' is not yet implemented or has no associated data file. Please select another model source.")

if not simulation_modules_available and 'all_results_df' in locals() and all_results_df is not None:
    st.sidebar.warning(f"Simulation modules could not be loaded (Error: {simulation_import_error}). Detailed re-simulation and 3D globe features will be disabled.")


# --- Main App ---
if all_results_df is not None:
    # Add the "Learn More" button at the top of the sidebar
    if st.sidebar.button("Learn More About This Tool", key="learn_more_sidebar_button_top"):
        # Directly call the dialog function when the button is clicked
        show_learn_more_dialog()

    st.sidebar.header("Define Constraints")
    st.sidebar.markdown("---") # Separator

    # Dynamically set slider ranges based on the data
    cost_min_actual, cost_max_actual = float(all_results_df['cost'].min()), float(all_results_df['cost'].max())
    revisit_min, revisit_max = float(all_results_df['revisit_time'].min()), float(all_results_df['revisit_time'].max())
    quality_min, quality_max = float(all_results_df['quality'].min()), float(all_results_df['quality'].max())
    
    # Convert cost to millions for slider
    cost_min_millions = cost_min_actual / 1_000_000
    cost_max_millions = cost_max_actual / 1_000_000
    slider_step_cost = max(0.1, round((cost_max_millions - cost_min_millions) / 100, 1)) # Step for millions
    if slider_step_cost == 0.0: slider_step_cost = 0.1 # Ensure step is not zero if min/max are too close

    # Sliders for user constraints
    max_cost_millions = st.sidebar.slider(
        "Maximum Acceptable Cost (Millions USD)",
        min_value=cost_min_millions, max_value=cost_max_millions, value=cost_max_millions,
        step=slider_step_cost, format="$%.1fM"
    )
    max_revisit = st.sidebar.slider(
        "Maximum Revisit Time (hr)",
        min_value=revisit_min, max_value=revisit_max, value=revisit_max,
        step=(revisit_max - revisit_min) / 100 if (revisit_max - revisit_min) > 0 else 0.1, format="%.2f hr"
    )
    min_quality = st.sidebar.slider(
        "Minimum Image Quality",
        min_value=quality_min, max_value=quality_max, value=quality_min,
        step=(quality_max - quality_min) / 100 if (quality_max - quality_min) > 0 else 0.001, format="%.3f"
    )
    st.sidebar.markdown("---") # Separator

    # --- Model Selection (Moved to Sidebar) ---
    st.sidebar.header("Select Model Source")
    # PERFORMANCE_METRICS_FILE, MODEL_OPTIONS, MODEL_NAME_TO_METRICS_NAME, model_names_list
    # are now defined globally. st.session_state.selected_model_name is also initialized globally.

    selected_model_name_sidebar = st.sidebar.radio(
        "Choose an optimization model:",
        model_names_list,
        index=model_names_list.index(st.session_state.selected_model_name) # Set current selection
    )

    if st.session_state.selected_model_name != selected_model_name_sidebar:
        st.session_state.selected_model_name = selected_model_name_sidebar
        # Clear dependent session state data when model changes
        keys_to_clear_on_model_change = [
            'simulation_results', 'plotly_orbit_data', 'simulation_run_for_design',
            'bdd_dot_string', 'bdd_for_design_id', 'requirements_csv', 'xmi_file_path',
        ]
        for key in keys_to_clear_on_model_change:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun() # Rerun *only if* the model changed

    st.sidebar.markdown("---") # Separator

    # Filter data based on slider inputs
    valid_points = all_results_df[
        (all_results_df['cost'] <= max_cost_millions * 1_000_000) & # Compare with actual cost
        (all_results_df['revisit_time'] <= max_revisit) &
        (all_results_df['quality'] >= min_quality)
    ].copy()
    invalid_points = all_results_df.drop(valid_points.index)

    # Find the best compromise within the filtered set
    best_compromise = find_best_compromise(valid_points)

    # --- Display Results ---
    # --- Load Performance Metrics and Display (Moved to Sidebar) ---
    st.sidebar.header("Model Performance")
    performance_df = load_performance_metrics(PERFORMANCE_METRICS_FILE)
    if performance_df is not None:
        metrics_model_name = MODEL_NAME_TO_METRICS_NAME.get(st.session_state.selected_model_name, st.session_state.selected_model_name)
        selected_model_metrics = performance_df[performance_df['model_name'] == metrics_model_name]
        if not selected_model_metrics.empty:
            display_metrics = selected_model_metrics[['target_variable', 'mse', 'r2']].copy()
            display_metrics.columns = ['Target Variable', 'MSE', 'R²']
            st.sidebar.dataframe(
                display_metrics,
                hide_index=True,
                column_config={
                    "MSE": st.column_config.NumberColumn(format="%.3f"),
                    "R²": st.column_config.NumberColumn(format="%.3f")
                }
            )
        else:
            st.sidebar.warning(f"No metrics found for '{st.session_state.selected_model_name}'.")
    
    st.sidebar.markdown("---") # Separator
    # Display selected model and data source in the sidebar
    st.sidebar.success(f"Active Model: **{st.session_state.selected_model_name}**")

    #st.header("Results")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Interactive Pareto Front Visualization")
        st.info("Explore optimal constellation designs along a Pareto front by balancing cost, revisit time, and image quality based on the KPPs in the sidebar.")
        fig = create_interactive_plot(valid_points, invalid_points, best_compromise)
        st.plotly_chart(fig, use_container_width=True)

        # --- Globe Visualization Section ---
        if simulation_modules_available: # Check if Poliastro etc. are loaded
            st.subheader("Interactive Initial Orbits Visualization")
            st.info("Click 'Run Simulation' to view the simulated satellite orbits and their initial positions in 3D space for the selected optimal design.")

            plotly_data_for_globe = []
            info_message_globe = None
            # Determine the identifier for the current best compromise design
            current_best_design_identifier = best_compromise.name if best_compromise is not None else None

            # Check if simulation was run for the *currently selected* best compromise design
            if ('simulation_run_for_design' in st.session_state and
                st.session_state.simulation_run_for_design == current_best_design_identifier):
                # If so, use its plotly_orbit_data (which might be empty if sim yielded no tracks)
                if 'plotly_orbit_data' in st.session_state and st.session_state.plotly_orbit_data:
                    plotly_data_for_globe = st.session_state.plotly_orbit_data
                else:
                    # Simulation ran for this design, but no visual data was generated (e.g., no satellites)
                    info_message_globe = "Plotly visualization data could not be generated for the selected design, or the design has no satellites."
            # If no simulation has been run for the current best_compromise,
            # plotly_data_for_globe remains an empty list, resulting in an empty globe.

            spinner_text_globe = "Loading 3D Globe..."
            if plotly_data_for_globe: # If there are tracks to draw
                spinner_text_globe = "Generating 3D Orbit Plotly view with tracks..."

            with st.spinner(spinner_text_globe):
                fig_3d_orbit_plotly = create_plotly_globe_visualization(plotly_data_for_globe)
            
            st.plotly_chart(fig_3d_orbit_plotly, use_container_width=True)

            if info_message_globe:
                st.info(info_message_globe)
        else:
            # If simulation modules are not available, inform the user.
            st.warning("3D Globe visualization is unavailable because simulation modules (like Poliastro) could not be loaded.")

    with col2:
        st.subheader("Optimal Design")
        if best_compromise is not None:
            st.info("Found an optimal design within your constraints.")
            
            st.metric("Number of Planes", f"{int(best_compromise['num_planes'])}")
            st.metric("Best Cost", f"${best_compromise['cost']/1_000_000:,.1f}M") # Display in millions
            st.metric("Best Mean Revisit Time", f"{best_compromise['revisit_time']:.2f} hr")
            st.metric("Best Image Quality", f"{best_compromise['quality']:.4f}")

            st.markdown("---")
            st.markdown("**Recommended Orbital Parameters:**")
            
            num_planes = int(best_compromise['num_planes'])
            for i in range(1, num_planes + 1):
                alt_val = best_compromise.get(f"plane_{i}_alt")
                inc_val = best_compromise.get(f"plane_{i}_inc")
                
                # Robustly get satellite counts for display
                hq_disp = get_int_param_from_series(best_compromise, f"plane_{i}_high_q")
                mq_disp = get_int_param_from_series(best_compromise, f"plane_{i}_med_q")
                lq_disp = get_int_param_from_series(best_compromise, f"plane_{i}_low_q")

                if pd.notna(alt_val) and alt_val is not None and pd.notna(inc_val) and inc_val is not None:
                    # Use st.markdown to create a formatted, multi-line output
                    st.markdown(
                        f"""
                        **Plane {i}:** Alt: `{float(alt_val):.1f}` km, Inc: `{float(inc_val):.1f}`°  
                        &nbsp;&nbsp;&nbsp;&nbsp;*Satellite distribution &mdash; High: `{hq_disp}`, Med: `{mq_disp}`, Low: `{lq_disp}`*
                        """,
                        unsafe_allow_html=True
                    )

            # Moved F-Phasing display to the end of orbital parameters
            if 'f_phasing' in best_compromise and pd.notna(best_compromise['f_phasing']):
                try:
                    f_phasing_val_disp = int(float(best_compromise['f_phasing']))
                    st.markdown(f"**Global F-Phasing:** `{f_phasing_val_disp}`")
                except ValueError:
                    # Display raw value if conversion fails, still using markdown
                    st.markdown(f"**Global F-Phasing (raw):** `{best_compromise['f_phasing']}`")
            else:
                # Handle case where f_phasing might be missing or NaN, using markdown
                st.markdown(f"**Global F-Phasing:** `N/A`")

            st.markdown("---")
            st.subheader("Poliastro Simulation Results") # Changed subheader

            if not simulation_modules_available:
                st.warning("Poliastro simulation is unavailable due to missing modules.")
            else:
                # Button to trigger simulation
                button_key = f"run_sim_btn_{best_compromise.name if best_compromise is not None else 'no_design'}"
                if st.button("Run Simulation for This Design", key=button_key):
                    with st.spinner("Running simulation... This may take a few minutes."):
                        sim_mrt, sim_quality, accessed_gp, total_gp, sim_p, plotly_data = run_simulation_for_optimal_design(best_compromise)
                        st.session_state.simulation_results = {"mrt": sim_mrt, "quality": sim_quality, "accessed_gp": accessed_gp, "total_gp": total_gp, "params": sim_p}
                        st.session_state.plotly_orbit_data = plotly_data
                        st.session_state.simulation_run_for_design = best_compromise.name
                    st.rerun() # Explicitly trigger a rerun to update displays based on new session state

                # Display simulation results if available for the current best_compromise
                current_design_id_for_sim_display = best_compromise.name # .name is the index
                if 'simulation_run_for_design' in st.session_state and \
                   st.session_state.simulation_run_for_design == current_design_id_for_sim_display and \
                   'simulation_results' in st.session_state:
                    
                    results = st.session_state.simulation_results
                    if results.get("accessed_gp") is not None and results.get("total_gp") is not None:
                        st.info("Simulation data loaded for the current design.") # More neutral message
                        
                        sim_mrt_disp = results.get("mrt", np.nan)
                        sim_quality_disp = results.get("quality", np.nan)
                        accessed_gp_disp = results.get("accessed_gp", 0)
                        total_gp_disp = results.get("total_gp", 0)
                        sim_params_disp = results.get("params", {})

                        coverage_percentage = (accessed_gp_disp / total_gp_disp) * 100 if total_gp_disp > 0 else 0
                        
                        col_sim1, col_sim2 = st.columns(2)
                        with col_sim1:
                            st.metric("Simulated Mean Revisit Time", f"{sim_mrt_disp:.2f} hr" if not np.isnan(sim_mrt_disp) else "N/A")
                        with col_sim2:
                            st.metric("Simulated Mean Achieved Quality", f"{sim_quality_disp:.4f}" if not np.isnan(sim_quality_disp) else "N/A")
                        
                        if total_gp_disp > 0 and coverage_percentage < 80:
                            st.warning(f"Coverage ({coverage_percentage:.1f}%) is below the 80% target for this detailed simulation.")
                        elif total_gp_disp == 0 and accessed_gp_disp == 0 : # Check if ground points were expected
                             st.info("Simulation ran, but no ground points were defined or accessed for coverage calculation.")

                        duration_days_disp = sim_params_disp.get('duration_days', 'N/A')
                        time_step_disp = sim_params_disp.get('time_step_minutes', 'N/A')
                        min_qual_thresh_disp = sim_params_disp.get('min_achieved_image_quality_threshold', 'N/A')
                        
                        st.caption(f"Sim. params: Duration {duration_days_disp}d, Step {time_step_disp}m, Min quality {float(min_qual_thresh_disp):.1f}.")
                    else:
                        st.error("Simulation was run for this design, but results are incomplete or indicate an error during simulation.")
                elif 'simulation_run_for_design' in st.session_state and best_compromise is not None: # Check best_compromise not None
                    st.info("Parameters have changed. Click 'Run Simulation for This Design' to update for the current selection.")
                else:
                    st.info("Click 'Run Simulation for This Design' to see detailed Poliastro results.")
        else:
            st.warning("No solutions found. Try relaxing the filters to find a viable solution.")
        
        # This code goes at the end of the `with col2:` block in your app.py

st.markdown("---")
st.subheader("Model-Based Systems Engineering (MBSE) Artifacts Generation")
st.info("The SysML system architecture diagram is automatically generated for the selected optimal design. You can also generate exportable requirements (CSV) and a system model (XMI).")

if best_compromise is not None:
    # --- BDD Generation (Automatic based on best_compromise) ---
    # Check if BDD needs to be regenerated for the current best_compromise
    current_design_id_for_bdd = best_compromise.name
    if ('bdd_for_design_id' not in st.session_state or
        st.session_state.bdd_for_design_id != current_design_id_for_bdd or
        'bdd_dot_string' not in st.session_state): # Or if string is missing
        
        bdd_dot_string = generate_bdd_dot(best_compromise, system_name="OptimalDesign")
        st.session_state.bdd_dot_string = bdd_dot_string
        st.session_state.bdd_for_design_id = current_design_id_for_bdd

    # Layout for MBSE: BDD on left, CSV/XMI buttons on right
    col_mbse_left, col_mbse_right = st.columns([2, 1])

    with col_mbse_left:
        st.markdown("##### System Architecture Diagram (SysML)")
        if 'bdd_dot_string' in st.session_state and st.session_state.bdd_dot_string:
            st.graphviz_chart(st.session_state.bdd_dot_string)
            st.caption("Block Definition Diagram for the optimal design. Updates automatically with filter changes.")
        elif 'bdd_dot_string' in st.session_state and not st.session_state.bdd_dot_string:
            st.warning("Could not generate BDD: No optimal design data available or an error occurred during generation.")
        else:
            # This case should ideally not be hit if the logic above is correct and best_compromise is not None
            st.info("BDD will be generated based on the optimal design.")

    with col_mbse_right:
        st.markdown("##### Export Artifacts")

        # --- Requirements CSV Generation ---
        st.markdown("###### Requirements Document (CSV)")
        if st.button("Generate Requirements (CSV)", key="csv_button"):
            constraints_dict = {
                'max_cost_millions': max_cost_millions,
                'max_revisit': max_revisit,
                'min_quality': min_quality
            }
            req_csv_data = generate_requirements_csv(best_compromise, constraints_dict)
            st.session_state.requirements_csv = req_csv_data

        if 'requirements_csv' in st.session_state and st.session_state.requirements_csv:
            st.download_button(
                 label="Download Requirements.csv",
                 data=st.session_state.requirements_csv,
                 file_name="requirements.csv",
                 mime="text/csv",
             )

        st.markdown("---") 

        # --- Architecture XMI Generation ---
        st.markdown("###### SysML Model Export (XMI)")
        if st.button("Generate Architecture Model (XMI)", key="xmi_button"):
            if not simulation_modules_available or SATELLITE_OPTIONS is None:
                st.error("Cannot generate XMI: SATELLITE_OPTIONS from constellation_sim.py are not available. Ensure simulation modules load correctly.")
                st.session_state.xmi_file_path = None
            else:
                xmi_path = generate_xmi_model(best_compromise, "system_model") # Filename "system_model.xmi"
                st.session_state.xmi_file_path = xmi_path

        if 'xmi_file_path' in st.session_state and st.session_state.xmi_file_path:
            xmi_path_to_download = st.session_state.xmi_file_path
            if os.path.exists(xmi_path_to_download):
                with open(xmi_path_to_download, "rb") as file:
                    st.download_button(
                        label="Download Model.xmi",
                        data=file,
                        file_name=os.path.basename(xmi_path_to_download), 
                        mime="application/xml",
                    )
            else:
                st.error(f"XMI file ({xmi_path_to_download}) not found. Please generate it again.")
        elif 'xmi_file_path' in st.session_state and st.session_state.xmi_file_path is None:
            # Handles case where button was clicked but XMI generation failed (e.g. SATELLITE_OPTIONS missing)
            pass # Error message already displayed by the button logic
else:
    # This message is shown if all_results_df is None, either due to QML selection or file not found for other models.
    st.info("Please select a model source with available data to explore designs and generate artifacts.")
    st.warning("No optimal design selected. Please adjust filters to select a design before generating MBSE artifacts.")

    # st.stop() # Keep the app running to show the KPP section header even if no design is selected

st.markdown("---")
st.subheader("Key Performance Parameter (KPP) Compliance")

if best_compromise is not None:
    # KPPs are based on the sidebar slider values
    kpp_cost_target_usd = max_cost_millions * 1_000_000  # Target is the MAX acceptable cost
    kpp_revisit_target_hr = max_revisit  # Target is the MAX acceptable revisit time
    kpp_quality_target = min_quality    # Target is the MIN acceptable quality

    # --- Design Cost KPP ---
    design_cost_usd = best_compromise['cost']
    cost_compliance = design_cost_usd <= kpp_cost_target_usd
    cost_emoji = "✅" if cost_compliance else "❌"
    st.markdown(f"**1. System Cost:** {cost_emoji}")
    st.markdown(f"   - Target (Max): `${kpp_cost_target_usd/1_000_000:,.1f}M`")
    st.markdown(f"   - Design Value: `${design_cost_usd/1_000_000:,.1f}M`")
    if not cost_compliance:
        st.caption(f"   ⚠️ Design cost exceeds the maximum acceptable cost by ${ (design_cost_usd - kpp_cost_target_usd)/1_000_000:,.1f}M.")

    # --- Simulation-based KPPs (Revisit Time & Quality) ---
    # The main KPP titles will be constructed dynamically below

    current_design_id_for_kpp_sim = best_compromise.name
    if 'simulation_run_for_design' in st.session_state and \
       st.session_state.simulation_run_for_design == current_design_id_for_kpp_sim and \
       'simulation_results' in st.session_state:

        sim_results = st.session_state.simulation_results
        sim_mrt = sim_results.get("mrt")
        sim_quality = sim_results.get("quality")

        # Revisit Time KPP
        if sim_mrt is not None and not np.isnan(sim_mrt):
            revisit_compliance = sim_mrt <= kpp_revisit_target_hr
            revisit_emoji = "✅" if revisit_compliance else "❌"
            st.markdown(f"**2. Mean Revisit Time (Simulated):** {revisit_emoji}")
            st.markdown(f"     - Target (Max): `{kpp_revisit_target_hr:.2f} hr`")
            st.markdown(f"     - Simulated Value: `{sim_mrt:.2f} hr`")
            if not revisit_compliance:
                st.caption(f"     ⚠️ Simulated revisit time exceeds the maximum acceptable by {sim_mrt - kpp_revisit_target_hr:.2f} hr.")
        else:
            st.markdown(f"**2. Mean Revisit Time (Simulated):** ❓")
            st.caption(f"     - Simulated revisit time not available or N/A.")

        # Image Quality KPP
        if sim_quality is not None and not np.isnan(sim_quality):
            quality_compliance = sim_quality >= kpp_quality_target
            quality_emoji = "✅" if quality_compliance else "❌"
            st.markdown(f"**3. Mean Achieved Quality (Simulated):** {quality_emoji}")
            st.markdown(f"     - Target (Min): `{kpp_quality_target:.4f}`")
            st.markdown(f"     - Simulated Value: `{sim_quality:.4f}`")
            if not quality_compliance:
                st.caption(f"     ⚠️ Simulated quality is below the minimum acceptable by {kpp_quality_target - sim_quality:.4f}.")
        else:
            st.markdown(f"**3. Mean Achieved Quality (Simulated):** ❓")
            st.caption(f"     - Simulated image quality not available or N/A.")
    else:
        st.info("Run the simulation for the selected design to see compliance for Revisit Time and Image Quality KPPs.")

else:
    st.info("Select an optimal design by adjusting filters to view KPP compliance.")

st.stop() # Ensure app stops here if it was previously stopped.