import streamlit as st
import json
import os
import sys
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd


def setup_sund_package():
    """Install and setup sund package in custom location."""
    Path("./custom_package").mkdir(parents=True, exist_ok=True)
    if "sund" not in os.listdir('./custom_package'):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", "sund<3.0"])
    
    if './custom_package' not in sys.path:
        sys.path.append('./custom_package')
    
    import sund
    return sund


def setup_model(model_name, model_file_path="./models/", param_file_path=None):
    """Setup a sund model with parameters.
    
    - model_name: name of the model (without .txt extension)
    - model_file_path: directory containing model files (default: ./models/)
    - param_file_path: path to parameter JSON file. If None, uses default based on model_name
    
    Returns: (model_instance, feature_names_list)
    """
    import sund
    
    sund.install_model(f"{model_file_path}{model_name}.txt")
    model_class = sund.import_model(model_name)
    model = model_class()
    
    # Determine parameter file path
    if param_file_path is None:
        # Use default parameter file naming convention
        if model_name == "alcohol_model":
            param_file_path = "./results/alcohol_model (186.99).json"
        elif model_name == "alcohol_model_2":
            param_file_path = "./results/alcohol_model_2 (642.74446).json"
        else:
            param_file_path = f"./results/{model_name}.json"
    
    # Load parameters
    if os.path.exists(param_file_path):
        with open(param_file_path, 'r') as f:
            param_in = json.load(f)
            params = param_in['x']
        model.parameter_values = params
    
    features = list(model.feature_names)
    return model, features


def simulate(model, anthropometrics, stim, extra_time=10):
    """Run a sund simulation with given stimuli and anthropometrics.
    
    - model: sund model instance
    - anthropometrics: dict of anthropometric values
    - stim: dict with keys like "EtOH_conc", "kcal_solid" etc, each with "t" and "f" keys
    - extra_time: hours to simulate after last event
    
    Returns: DataFrame with Time column and all simulated features
    """
    import sund
    
    act = sund.Activity(time_unit='h')
    
    for key, val in stim.items():
        act.add_output(name=key, type='piecewise_constant', t=val["t"], f=val["f"])
    
    # Only pass anthropometric parameters that the model expects and have valid values
    valid_params = {'sex', 'weight', 'height', 'age'}
    for key, val in anthropometrics.items():
        if key in valid_params and val is not None:
            try:
                # Ensure val is numeric
                numeric_val = float(val) if not isinstance(val, (int, float)) else val
                act.add_output(name=key, type='constant', f=numeric_val)
            except (ValueError, TypeError):
                # Skip invalid values
                pass
    
    sim = sund.Simulation(models=model, activities=act, time_unit='h')
    
    t_start = min(stim["EtOH_conc"]["t"] + stim["kcal_solid"]["t"]) - 0.25
    
    sim.simulate(time=np.linspace(t_start, max(stim["EtOH_conc"]["t"]) + extra_time, 10000))
    
    sim_results = pd.DataFrame(sim.feature_values, columns=sim.feature_names)
    sim_results.insert(0, 'Time', sim.time_vector)
    
    t_start_drink = min(stim["EtOH_conc"]["t"]) - 0.25
    sim_drink_results = sim_results[(sim_results['Time'] >= t_start_drink)]
    
    return sim_drink_results


def flatten(nested_list):
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested_list for item in sublist]


def get_drink_specs(drink_type):
    """Get specifications for a drink type.
    
    Returns: dict with keys: conc, volume, kcal, length
    """
    specs = {
        "Beer": {
            "conc": 5.1,           # % alcohol
            "volume": 0.33,        # liters
            "kcal": 129.827,       # per liter
            "length": 20           # minutes to drink
        },
        "Wine": {
            "conc": 12.0,
            "volume": 0.15,
            "kcal": 133.2526,
            "length": 20
        },
        "Spirit": {
            "conc": 40.0,
            "volume": 0.04,
            "kcal": 10.0,
            "length": 2
        }
    }
    
    if drink_type not in specs:
        raise ValueError(f"Unknown drink type: {drink_type}. Must be one of {list(specs.keys())}")
    
    return specs[drink_type]


def init_anthropometrics(defaults=None):
    """Initialize anthropometric session state variables.
    
    - defaults: optional dict with keys 'sex', 'weight', 'height', 'age'
    
    Returns: dict with anthropometric values
    """
    if defaults is None:
        defaults = {"sex": "Man", "weight": 70.0, "height": 1.72, "age": 30}
    
    keys = ["sex", "weight", "height", "age"]
    
    # Initialize session state
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = defaults.get(key, None)
    
    return {key: st.session_state[key] for key in keys}


def build_stimulus_dict(drink_times, drink_lengths, drink_concentrations, 
                       drink_volumes, drink_kcals, meal_times, meal_kcals):
    """Build the stimulus dictionary for simulation.
    
    Constructs piecewise constant stimuli for alcohol, volume, kcal intake, and meals.
    
    Returns: dict with keys EtOH_conc, vol_drink_per_time, kcal_liquid_per_vol, 
             drink_length, kcal_solid
    """
    EtOH_conc = [0] + [c*on for c in drink_concentrations for on in [1, 0]]
    vol_drink_per_time = [0] + [v/t*on if t > 0 else 0 for v, t in zip(drink_volumes, drink_lengths) for on in [1, 0]]
    kcal_liquid_per_vol = [0] + [k/v*on if v > 0 else 0 for v, k in zip(drink_volumes, drink_kcals) for on in [1, 0]]
    drink_length = [0] + [t*on for t in drink_lengths for on in [1, 0]]
    t = [t + (l/60)*on for t, l in zip(drink_times, drink_lengths) for on in [0, 1]]
    
    stim = {
        "EtOH_conc": {"t": t, "f": EtOH_conc},
        "vol_drink_per_time": {"t": t, "f": vol_drink_per_time},
        "kcal_liquid_per_vol": {"t": t, "f": kcal_liquid_per_vol},
        "drink_length": {"t": t, "f": drink_length},
        "kcal_solid": {"t": meal_times, "f": meal_kcals},
    }
    
    return stim


def get_anthropometrics_ui(defaults=None):
    """Render anthropometric UI inputs and return values.
    
    Displays Streamlit UI inputs for sex, weight, height, and age.
    Uses session state for persistence across reruns.
    
    - defaults: optional dict with keys 'sex', 'weight', 'height', 'age'
    
    Returns: dict with anthropometric values (sex as numeric: 1.0 for male, 0.0 for female)
    """
    if defaults is None:
        defaults = {"sex": "Man", "weight": 70.0, "height": 1.72, "age": 30}
    
    anthropometrics = init_anthropometrics(defaults=defaults)
    
    anthropometrics["sex"] = st.selectbox(
        "Sex:", 
        ["Man", "Woman"], 
        index=["Man", "Woman"].index(anthropometrics['sex']) if isinstance(anthropometrics['sex'], str) else (0 if anthropometrics['sex'] == 1.0 else 1),
        key="sex"
    )
    anthropometrics["weight"] = st.number_input(
        "Weight (kg):", 
        min_value=0.0, 
        max_value=200.0, 
        value=float(st.session_state.get("weight", 70.0) or 70.0), 
        step=1.0, 
        key="weight"
    )
    anthropometrics["height"] = st.number_input(
        "Height (m):", 
        min_value=0.0, 
        max_value=2.5, 
        value=float(st.session_state.get("height", 1.72) or 1.72), 
        key="height"
    )
    anthropometrics["age"] = st.number_input(
        "Age (years):", 
        min_value=0, 
        max_value=120, 
        value=int(st.session_state.get("age", 30) or 30),
        key="age"
    )
    
    # Convert sex string to numeric (1.0 for male, 0.0 for female)
    anthropometrics["sex"] = float(anthropometrics["sex"].lower() in ["male", "man", "men", "boy", "1", "chap", "guy"])
    
    return anthropometrics


def simulate_week(model, anthropometrics, drink_type, drink_grams_total, n_weeks=1):
    """Simulate weekly drinking pattern and return results.
    
    - model: sund model instance (should be 'alcohol_model')
    - anthropometrics: dict with 'sex', 'weight', 'height' keys
    - drink_type: "Beer", "Wine", or "Spirit"
    - drink_grams_total: total ethanol in grams for the week
    - n_weeks: number of weeks to simulate (default: 1)
    
    Returns: DataFrame with simulation results
    """
    specs = get_drink_specs(drink_type)
    drink_conc = specs["conc"]
    drink_volume = specs["volume"]
    drink_kcal_per_liter = specs["kcal"]
    drink_kcal = drink_kcal_per_liter * drink_volume
    drink_length = specs["length"]

    single_drink_grams = drink_conc * drink_volume * 0.7891 * 10

    # Generate drinking schedule across days
    drink_times = [[], [], [], [], [], [], []]
    drink_lengths = [[], [], [], [], [], [], []]
    drink_concentrations = [[], [], [], [], [], [], []]
    drink_volumes = [[], [], [], [], [], [], []]
    drink_kcals = [[], [], [], [], [], [], []]

    start_time = 18.0
    drinks_total = drink_grams_total / single_drink_grams
    drinks_per_day = drinks_total / 7

    for drink in range(0, int(np.ceil(drinks_per_day)) + 1):
        for day in range(5, -2, -1):
            if drinks_total > 0:
                drink_times[day].append(start_time)
                drink_lengths[day].append(drink_length)
                drink_concentrations[day].append(drink_conc)
                drink_volumes[day].append(min(1, drinks_total) * drink_volume)
                drink_kcals[day].append(drink_kcal)
                drinks_total -= 1
        start_time += 1

    # Flatten and repeat for n_weeks
    drink_times = [t + 24*day for day, times in enumerate(drink_times) for t in times]
    drink_lengths = flatten(drink_lengths)
    drink_concentrations = flatten(drink_concentrations)
    drink_volumes = flatten(drink_volumes)
    drink_kcals = flatten(drink_kcals)

    drink_times = [t + 24*7*week for week in range(0, n_weeks) for t in drink_times]
    drink_lengths = drink_lengths * n_weeks
    drink_concentrations = drink_concentrations * n_weeks
    drink_volumes = drink_volumes * n_weeks
    drink_kcals = drink_kcals * n_weeks

    # Generate meal schedule
    daily_kcal = 2500 if anthropometrics["sex"] == 1 else 2000

    meal_times = [t + 24*d + 24*7*w for w in range(n_weeks) for d in range(0, 7) for t in [7, 12, 18]]
    meal_lengths = 30.0 / 60
    meal_times = [t + meal_lengths*on for t in meal_times for on in [0, 1]]

    meal_kcals = [0.2*daily_kcal, 0.4*daily_kcal, 0.4*daily_kcal] * 7 * n_weeks
    meal_kcals = [k*on for k in meal_kcals for on in [1, 0]]

    # Build stimulus using helper
    stim = build_stimulus_dict(
        drink_times, drink_lengths, drink_concentrations, 
        drink_volumes, drink_kcals, meal_times, [0] + meal_kcals
    )

    sim_results = simulate(model, anthropometrics, stim, extra_time=12)
    return sim_results


def prune_session_keys(page, prefix, indices, keys):
    """Remove session_state keys for given indices and key name patterns.

    - page: page id string (e.g., '02')
    - prefix: base prefix like 'drink' or 'meal'
    - indices: iterable of integer indices to prune
    - keys: list of suffix patterns (e.g., ["time", "time_locked", "length0"]) - will be formatted
    """
    for i in indices:
        for k in keys:
            full = f"{k.format(prefix=prefix, page=page, i=i)}"
            if full in st.session_state:
                del st.session_state[full]


def ensure_prev_counter(page, name, default=0):
    key = f"_prev_{name}_{page}"
    if key not in st.session_state:
        st.session_state[key] = default
    return key


def seed_new_items(page, name, n, default_start, step, seed_key_template, lock_key_template, key_prefix=None):
    """Seed newly added items for a list field.

    - seed_key_template e.g. 'drink_time_{page}_{i}' (pass with braces for format)
    - lock_key_template e.g. 'drink_time_locked_{page}_{i}'
    Returns nothing; writes to st.session_state.
    """
    # Determine a stable prefix to use in generated keys (allow plural or singular names)
    prefix = key_prefix if key_prefix is not None else (name[:-1] if isinstance(name, str) and name.endswith('s') else name)

    # Determine which prev counter key is present or should be used. Support variants for backwards compatibility.
    candidates = [f"_prev_{name}_{page}", f"_prev_n_{name}_{page}", f"_prev_{prefix}_{page}"]
    prev_key = next((c for c in candidates if c in st.session_state), candidates[0])
    prev = st.session_state.get(prev_key, 0)
    if n < prev:
        # prune removed indices
        prune_session_keys(page, prefix, range(n, prev), [seed_key_template, lock_key_template, f"{prefix}_length{{i}}", f"{prefix}_concentrations{{i}}", f"{prefix}_volumes{{i}}", f"{prefix}_kcals{{i}}"])
        st.session_state[prev_key] = n
        return
    if n == prev:
        return
    last_index = prev - 1
    if last_index >= 0:
        last_time = st.session_state.get(seed_key_template.format(prefix=prefix, page=page, i=last_index))
    else:
        last_time = None
    if last_time is None:
        last_time = default_start + (last_index if last_index >= 0 else 0) * step
    for i in range(prev, n):
        st.session_state.setdefault(seed_key_template.format(prefix=prefix, page=page, i=i), last_time + (i - last_index) * step if last_index >= 0 else default_start + i * step)
        st.session_state.setdefault(lock_key_template.format(prefix=prefix, page=page, i=i), False)
    st.session_state[prev_key] = n


def lock_all(page, what, n, locked=True):
    """Set all lock flags for items on a page."""
    for i in range(n):
        st.session_state[f"{what}_time_locked_{page}_{i}"] = locked


def on_change_time_propagate(page, what, index, n, step):
    """Propagate time changes forward for an item list (drinks or meals).

    - what: 'drink' or 'meal'
    - index: int index that changed
    - n: total count
    - step: hours to add per subsequent item
    """
    base_key = f"{what}_time_{page}_{index}"
    base_time = st.session_state.get(base_key, None)
    if base_time is None:
        return
    for j in range(index + 1, n):
        lock_key = f"{what}_time_locked_{page}_{j}"
        if not st.session_state.get(lock_key, False):
            st.session_state[f"{what}_time_{page}_{j}"] = base_time + (j - index) * step


def enforce_minimum_time(page, what, index, n, min_gap=None):
    """Enforce that an item's time is not before the previous item's time + min_gap.
    
    If the current time is less than the previous item's time + min_gap,
    reset it to the previous time + min_gap (or min_gap if index 0).
    
    - page: page id string (e.g., '02')
    - what: 'drink' or 'meal'
    - index: int index of the item to check
    - n: total count
    - min_gap: minimum gap from previous item. If None, uses duration of previous item.
               For drinks, this is drink_length (in minutes), for meals it defaults to 0.0.
    """
    current_key = f"{what}_time_{page}_{index}"
    current_time = st.session_state.get(current_key, 0.0)
    
    if index == 0:
        # First item must be >= 0
        if current_time < 0.0:
            st.session_state[current_key] = 0.0
    else:
        # Subsequent items must be >= previous item's time + min_gap
        prev_key = f"{what}_time_{page}_{index - 1}"
        prev_time = st.session_state.get(prev_key, 0.0)
        
        # Compute min_gap from previous item's duration if not provided
        if min_gap is None:
            # Get the duration of the previous item in hours
            prev_length_key = f"{what}_length{index - 1}"
            prev_length_minutes = st.session_state.get(prev_length_key, 0.0)
            min_gap = prev_length_minutes / 60.0  # Convert minutes to hours
        
        min_allowed = prev_time + min_gap
        if current_time < min_allowed:
            st.session_state[current_key] = min_allowed


def draw_drink_timeline_plotly(sim_df, feature, drink_starts, drink_lengths, title=None, uncert_time=None, uncert_min=None, uncert_max=None, uncert_color='rgba(200,200,200,0.25)'):
    """Return a plotly Figure showing the simulation line and drink-duration rectangles at the bottom.

    - sim_df: pandas DataFrame with 'Time' column (hours) and the feature column
    - feature: column name to plot
    - drink_starts: list of start times (hours)
    - drink_lengths: list of durations (minutes)
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        raise

    x = sim_df['Time'].tolist()
    y = sim_df[feature].tolist() if feature in sim_df.columns else [0] * len(x)

    fig = go.Figure()
    
    # Optional uncertainty band
    if uncert_time is not None and uncert_min is not None and uncert_max is not None:
        fig.add_trace(
            go.Scatter(
                x=list(uncert_time) + list(reversed(uncert_time)), 
                y=list(uncert_max) + list(reversed(uncert_min)),
                fill='toself', fillcolor=uncert_color, line=dict(color='rgba(255,255,255,0)'), 
                hoverinfo='skip', showlegend=True, name='Uncertainty'
            )
        )

    # Main simulation line
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Simulation', line=dict(color='blue')))

    # Determine y-range and space for timeline
    y_min = min(y) if y else 0
    y_max = max(y) if y else 1
    yrange = y_max - y_min if (y_max - y_min) != 0 else 1.0

    # Timeline occupies bottom 6% of y-range
    timeline_height = 0.04 * yrange
    y0 = y_min - 0.02 * yrange
    y1 = y0 + timeline_height

    # Add colored rectangles for each drink
    colors = ['rgba(255,127,0,0.4)', 'rgba(127,0,255,0.35)', 'rgba(0,127,255,0.35)', 'rgba(0,200,0,0.35)']
    for i, start in enumerate(drink_starts):
        dur_min = drink_lengths[i] if i < len(drink_lengths) else 0
        end = start + (dur_min / 60.0)
        color = colors[i % len(colors)]
        
        # Rectangle for drink
        fig.add_shape(
            type='rect', x0=start, x1=end, y0=y0, y1=y1, fillcolor=color, line=dict(width=0),
            name=f'drink_{i}'
        )

    # Update layout
    fig.update_layout(margin=dict(l=40, r=20, t=80, b=60), hovermode='closest')
    fig.update_xaxes(title_text='Time (h)')
    fig.update_yaxes(title_text=feature)
    if title:
        fig.update_layout(title=title)

    return fig


def create_multi_feature_plot(sim_results, selected_features, uncert_data=None, demo_scenario=None, demo_color=None, feature_map=None, drink_starts=None, drink_lengths=None, data_points=None, data_sem=5.0):
    """Create a multi-feature Plotly grid plot (1x1 for single feature, nx2 for multiple).
    
    - sim_results: pandas DataFrame with 'Time' column and feature columns
    - selected_features: list of feature names to plot
    - uncert_data: optional dict with uncertainty data (for page 08)
    - demo_scenario: optional scenario key for uncertainty lookup
    - demo_color: optional color for uncertainty band (hex string)
    - feature_map: optional dict to map feature names to uncertainty keys
    - drink_starts: optional list of drink start times (hours)
    - drink_lengths: optional list of drink durations (minutes)
    - data_points: optional dict with feature names as keys and lists of {"time": t, "value": v} as values
    - data_sem: optional SEM (Standard Error of the Mean) as percentage (default: 5.0)
    
    Returns: plotly Figure object
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        raise
    
    if not selected_features:
        return None
    
    # Calculate grid dimensions
    n_features = len(selected_features)
    if n_features == 1:
        n_rows, n_cols = 1, 1
    else:
        n_rows = (n_features + 1) // 2  # Ceiling division
        n_cols = 2
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=selected_features,
        horizontal_spacing=0.15,
        vertical_spacing=0.15
    )
    
    for idx, feature in enumerate(selected_features):
        if n_cols == 1:
            row_idx = idx + 1
            col_idx = 1
        else:
            row_idx = (idx // 2) + 1
            col_idx = (idx % 2) + 1
        
        if feature in sim_results.columns:
            # Add uncertainty band if provided
            if uncert_data and demo_scenario and demo_color and feature_map:
                mapped_feature = feature_map.get(feature, feature)
                if demo_scenario in uncert_data and mapped_feature in uncert_data[demo_scenario]:
                    feat_data = uncert_data[demo_scenario][mapped_feature]
                    uncert_time = np.array(feat_data['Time']) / 60.0  # Convert minutes to hours
                    uncert_max = np.array(feat_data['Max'])
                    uncert_min = np.array(feat_data['Min'])
                    
                    # Convert hex color to rgba
                    rgb = tuple(int(demo_color[i:i+2], 16) for i in (1, 3, 5))
                    rgba_str = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.25)'
                    
                    # Add uncertainty band
                    fig.add_trace(
                        go.Scatter(
                            x=list(uncert_time) + list(reversed(uncert_time)),
                            y=list(uncert_max) + list(reversed(uncert_min)),
                            fill='toself',
                            fillcolor=rgba_str,
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo='skip',
                            showlegend=(idx == 0),
                            name='Uncertainty',
                            legendgroup='uncertainty'
                        ),
                        row=row_idx, col=col_idx
                    )
            
            # Add simulation line
            fig.add_trace(
                go.Scatter(
                    x=sim_results['Time'],
                    y=sim_results[feature],
                    mode='lines',
                    name='Simulation',
                    line=dict(width=2, color=demo_color) if demo_color else dict(width=2),
                    showlegend=(idx == 0),
                    legendgroup='simulation'
                ),
                row=row_idx, col=col_idx
            )

            # Add ±5% confidence band if data_points provided
            if data_points:
                # Add experimental data points if they exist for this feature
                if feature in data_points and data_points[feature]:
                    data_times = [float(dp['time']) for dp in data_points[feature]]
                    data_values = [float(dp['value']) for dp in data_points[feature]]
                    
                    # Calculate ±SEM error bars
                    data_values_array = np.array(data_values)
                    error_values = data_values_array * (data_sem / 100.0)
                    
                    # Add experimental data points with error bars
                    fig.add_trace(
                        go.Scatter(
                            x=data_times,
                            y=data_values,
                            mode='markers',
                            name=f'Experimental data (±{data_sem}%)',
                            marker=dict(size=10, color='red', symbol='circle'),
                            error_y=dict(
                                type='data',
                                array=error_values,
                                visible=True,
                                color='rgba(255, 0, 0, 0.6)',
                                thickness=2
                            ),
                            showlegend=(idx == 0),
                            legendgroup='data'
                        ),
                        row=row_idx, col=col_idx
                    )
            
            # Add drink duration rectangles if provided
            if drink_starts and drink_lengths:
                # Store drink info for later layout update (we'll set fixed y-positions after layout)
                # For now, use a relative position that will be adjusted in layout
                # Get data range for positioning
                y_data = sim_results[feature].dropna()
                y_min_data = y_data.min() if len(y_data) > 0 else 0
                y_max_data = y_data.max() if len(y_data) > 0 else 1
                yrange = y_max_data - y_min_data if (y_max_data - y_min_data) != 0 else 1.0
                
                # Position drinks at bottom 4% of visible range
                timeline_height = 0.04 * yrange
                y0 = y_min_data - 0.06 * yrange
                y1 = y0 + timeline_height
                
                # Add colored rectangles for each drink
                colors = ['rgba(255,127,0,0.4)', 'rgba(127,0,255,0.35)', 'rgba(0,127,255,0.35)', 'rgba(0,200,0,0.35)']
                for i, start in enumerate(drink_starts):
                    dur_min = drink_lengths[i] if i < len(drink_lengths) else 0
                    end = start + (dur_min / 60.0)
                    color = colors[i % len(colors)]
                    
                    # Add rectangle for drink duration
                    fig.add_shape(
                        type='rect', x0=start, x1=end, y0=y0, y1=y1, fillcolor=color, line=dict(width=0),
                        row=row_idx, col=col_idx
                    )
            
            # Update axes
            fig.update_xaxes(title_text='Time (h)', row=row_idx, col=col_idx)
            fig.update_yaxes(title_text=feature, row=row_idx, col=col_idx)
    
    # Update layout
    fig.update_layout(
        height=400 * n_rows,
        hovermode='closest',
        margin=dict(l=50, r=50, t=100, b=60),
        showlegend=bool(uncert_data) or bool(data_points)
    )
    
    return fig
