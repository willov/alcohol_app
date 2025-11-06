import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sidebar_config import setup_sidebar
from functions.ui_helpers import (
    setup_sund_package, setup_model, simulate,
    seed_new_items, on_change_time_propagate,
    on_change_duration_validate_next, lock_all,
    enforce_minimum_time, build_stimulus_dict,
    create_multi_feature_plot
)

# Setup sund and sidebar
sund = setup_sund_package()
setup_sidebar()

# Setup the model
model, model_features = setup_model('alcohol_model_2')

# Helper functions for page 08
def load_uncertainty(path="./results/UC_PPL_alcohol_model_individual.json"):
    """Load uncertainty data from PPL results JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# Start the app

st.markdown("# Plausibility of behavior of secondary alcohol metabolites")
st.write("Plausibility of secondary alcohol metabolites following claimed alcohol consumption")
st.markdown("""Using the model, we predict the plausibility of the behavior of secondary alcohol metabolites - including ethyl glucuronide (EtG), ethyl sulphate (EtS), and urine alcohol concentration (UAC). 

Below, a showcase is presented - including a set of sampled data points and a claimed consumption of two alcoholic drinks. The model is used to simulate the expected time course of the secondary metabolites, and compare these to the sampled data points. You can choose between a man or a woman.
""")

# Load uncertainty and data
uncert_data = load_uncertainty("./results/UC_PPL_alcohol_model_individual.json")
with open("./data/data_hip_flask_scenarios.json", 'r') as f:
    data_scenarios = json.load(f)

# === SHOWCASE SECTION ===
st.header("Showcase: model uncertainty vs measured data")
st.markdown("This section displays the model's prediction uncertainty compared to actual measured data from controlled experiments.")

# Sex toggle for showcase
def _on_sex_change():
    # Clear drink-related session state so defaults get recalculated
    for i in range(5):  # Clear up to 5 drinks
        key_time = f"drink_time_08_{i}"
        if key_time in st.session_state:
            del st.session_state[key_time]
        lock_key = f"drink_time_locked_08_{i}"
        if lock_key in st.session_state:
            del st.session_state[lock_key]
    # Clear simulation results so they get recalculated
    if 'sim_results' in st.session_state:
        del st.session_state['sim_results']
    if 'demo_anthropometrics' in st.session_state:
        del st.session_state['demo_anthropometrics']
    st.rerun()

showcase_sex = st.selectbox("Select sex for showcase:", ["Man", "Woman"], key="showcase_sex", on_change=_on_sex_change)


# Create 4-panel plot (BAC, UAC, EtG, EtS) - Interactive Plotly version
st.subheader(f"Model prediction of wine + vodka scenario ({showcase_sex})".capitalize())

features_to_plot = ["EtOH", "UAC", "EtG", "EtS"]
feature_labels = ["BAC", "UAC", "EtG", "EtS"]
feature_ylabels = ["mg/dL", "mg/dL", "", ""]

# Color scheme matching the manuscript
if showcase_sex == "Man":
    model_color = '#FF7F00'  # Orange
else:
    model_color = '#FF007D'  # Pink

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=feature_labels,
    horizontal_spacing=0.15,
    vertical_spacing=0.15
)

for idx, (feat, label, ylabel) in enumerate(zip(features_to_plot, feature_labels, feature_ylabels)):
    row_idx = (idx // 2) + 1
    col_idx = (idx % 2) + 1
    
    # Get plotting_info from data if available
    plot_info = {}
    if showcase_sex in data_scenarios:
        obs_data = data_scenarios[showcase_sex].get('Observables', {})
        if feat in obs_data:
            plot_info = obs_data[feat].get('plotting_info', {})
    
    # Plot uncertainty band
    if uncert_data and showcase_sex in uncert_data:
        if feat in uncert_data[showcase_sex]:
            feat_data = uncert_data[showcase_sex][feat]
            time = np.array(feat_data['Time']) / 60.0  # Convert minutes to hours
            max_val = np.array(feat_data['Max'])
            min_val = np.array(feat_data['Min'])
            
            # Add uncertainty band
            fig.add_trace(
                go.Scatter(
                    x=list(time) + list(reversed(time)),
                    y=list(max_val) + list(reversed(min_val)),
                    fill='toself',
                    fillcolor=f'rgba({int(model_color[1:3], 16)}, {int(model_color[3:5], 16)}, {int(model_color[5:7], 16)}, 0.25)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    showlegend=(idx == 0),
                    name='Model uncertainty',
                    legendgroup='uncertainty'
                ),
                row=row_idx, col=col_idx
            )

    # Plot measured data points
    if showcase_sex in data_scenarios:
        obs_data = data_scenarios[showcase_sex]['Observables']
        if feat in obs_data:
            time_data = obs_data[feat]['Time']
            mean_data = obs_data[feat]['Mean']
            # Filter out NaN values
            valid_idx = [i for i, val in enumerate(mean_data) if not (isinstance(val, float) and np.isnan(val))]
            time_valid = [time_data[i] / 60.0 for i in valid_idx]  # Convert minutes to hours
            mean_valid = [mean_data[i] for i in valid_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=time_valid,
                    y=mean_valid,
                    mode='markers',
                    marker=dict(color='black', size=6),
                    hoverinfo='x+y',
                    showlegend=(idx == 0),
                    name='Data',
                    legendgroup='data'
                ),
                row=row_idx, col=col_idx
            )
    
    # Update axes with plotting info
    xlim = plot_info.get('xlim', [0, 900])
    xlim = [x / 60.0 for x in xlim]
    ylim = plot_info.get('ylim')
    
    fig.update_xaxes(title_text='Time (hours)', range=xlim, row=row_idx, col=col_idx)
    fig.update_yaxes(
        title_text=plot_info.get('ylabel', ylabel if ylabel else ''),
        row=row_idx, col=col_idx
    )
    if ylim:
        fig.update_yaxes(range=ylim, row=row_idx, col=col_idx)

# Update layout
fig.update_layout(
    height=900,
    hovermode='closest',
    margin=dict(l=50, r=50, t=100, b=60)
)

st.plotly_chart(fig, use_container_width=True, key="showcase_plot")

# Add drink timeline visualization
if showcase_sex == "Man":
    st.markdown("**Drink timeline:** Wine at t=[0, 0.25, 0.5, 0.75] h | Vodka at t=[5] h")
else:
    st.markdown("**Drink timeline:** Wine at t=[0, 0.25, 0.5, 0.75] h | Vodka at t=[2] h")
st.divider()

# === INTERACTIVE DEMO SECTION ===
st.header("Interactive demo: simulate your own scenario")
st.markdown("Specify your own drinking pattern and anthropometric characteristics to simulate alcohol kinetics. The simulation will run for the same duration as the model uncertainty data (~24 hours), allowing full comparison of your scenario against the uncertainty bounds (no measured data shown in this section). This is a tool intended for exploration and educational purposes, where you are intended to investigate how different claimed drinking patterns and anthropometrics affect the predicted alcohol kinetics.")

# Anthropometrics            
st.subheader("Anthropometrics")

st.write(f"**Sex:** {showcase_sex} (linked to showcase selection)")
anthropometrics = {
    "sex": float(showcase_sex.lower() in ["male", "man", "men", "boy", "1", "chap", "guy"]),
    "weight": st.number_input("Weight (kg):", 0.0, 200.0, 70.0, 1.0, key="weight_08"),
    "height": st.number_input("Height (m):", 0.0, 2.5, 1.72, key="height_08"),
    "age": st.number_input("Age (years):", 0, 120, 30, key="age_08")
}

# Specifying the drinks
st.subheader("Specifying the alcoholic drinks")


# Number of drinks for demo page
def _on_change_n_drinks_08():
    seed_new_items(
        page="08", name="drinks", n=st.session_state.get("n_drinks_08", 1), default_start=15.0, step=1.0,
        seed_key_template="{prefix}_time_{page}_{i}", lock_key_template="{prefix}_time_locked_{page}_{i}", key_prefix="drink"
    )
    _trigger_simulation_update_08()

n_drinks = st.slider("Number of drinks:", 1, 5, 1, key="n_drinks_08", on_change=_on_change_n_drinks_08)

# Lock all / Unlock all controls for drinks
la, lb = st.columns(2)
if la.button("Lock all drinks", key="lock_all_drinks_08"):
    lock_all(page="08", what="drink", n=n_drinks, locked=True)
if lb.button("Unlock all drinks", key="unlock_all_drinks_08"):
    lock_all(page="08", what="drink", n=n_drinks, locked=False)

drink_times = []
drink_lengths = []
drink_concentrations = []
drink_volumes = []
drink_kcals = []

st.divider()
start_time = 15.0  # Start at 15:00 (3 PM)

def _on_change_drink_time_08(index):
    enforce_minimum_time(page="08", what="drink", index=index, min_gap=None)
    on_change_time_propagate(page="08", what="drink", index=index, n=st.session_state.get("n_drinks_08", 1), step=1.0)
    _trigger_simulation_update_08()

def _on_change_drink_length_08(index):
    on_change_duration_validate_next(page="08", what="drink", index=index, n=st.session_state.get("n_drinks_08", 1), min_gap=None)
    _trigger_simulation_update_08()

def _on_change_meal_time_08(index):
    # Enforce that this meal's time is not before the previous meal's time + 10 minutes
    enforce_minimum_time(page="08", what="meal", index=index, min_gap=10.0/60.0)  # 10 minutes in hours
    on_change_time_propagate(page="08", what="meal", index=index, n=st.session_state.get("n_meals_08", 0), step=6.0)
    _trigger_simulation_update_08()

def _trigger_simulation_update_08():
    """Trigger simulation update when drink/meal parameters change."""
    # This will be called after collecting all drink/meal data
    st.session_state['_should_update_sim_08'] = True

def _refresh_drink_times_08(n_drinks_count):
    """Refresh drink times with consistent 15-minute spacing."""
    start_time = 15.0
    min_gap_hours = 0.25  # 15 minutes in hours
    
    # Set all drinks with consistent 15-minute spacing
    for i in range(n_drinks_count):
        st.session_state[f"drink_time_08_{i}"] = start_time + i * min_gap_hours

# Initialize drinks using seed_new_items
seed_new_items(
    page="08", name="drinks", n=n_drinks, default_start=start_time, step=0.25,
    seed_key_template="{prefix}_time_{page}_{i}", lock_key_template="{prefix}_time_locked_{page}_{i}", key_prefix="drink"
)

# Reset drink times on sex change
_refresh_drink_times_08(n_drinks)

for i in range(n_drinks):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        drink_times.append(st.number_input("Time (h)", 0.0, 100.0, key=f"drink_time_08_{i}", on_change=_on_change_drink_time_08, args=(i,)))
    lock_key = f"drink_time_locked_08_{i}"
    st.checkbox("Lock", key=lock_key, help="Prevent auto-fill changes to this drink time")
    with col2:
        drink_lengths.append(st.number_input("Length (min)", 0.0, 240.0, 20.0, 1.0, key=f"drink_length{i}", on_change=_on_change_drink_length_08, args=(i,)))
    with col3:
        drink_concentrations.append(st.number_input("ABV (%)", 0.0, 100.0, 5.0, 0.1, key=f"drink_concentrations{i}"))
    with col4:
        drink_volumes.append(st.number_input("Vol (L)", 0.0, 24.0, 1.0, 0.1, key=f"drink_volumes{i}"))
    with col5:
        drink_kcals.append(st.number_input("kcal", 0.0, 1000.0, 135.0, 10.0, key=f"drink_kcals{i}"))

# Setup meals (interactive)
meal_times = []
meal_kcals = []

def _on_change_n_meals_08():
    seed_new_items(
        page="08", name="meals", n=st.session_state.get("n_meals_08", 0), default_start=12.0, step=6.0,
        seed_key_template="{prefix}_time_{page}_{i}", lock_key_template="{prefix}_time_locked_{page}_{i}", key_prefix="meal"
    )

n_meals = st.slider("Number of (solid) meals:", 0, 15, 1, key="n_meals_08", on_change=_on_change_n_meals_08)

# Lock all / Unlock all controls for meals
lm_a, lm_b = st.columns(2)
if lm_a.button("Lock all meals", key="lock_all_meals_08"):
    lock_all(page="08", what="meal", n=n_meals, locked=True)
if lm_b.button("Unlock all meals", key="unlock_all_meals_08"):
    lock_all(page="08", what="meal", n=n_meals, locked=False)

st.divider()
start_time_meal = 12.0

def _on_change_meal_time_08(index):
    # Enforce that this meal's time is not before the previous meal's time + 10 minutes
    enforce_minimum_time(page="08", what="meal", index=index, min_gap=10.0/60.0)  # 10 minutes in hours
    on_change_time_propagate(page="08", what="meal", index=index, n=st.session_state.get("n_meals_08", 0), step=6.0)

# Initialize meal defaults and locks for page 08
for i in range(n_meals):
    key_time = f"meal_time_08_{i}"
    lock_key = f"meal_time_locked_08_{i}"
    if key_time not in st.session_state:
        st.session_state[key_time] = start_time_meal + i * 6.0
    if lock_key not in st.session_state:
        st.session_state[lock_key] = False

for i in range(n_meals):
    col1, col2 = st.columns(2)
    with col1:
        meal_times.append(st.number_input("Time (h)", 0.0, 100.0, key=f"meal_time_08_{i}", on_change=_on_change_meal_time_08, args=(i,)))
    lock_key = f"meal_time_locked_08_{i}"
    st.checkbox("Lock", key=lock_key, help="Prevent auto-fill changes to this meal time")
    with col2:
        meal_kcals.append(st.number_input("kcal", 0.0, 10000.0, 500.0, 10.0, key=f"meal_kcals{i}"))
    start_time_meal += 6

if n_meals < 1.0:
    st.divider()

meal_times = [t+(30/60)*on for t in meal_times for on in [0,1]]
meal_kcals = [0]+[m*on for m in meal_kcals for on in [1 , 0]]

# Setup stimulation to the model using helper
stim = build_stimulus_dict(
    drink_times, drink_lengths, drink_concentrations, 
    drink_volumes, drink_kcals, meal_times, meal_kcals
)

# Initialize session state for simulation results
if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = None
if 'demo_anthropometrics' not in st.session_state:
    st.session_state['demo_anthropometrics'] = None

# Calculate simulation time based on uncertainty data duration
# Get max time from uncertainty data (in minutes, convert to hours)
max_uncert_time_hours = 0
if uncert_data:
    for scenario in uncert_data.values():
        for feature_data in scenario.values():
            if 'Time' in feature_data:
                max_time = max(feature_data['Time']) / 60.0  # Convert minutes to hours
                max_uncert_time_hours = max(max_uncert_time_hours, max_time)

# Run simulation button
if st.button("Run Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        # Calculate extra_time needed to reach the uncertainty data duration
        # The simulation should cover the same time range as the uncertainty data
        # Uncertainty starts at 0 (representing the first drink), so we need to simulate
        # from t_start to max_uncert_time_hours
        first_drink_time = min(drink_times) if drink_times else 15.0
        extra_time = max_uncert_time_hours  # Total time from start
        
        st.session_state['sim_results'] = simulate(model, anthropometrics, stim, extra_time=extra_time)
        st.session_state['demo_anthropometrics'] = anthropometrics.copy()
    st.success("✅ Simulation complete!")

# Auto-update simulation when drink/meal parameters change
if st.session_state.get('_should_update_sim_08', False):
    with st.spinner("Updating simulation..."):
        first_drink_time = min(drink_times) if drink_times else 15.0
        extra_time = max_uncert_time_hours  # Total time from start
        
        st.session_state['sim_results'] = simulate(model, anthropometrics, stim, extra_time=extra_time)
        st.session_state['demo_anthropometrics'] = anthropometrics.copy()
    st.session_state['_should_update_sim_08'] = False
    st.rerun()

# Display results if simulation has been run
if st.session_state['sim_results'] is not None:
    sim_results = st.session_state['sim_results']
    demo_anthropometrics = st.session_state['demo_anthropometrics']
    
    # Define allowed features for page 08
    allowed_features = ["EtOH", "UAC", "EtG", "EtS"]
    available_features = [f for f in allowed_features if f in model_features]
    
    # Select features to plot
    selected_features = st.multiselect("Select features to visualize:", available_features, default=available_features, key="plot_features_08")
    
    if selected_features:
        # Determine which scenario to use for uncertainty based on sex
        demo_scenario = "Man" if demo_anthropometrics["sex"] == 1.0 else "Woman"
        demo_color = '#FF7F00' if demo_anthropometrics["sex"] == 1.0 else '#FF007D'
        
        # Map feature names (model uses different names)
        feature_map = {"BAC": "EtOH", "EtOH": "EtOH", "UAC": "UAC", "EtG": "EtG", "EtS": "EtS"}
        
        # Adjust sim_results time to start at 0 for plotting against uncertainty
        sim_df = sim_results.copy()
        sim_df['Time'] = sim_df['Time'] - sim_df['Time'].min()
        
        # Adjust drink times to start at 0 (relative to first drink)
        first_drink_time = min(drink_times) if drink_times else 15.0
        drink_times_relative = [t - first_drink_time for t in drink_times]
        
        fig = create_multi_feature_plot(
            sim_df, 
            selected_features,
            uncert_data=uncert_data,
            demo_scenario=demo_scenario,
            demo_color=demo_color,
            feature_map=feature_map,
            drink_starts=drink_times_relative,
            drink_lengths=drink_lengths,
            uncertainty_legend="Target interval"
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"plot_08_multi")
    else:
        st.info("Select at least one feature to plot.")
else:
    st.info("Click the button above to run the simulation with your chosen parameters.")
