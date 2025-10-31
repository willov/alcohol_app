import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Install sund in a custom location
import subprocess
import sys
Path("./custom_package").mkdir(parents=True, exist_ok=True)
if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", "sund<3.0"])

sys.path.append('./custom_package')
import sund
from sidebar_config import setup_sidebar

# Setup sidebar
setup_sidebar()

# Main content
# Setup the models

def setup_model(model_name):
    sund.install_model(f"./models/{model_name}.txt")
    model_class = sund.import_model(model_name)
    model = model_class() 

    with open("./results/alcohol_model_2 (642.74446).json",'r') as f:
        param_in = json.load(f)
        params = param_in['x']

    model.parameter_values = params
    features = model.feature_names
    return model, features

model, model_features = setup_model('alcohol_model_2')

# Define functions needed

def flatten(list):
    return [item for sublist in list for item in sublist]

def load_uncertainty(path="./results/UC_PPL_alcohol_model_individual.json"):
    """Load uncertainty data from PPL results JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def plot_with_uncertainty(ax, uncert_data, scenario_key, feature, color, label=None, convert_to_hours=False):
    """Plot median line with shaded uncertainty band."""
    if scenario_key not in uncert_data:
        return
    if feature not in uncert_data[scenario_key]:
        return
    
    feat_data = uncert_data[scenario_key][feature]
    time = np.array(feat_data['Time'])
    if convert_to_hours:
        time = time / 60.0
    max_val = np.array(feat_data['Max'])
    min_val = np.array(feat_data['Min'])
    median = (max_val + min_val) / 2
    
    ax.plot(time, median, color=color, linewidth=2, label=label)
    ax.fill_between(time, min_val, max_val, color=color, alpha=0.25)

def simulate(m, anthropometrics, stim, extra_time = 10):
    act = sund.Activity(time_unit = 'h')

    for key,val in stim.items():
        act.add_output(name = key, type='piecewise_constant', t = val["t"], f = val["f"]) 
    for key,val in anthropometrics.items():
        act.add_output(name = key, type='constant', f = val) 
    
    sim = sund.Simulation(models = m, activities = act, time_unit = 'h')
    
    t_start = min(stim["EtOH_conc"]["t"]+stim["kcal_solid"]["t"])-0.25

    sim.simulate(time = np.linspace(t_start, max(stim["EtOH_conc"]["t"])+extra_time, 10000))
    
    sim_results = pd.DataFrame(sim.feature_values,columns=sim.feature_names)
    sim_results.insert(0, 'Time', sim.time_vector)

    t_start_drink = min(stim["EtOH_conc"]["t"])-0.25

    sim_drink_results = sim_results[(sim_results['Time']>=t_start_drink)]
    return sim_drink_results

# Start the app

st.markdown("# Plausibility of behavior of secondary alcohol metabolites")
st.write("Plausibility of secondary alcohol metabolites following claimed alcohol consumption")
st.markdown("""Using the model, we predict the plausibility of the behavior of secondary alcohol metabolites - including ethyl glucuronide (EtG), ethyl sulphate (EtS), and urine alcohol concentration (UAC). 

Below, a showcase is presented - including a set of sampled data points and a claimed consumption of two alcoholic drinks. The model is used to simulate the expected time course of the secondary metabolites, and compare these to the sampled data points. You can choose between a female and a male.
""")

# Load uncertainty and data
uncert_data = load_uncertainty("./results/UC_PPL_alcohol_model_individual.json")
with open("./data/data_hip_flask_scenarios.json", 'r') as f:
    data_scenarios = json.load(f)

# === SHOWCASE SECTION ===
st.header("📊 Showcase: Model Uncertainty vs Measured Data")
st.markdown("This section displays the model's prediction uncertainty compared to actual measured data from controlled experiments.")

# Sex toggle for showcase
showcase_sex = st.selectbox("Select sex for showcase:", ["Man", "Woman"], key="showcase_sex")


# Create 4-panel plot (BAC, UAC, EtG, EtS)
st.subheader(f"Model prediction of wine + vodka scenario ({showcase_sex})")

features_to_plot = ["EtOH", "UAC", "EtG", "EtS"]
feature_labels = ["BAC", "UAC", "EtG", "EtS"]
feature_ylabels = ["mg/dL", "mg/dL", "", ""]

# Color scheme matching the manuscript
if showcase_sex == "Man":
    model_color = '#FF7F00'  # Orange
else:
    model_color = '#FF007D'  # Pink

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, (feat, label, ylabel) in enumerate(zip(features_to_plot, feature_labels, feature_ylabels)):
    ax = axes[idx]
    
    # Get plotting_info from data if available
    plot_info = {}
    if showcase_sex in data_scenarios:
        obs_data = data_scenarios[showcase_sex].get('Observables', {})
        if feat in obs_data:
            plot_info = obs_data[feat].get('plotting_info', {})
    
    # Plot uncertainty band
    if uncert_data and showcase_sex in uncert_data:
        plot_with_uncertainty(ax, uncert_data, showcase_sex, feat, model_color, label="Model uncertainty", convert_to_hours=True)

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
            ax.scatter(time_valid, mean_valid, c='black', s=40, zorder=5, label='Data')
    
    # Apply plotting settings from plotting_info
    ax.set_title(plot_info.get('title', label))
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel(plot_info.get('ylabel', ylabel if ylabel else ''))
    
    # Set axis limits from plotting_info
    xlim = plot_info.get('xlim', [0, 900])
    xlim = [x / 60.0 for x in xlim]
    ylim = plot_info.get('ylim')
    ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Add drink timeline visualization
if showcase_sex == "Man":
    st.markdown("**Drink timeline:** 🍷 Wine at t=0, 0.25, 0.5, 0.75 h | 🍸 Vodka at t=5 h")
else:
    st.markdown("**Drink timeline:** 🍷 Wine at t=0, 0.25, 0.5, 0.75 h | 🍸 Vodka at t=2 h")
st.divider()

# === INTERACTIVE DEMO SECTION ===
st.header("🔬 Interactive Demo: Simulate Your Own Scenario")
st.markdown("Specify your own drinking pattern and anthropometric characteristics to simulate alcohol kinetics. The simulation will run for the same duration as the model uncertainty data (~24 hours), allowing full comparison of your scenario against the uncertainty bounds (no measured data shown in this section). This is a tool intended for exploration and educational purposes, where you are intended to investigate how different claimed drinking patterns and anthropometrics affect the predicted alcohol kinetics.")

# Anthropometrics            
st.subheader("Anthropometrics")

# Shared variables between the pages
if 'sex' not in st.session_state:
    st.session_state['sex'] = 'Man'
if 'weight' not in st.session_state:
    st.session_state['weight'] = 70.0
if 'height' not in st.session_state:
    st.session_state['height'] = 1.72
if 'age' not in st.session_state:
    st.session_state['age'] = 30

# Sync sex with showcase selection
st.session_state['sex'] = showcase_sex

anthropometrics = {"sex": st.session_state['sex'], "weight": st.session_state['weight'], "height": st.session_state['height'], "age": st.session_state['age']}
st.write(f"**Sex:** {showcase_sex} (linked to showcase selection)")
anthropometrics["weight"] = st.number_input("Weight (kg):", 0.0, 200.0, st.session_state.weight, 1.0, key="weight")
anthropometrics["height"] = st.number_input("Height (m):", 0.0, 2.5, st.session_state.height, key="height")
anthropometrics["age"] = st.number_input("Age (years):", 0, 120, st.session_state.age, key="age")

anthropometrics["sex"] = float(anthropometrics["sex"].lower() in ["male", "man", "men", "boy", "1", "chap", "guy"]) #Converts to a numerical representation

# Specifying the drinks
st.subheader("Specifying the alcoholic drinks")

n_drinks = st.slider("Number of drinks:", 1, 5, 2, key="n_drinks_demo")
extra_time = st.number_input("Additional time to simulate after last drink (h):", 0.0, 100.0, 12.0, 0.1, key="extra_time_demo")

drink_times = []
drink_lengths = []
drink_concentrations = []
drink_volumes = []
drink_kcals = []

st.divider()
start_time = 15.0  # Start at 15:00 (3 PM)
for i in range(n_drinks):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        drink_times.append(st.number_input("Time (h)", 0.0, 100.0, start_time, 0.1, key=f"drink_time{i}"))
    with col2:
        drink_lengths.append(st.number_input("Length (min)", 0.0, 240.0, 20.0, 1.0, key=f"drink_length{i}"))
    with col3:
        drink_concentrations.append(st.number_input("ABV (%)", 0.0, 100.0, 5.0, 0.1, key=f"drink_concentrations{i}"))
    with col4:
        drink_volumes.append(st.number_input("Vol (L)", 0.0, 24.0, 0.33, 0.1, key=f"drink_volumes{i}"))
    with col5:
        drink_kcals.append(st.number_input("kcal", 0.0, 1000.0, 45.0, 10.0, key=f"drink_kcals{i}"))
    start_time += 1

EtOH_conc = [0]+[c*on for c in drink_concentrations for on in [1 , 0]]
vol_drink_per_time = [0]+[v/t*on if t>0 else 0 for v,t in zip(drink_volumes, drink_lengths) for on in [1 , 0]]
kcal_liquid_per_vol = [0]+[k/v*on if v>0 else 0 for v,k in zip(drink_volumes, drink_kcals) for on in [1 , 0]]
drink_length = [0]+[t*on for t in drink_lengths for on in [1 , 0]]
t = [t+(l/60)*on for t,l in zip(drink_times, drink_lengths) for on in [0,1]]

# Setup meals (simplified, with defaults)
meal_times = [12.0, 12.5]
meal_kcals = [0, 500, 0]

# Setup stimulation to the model
stim = {
    "EtOH_conc": {"t": t, "f": EtOH_conc},
    "vol_drink_per_time": {"t": t, "f": vol_drink_per_time},
    "kcal_liquid_per_vol": {"t": t, "f": kcal_liquid_per_vol},
    "drink_length": {"t": t, "f": drink_length},
    "kcal_solid": {"t": meal_times, "f": meal_kcals},
    }

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
if st.button("🚀 Run Simulation", type="primary"):
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

# Display results if simulation has been run
if st.session_state['sim_results'] is not None:
    sim_results = st.session_state['sim_results']
    demo_anthropometrics = st.session_state['demo_anthropometrics']
    
    # Select feature to plot
    feature = st.selectbox("Select feature to visualize:", model_features, index=model_features.index("BAC") if "BAC" in model_features else 0)
    
    # Create comparison plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    
    # Determine which scenario to use for uncertainty based on sex
    demo_scenario = "Man" if demo_anthropometrics["sex"] == 1.0 else "Woman"
    demo_color = '#FF7F00' if demo_anthropometrics["sex"] == 1.0 else '#FF007D'
    
    # Plot uncertainty (using appropriate scenario)
    if uncert_data and demo_scenario in uncert_data:
        # Map feature names (model uses different names)
        feature_map = {"BAC": "EtOH", "EtOH": "EtOH", "UAC": "UAC", "EtG": "EtG", "EtS": "EtS"}
        mapped_feature = feature_map.get(feature, feature)
        
        if mapped_feature in uncert_data[demo_scenario]:
            # Uncertainty time is in minutes, convert to hours
            uncert_time_hours = np.array(uncert_data[demo_scenario][mapped_feature]['Time']) / 60.0
            uncert_max = np.array(uncert_data[demo_scenario][mapped_feature]['Max'])
            uncert_min = np.array(uncert_data[demo_scenario][mapped_feature]['Min'])
            uncert_median = (uncert_max + uncert_min) / 2
            
            ax2.fill_between(uncert_time_hours, uncert_min, uncert_max, color=demo_color, alpha=0.25, label='Scenario truth')
    
    # Plot simulation - adjust time to start at 0 (representing 15:00)
    # The simulation time starts at drink time, need to shift to match uncertainty (which starts at 0)
    sim_time_adjusted = sim_results['Time'] - sim_results['Time'].min()
    if feature in sim_results.columns:
        ax2.plot(sim_time_adjusted, sim_results[feature], color='blue', linewidth=2, label='Your simulation', linestyle='--')
    
    ax2.set_xlabel("Time (hours since 15:00)")
    ax2.set_ylabel(feature)
    ax2.set_title(f"{feature} - Simulation vs Model Uncertainty")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)
else:
    st.info("👆 Click the button above to run the simulation with your chosen parameters.")
