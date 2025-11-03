import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Install sund in a custom location
import subprocess
import sys
Path("./custom_package").mkdir(parents=True, exist_ok=True)
if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", "sund<3.0"])

sys.path.append('./custom_package')
import sund

from sidebar_config import setup_sidebar
from functions.ui_helpers import seed_new_items, on_change_time_propagate, lock_all, draw_drink_timeline_plotly

# Setup sidebar
setup_sidebar()

# Setup the models

def setup_model(model_name):
    sund.install_model(f"./models/{model_name}.txt")
    model_class = sund.import_model(model_name)
    model = model_class() 

    with open("./results/alcohol_model (186.99).json",'r') as f:
        param_in = json.load(f)
        params = param_in['x']

    model.parameter_values = params
    features = model.feature_names
    return model, features

model, model_features = setup_model('alcohol_model')

# Define functions needed

def flatten(list):
    return [item for sublist in list for item in sublist]

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

st.markdown("# Alcohol dynamics")
st.write("Simulation of alcohol dynamics")
st.markdown("""Current models for alcohol dynamics lack detailed dynamics of e.g., gastric emptying. 
We have constructed a more detailed model able to both explain differing dynamics of different types of drinks, as well as differences based on anthropometrics. 
Using the model, you can simulate the dynamics of different drinks based on custom anthropometrics. 

Below, you can specify one or more alcoholic drinks, and some anthropometrics to get simulations of the ethanol concentration in blood as well as clinical markers of alcohol use like phosphatidylethanol (PEth).  

""")

# Anthropometrics            
st.subheader("Anthropometrics")

# Shared variables between the pages
if 'sex' not in st.session_state:
    st.session_state['sex'] = 'Man'
if 'weight' not in st.session_state:
    st.session_state['weight'] = 70.0
if 'height' not in st.session_state:
    st.session_state['height'] = 1.72

anthropometrics = {"sex": st.session_state['sex'], "weight": st.session_state['weight'], "height": st.session_state['height']}
anthropometrics["sex"] = st.selectbox("Sex:", ["Man", "Woman"], ["Man", "Woman"].index(st.session_state['sex']), key="sex")
anthropometrics["weight"] = st.number_input("Weight (kg):", 0.0, 1000.0, st.session_state.weight, 0.1, key="weight")
anthropometrics["height"] = st.number_input("Height (m):", 0.0, 2.5, st.session_state.height, key="height")

anthropometrics["sex"] = float(anthropometrics["sex"].lower() in ["male", "man", "men", "boy", "1", "chap", "guy"]) #Converts to a numerical representation

# Specifying the drinks
st.subheader("Specifying the alcoholic drinks")

# Callback when number of drinks changes: initialize new drink times to follow last existing time
def _on_change_n_drinks_02():
    # Use generic seeding/pruning helper for drinks on page 02
    seed_new_items(
        page="02", name="drink", n=st.session_state.get("n_drinks_02", 1), default_start=18.0, step=1.0,
        seed_key_template="{prefix}_time_{page}_{i}", lock_key_template="{prefix}_time_locked_{page}_{i}"
    )

# Number of drinks (store in session state so callbacks can read it)
n_drinks = st.slider("Number of drinks:", 1, 15, 1, key="n_drinks_02", on_change=_on_change_n_drinks_02)
extra_time = st.number_input("Additional time to simulate after last drink (h):", 0.0, 100.0, 12.0, 0.1)

# Lock all / Unlock all controls for drinks
lock_col_a, lock_col_b = st.columns(2)
if lock_col_a.button("Lock all drinks", key="lock_all_drinks_02"):
    for i in range(n_drinks):
        st.session_state[f"drink_time_locked_02_{i}"] = True
if lock_col_b.button("Unlock all drinks", key="unlock_all_drinks_02"):
    for i in range(n_drinks):
        st.session_state[f"drink_time_locked_02_{i}"] = False

drink_times = []
drink_lengths = []
drink_concentrations = []
drink_volumes = []
drink_kcals = []

st.divider()
start_time = 18.0

def _on_change_drink_time(index):
    on_change_time_propagate(page="02", what="drink", index=index, n=st.session_state.get("n_drinks_02", 1), step=1.0)

# Initialize default times and locked flags when not present in session_state
for i in range(n_drinks):
    key_time = f"drink_time_02_{i}"
    lock_key = f"drink_time_locked_02_{i}"
    if key_time not in st.session_state:
        st.session_state[key_time] = start_time + i * 1.0
    if lock_key not in st.session_state:
        st.session_state[lock_key] = False

for i in range(n_drinks):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        # attach on_change to allow auto-fill of subsequent times
        drink_times.append(st.number_input("Time (h)", 0.0, 100.0, key=f"drink_time_02_{i}", on_change=_on_change_drink_time, args=(i,)))
    # visible lock checkbox to prevent cascading changes
    lock_key = f"drink_time_locked_02_{i}"
    lock_label = "Lock 🔒" if st.session_state.get(lock_key, False) else "Lock"
    st.checkbox(lock_label, key=lock_key, help="Prevent auto-fill changes to this drink time")
    with col2:
        drink_lengths.append(st.number_input("Length (min)", 0.0, 240.0, 20.0, 1.0, key=f"drink_length{i}"))
    with col3:
        drink_concentrations.append(st.number_input("ABV (%)", 0.0, 100.0, 5.0, 0.01, key=f"drink_concentrations{i}"))
    with col4:
        drink_volumes.append(st.number_input("Vol (L)", 0.0, 24.0, 0.33, 0.1, key=f"drink_volumes{i}"))
    with col5:
        drink_kcals.append(st.number_input("kcal", 0.0, 1000.0, 45.0, 1.0, key=f"drink_kcals{i}"))
    start_time += 1

EtOH_conc = [0]+[c*on for c in drink_concentrations for on in [1 , 0]]
vol_drink_per_time = [0]+[v/t*on if t>0 else 0 for v,t in zip(drink_volumes, drink_lengths) for on in [1 , 0]]
kcal_liquid_per_vol = [0]+[k/v*on if v>0 else 0 for v,k in zip(drink_volumes, drink_kcals) for on in [1 , 0]]
drink_length = [0]+[t*on for t in drink_lengths for on in [1 , 0]]
t = [t+(l/60)*on for t,l in zip(drink_times, drink_lengths) for on in [0,1]]


# Setup meals
st.subheader(f"Specifying the meals")

start_time = 12.0

meal_times = []
meal_kcals = []

def _on_change_n_meals_02():
    # Use generic seeding/pruning helper for meals on page 02
    seed_new_items(
        page="02", name="meal", n=st.session_state.get("n_meals_02", 0), default_start=12.0, step=6.0,
        seed_key_template="{prefix}_time_{page}_{i}", lock_key_template="{prefix}_time_locked_{page}_{i}"
    )

n_meals = st.slider("Number of (solid) meals:", 0, 15, 1, key="n_meals_02", on_change=_on_change_n_meals_02)

# Lock all / Unlock all controls for meals
lockm_a, lockm_b = st.columns(2)
if lockm_a.button("Lock all meals", key="lock_all_meals_02"):
    lock_all(page="02", what="meal", n=n_meals, locked=True)
if lockm_b.button("Unlock all meals", key="unlock_all_meals_02"):
    lock_all(page="02", what="meal", n=n_meals, locked=False)

st.divider()

def _on_change_meal_time_02(index):
    on_change_time_propagate(page="02", what="meal", index=index, n=st.session_state.get("n_meals_02", 0), step=6.0)

# Initialize meal defaults and locks
for i in range(n_meals):
    key_time = f"meal_time_02_{i}"
    lock_key = f"meal_time_locked_02_{i}"
    if key_time not in st.session_state:
        st.session_state[key_time] = start_time + i * 6.0
    if lock_key not in st.session_state:
        st.session_state[lock_key] = False

for i in range(n_meals):
    col1, col2 = st.columns(2)
    with col1:
        meal_times.append(st.number_input("Time (h)", 0.0, 100.0, key=f"meal_time_02_{i}", on_change=_on_change_meal_time_02, args=(i,)))
    lock_key = f"meal_time_locked_02_{i}"
    lock_label = "Lock 🔒" if st.session_state.get(lock_key, False) else "Lock"
    st.checkbox(lock_label, key=lock_key, help="Prevent auto-fill changes to this meal time")
    with col2:
        meal_kcals.append(st.number_input("kcal", 0.0, 10000.0, 500.0, 1.0, key=f"meal_kcals{i}"))
    start_time += 6

if n_meals < 1.0:
    st.divider()

meal_times = [t+(30/60)*on for t in meal_times for on in [0,1]]
meal_kcals = [0]+[m*on for m in meal_kcals for on in [1 , 0]]


# Setup stimulation to the model

stim = {
    "EtOH_conc": {"t": t, "f": EtOH_conc},
    "vol_drink_per_time": {"t": t, "f": vol_drink_per_time},
    "kcal_liquid_per_vol": {"t": t, "f": kcal_liquid_per_vol},
    "drink_length": {"t": t, "f": drink_length},
    "kcal_solid": {"t": meal_times, "f": meal_kcals},
    }

# Plotting the drinks

sim_results = simulate(model, anthropometrics, stim, extra_time=extra_time)

st.subheader("Plotting the time course given the alcoholic drinks specified")
feature = st.selectbox("Feature of the model to plot", model_features)
st.line_chart(sim_results, x="Time", y=feature)
