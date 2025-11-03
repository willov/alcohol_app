import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import scipy.interpolate as interp

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

    with open("./results/alcohol_model (186.99).json",'r') as f:
        param_in = json.load(f)
        params = param_in['x']

    model.parameter_values = params
    features = list(model.feature_names)
    return model, features

model, model_features = setup_model('alcohol_model')
model_features = [feature for feature in model_features if feature not in ['Acetate in plasma', 'Gastric volume']]
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

def simulate_week(anthropometrics, drink_type, drink_grams_total, n_weeks = 1):
    if drink_type == "Beer":
        drink_conc = 5.1
        drink_volume = 0.33
        drink_kcal = 129.82*drink_volume
        drink_length = 20
    elif drink_type == "Wine":
        drink_conc = 12.0
        drink_volume = 0.15
        drink_kcal = 133.2526*drink_volume
        drink_length = 20
    elif drink_type == "Spirit":
        drink_conc = 40.0
        drink_volume = 0.04
        drink_kcal = 10
        drink_length = 2

    single_drink_grams = drink_conc*drink_volume*0.7891*10

    # Selecting how much grams to drink in a week, and then handling the selection

    drink_times = [[],[],[],[],[],[],[]]
    drink_lengths = [[],[],[],[],[],[],[]]
    drink_concentrations =[[],[],[],[],[],[],[]]
    drink_volumes = [[],[],[],[],[],[],[]]
    drink_kcals = [[],[],[],[],[],[],[]]

    start_time = 18.0
    drinks_total = drink_grams_total/single_drink_grams
    drinks_per_day = drinks_total/7

    for drink in range(0,int(np.ceil(drinks_per_day))+1):
        for day in range(5,-2,-1):
            if drinks_total >0:
                drink_times[day].append(start_time)         
                drink_lengths[day].append(drink_length)
                drink_concentrations[day].append(drink_conc)
                drink_volumes[day].append(min(1,drinks_total)*drink_volume)
                drink_kcals[day].append(drink_kcal)
                drinks_total-=1
        start_time+=1

    drink_times=[t+24*day for day,times in enumerate(drink_times) for t in times]
    drink_lengths = flatten(drink_lengths)
    drink_concentrations = flatten(drink_concentrations)
    drink_volumes = flatten(drink_volumes)
    drink_kcals = flatten(drink_kcals)

    drink_times = [t+24*7*week for week in range(0, n_weeks) for t in drink_times]
    drink_lengths = drink_lengths*n_weeks
    drink_concentrations = drink_concentrations*n_weeks
    drink_volumes = drink_volumes*n_weeks
    drink_kcals = drink_kcals*n_weeks

    EtOH_conc = [0]+[c*on for c in drink_concentrations for on in [1 , 0]]
    vol_drink_per_time = [0]+[v/t*on if t>0 else 0 for v,t in zip(drink_volumes, drink_lengths) for on in [1 , 0]]
    kcal_liquid_per_vol = [0]+[k/v*on if v>0 else 0 for v,k in zip(drink_volumes, drink_kcals) for on in [1 , 0]]
    drink_length = [0]+[t*on for t in drink_lengths for on in [1 , 0]]
    t = [t+(l/60)*on for t,l in zip(drink_times, drink_lengths) for on in [0,1]]

    meal_times = [7, 12, 18]

    if anthropometrics["sex"]==1:
        daily_kcal = 2500
    else:
        daily_kcal = 2000

    meal_times = [t+24*d+24*7*w for w in range(n_weeks) for d in range(0,7) for t in [7, 12, 18]]
    meal_lengths = 30.0/60
    meal_times = [t+meal_lengths*on for t in meal_times for on in [0,1]]

    meal_kcals = [0.2*daily_kcal, 0.4*daily_kcal, 0.4*daily_kcal]*7*n_weeks
    meal_kcals = [k*on for k in meal_kcals for on in [1 , 0]]

    stim = {
        "EtOH_conc": {"t": t, "f": EtOH_conc},
        "vol_drink_per_time": {"t": t, "f": vol_drink_per_time},
        "kcal_liquid_per_vol": {"t": t, "f": kcal_liquid_per_vol},
        "drink_length": {"t": t, "f": drink_length},
        "kcal_solid": {"t": meal_times, "f": [0]+meal_kcals},
        }

    sim_results = simulate(model, anthropometrics, stim, extra_time=12)
    return sim_results

# Start the app

st.markdown("# Evaluating reported PEth levels")

# Anthropometrics            
st.subheader("Anthropometrics")

# Shared variables between the pages
if 'sex' not in st.session_state:
    st.session_state['sex'] = 'Man'
if 'weight' not in st.session_state:
    st.session_state['weight'] = 104.0
if 'height' not in st.session_state:
    st.session_state['height'] = 1.85

anthropometrics = {"sex": st.session_state['sex'], "weight": st.session_state['weight'], "height": st.session_state['height']}
anthropometrics["sex"] = st.selectbox("Sex:", ["Man", "Woman"], ["Man", "Woman"].index(st.session_state['sex']), key="sex")
anthropometrics["weight"] = st.number_input("Weight (kg):", 0.0, 1000.0, st.session_state.weight, step=0.1, key="weight")
anthropometrics["height"] = st.number_input("Height (m):", 0.0, 2.5, st.session_state.height, key="height")

anthropometrics["sex"] = float(anthropometrics["sex"].lower() in ["male", "man", "men", "boy", "1", "chap", "guy"]) #Converts to a numerical representation

# Simulating long term PEth
st.divider()
st.subheader("Reported drinking")

PEth_unit = st.selectbox("PEth unit", ["ng/ml", "µM"], 0)

if PEth_unit == "µM":
    PEth_scaling = 703.0
else:
    PEth_scaling = 1

reported_PEth = st.number_input(f"Reported PEth levels ({PEth_unit})", 0.0, 1000000.0, 1000.0/PEth_scaling, 10.0)
drink_grams_total = st.number_input("Reported usage (gram of ethanol per week)", 0.0, 10000.0,195.0,10.0)
drink_type = st.selectbox("Type of drink (only minor differences)", ["Wine", "Beer", "Spirit"])

st.divider()
st.subheader("Simulating the reported weekly consumption")
with st.expander("About the simulation"):
    st.markdown(r"""
    The simulating starts with drinking one (partial) drink on the saturday. Once a full drink is drank on saturday the next drink starts on the Friday, the next on Thursday, Wednesday, Tuesday, Monday, and finally Sunday. 
    Then, the simulates starts on a second drink per day in the same order as for the first. 

    The simulation also assumes that the person eats a total amount of 2000/2500 kcal per day, spread as 20\% breakfast at 7:00, 40\% as lunch at 12:00, and 40% as dinner at 18:00. 
    """)

n_weeks = st.slider("Number of weeks to simulate", 1,12,4)

plot_timeseries = st.checkbox("Plot the simulated drinking pattern for the reported weekly usage")
if plot_timeseries:
    sim_results = simulate_week(anthropometrics, drink_type, drink_grams_total, n_weeks)
    feature = st.selectbox("Feature of the model to plot", model_features,len(model_features)-1, key="Feature of the model to plot_2")

    sim_results["Time"] = sim_results["Time"]/24
    sim_results.rename(columns = {"Time":"Time (days)"}, inplace = True)
    st.line_chart(sim_results, x="Time (days)", y=feature, width=True)

max_dose = st.number_input("Maximum weekly consumption to simulate (g/week)", 0.0, 10000.0, 1000.0, 100.0)
num_points = st.slider("Number of weekly doses to simulate (more is slower)",10,1000,10,1)
max_sim_dose = max(max_dose,drink_grams_total)
dose_response = {"Ethanol/week (gram)" : np.linspace(1,max_sim_dose,num_points)}
dose_response["Simulated PEth"] = []

with open('results/PEth_uncertainty.json','r') as f:
    PEth_params = json.load(f)
model.parameter_values = PEth_params["low"]
dose_response["Simulated PEth (low)"] = []
for gram in dose_response["Ethanol/week (gram)"]:
        sim_results = simulate_week(anthropometrics, "Beer", gram, n_weeks)
        dose_response["Simulated PEth (low)"].append(sim_results["PEth (ng/mL)"].values[-1]/PEth_scaling)

model.parameter_values = PEth_params["high"]
dose_response["Simulated PEth (high)"] = []
for gram in dose_response["Ethanol/week (gram)"]:
        sim_results = simulate_week(anthropometrics, "Beer", gram, n_weeks)
        dose_response["Simulated PEth (high)"].append(sim_results["PEth (ng/mL)"].values[-1]/PEth_scaling)

fig = go.Figure()
fig.add_trace(go.Scatter(name="Simulated PEth", x = dose_response["Ethanol/week (gram)"], y=dose_response["Simulated PEth (low)"], showlegend=False, mode='lines', line=dict(width=2)))
fig.add_trace(go.Scatter(name="Simulated PEth", x = dose_response["Ethanol/week (gram)"], y=dose_response["Simulated PEth (high)"], fill='tonexty', showlegend=True, mode='none', line=dict(color="#636EFA")))
fig.add_trace(go.Scatter(name="Reported PEth", x=[drink_grams_total], y=[reported_PEth], marker=dict(color="rgb(255, 191, 0)", size=10)))
fig.add_trace(go.Scatter(name="Reported PEth", x=[0, max_sim_dose], y=[reported_PEth,reported_PEth], showlegend=False, mode='lines', line=dict(width=2, color="rgb(255, 191, 0)")))

fig.update_layout(
    xaxis_title="Ethanol/week (gram)", yaxis_title=f"PEth ({PEth_unit})", 
    legend=dict(orientation="h", xanchor="center", y=-0.2, x=0.5),
    margin=dict(l=0, r=0, t=0, b=0)
)
st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

#fig.write_image("PEth-prediction.svg")
weekly_dose_interp_low = interp.interp1d(dose_response["Simulated PEth (low)"], dose_response["Ethanol/week (gram)"])
weekly_dose_interp_high = interp.interp1d(dose_response["Simulated PEth (high)"], dose_response["Ethanol/week (gram)"])

interpolated_weekly_dose_low = weekly_dose_interp_low(reported_PEth)
interpolated_weekly_dose_high = weekly_dose_interp_high(reported_PEth)

st.markdown(f"""The interpolated weekly dose of ethanol based on reported PEth levels and anthropometrics is in the range {interpolated_weekly_dose_high:.2f} – {interpolated_weekly_dose_low:.2f} g/week.
This is {interpolated_weekly_dose_high-drink_grams_total:.2f} – {interpolated_weekly_dose_low-drink_grams_total:.2f} g/week from the reported value of {drink_grams_total} g/week.""")