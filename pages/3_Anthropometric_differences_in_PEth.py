import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# Install sund in a custom location
import subprocess
import sys
if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", 'https://isbgroup.eu/edu/assets/sund-1.0.1.tar.gz#sha256=669a1d05c5c8b68500086e183d831650277012b3ea57e94356de1987b6e94e3e'])

sys.path.append('./custom_package')
import sund

# Setup the models

def setup_model(model_name):
    sund.installModel(f"./models/{model_name}.txt")
    model_class = sund.importModel(model_name)
    model = model_class() 

    fs = []
    for path, subdirs, files in os.walk('./results'):
        for name in files:
            if model_name in name.split('(')[0] and "ignore" not in path:
                fs.append(os.path.join(path, name))
    fs.sort()
    with open(fs[0],'r') as f:
        param_in = json.load(f)
        params = param_in['x']

    model.parametervalues = params
    features = list(model.featurenames)
    return model, features

model, model_features = setup_model('alcohol_model')
model_features = [feature for feature in model_features if feature not in ['Acetate in plasma', 'Gastric volume']]
# Define functions needed

def flatten(list):
    return [item for sublist in list for item in sublist]

def simulate(m, anthropometrics, stim, extra_time = 10):
    act = sund.Activity(timeunit = 'h')
    pwc = sund.PIECEWISE_CONSTANT # space saving only
    const = sund.CONSTANT # space saving only

    for key,val in stim.items():
        act.AddOutput(name = key, type=pwc, tvalues = val["t"], fvalues = val["f"]) 
    for key,val in anthropometrics.items():
        act.AddOutput(name = key, type=const, fvalues = val) 
    
    sim = sund.Simulation(models = m, activities = act, timeunit = 'h')
    
    sim.ResetStatesDerivatives()
    t_start = min(stim["EtOH_conc"]["t"]+stim["kcal_solid"]["t"])-0.25

    sim.Simulate(timevector = np.linspace(t_start, max(stim["EtOH_conc"]["t"])+extra_time, 10000))
    
    sim_results = pd.DataFrame(sim.featuredata,columns=sim.featurenames)
    sim_results.insert(0, 'Time', sim.timevector)

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

st.title("Anthropometric differences on long-term PEth levels")
            
# Simulating anthropometric differences
st.subheader("Simulating differences based on anthropometrics")
with st.expander("About the simulation"):
    st.markdown("""
    The simulating starts with drinking one (partial) drink on the saturday. Once a full drink is drank on saturday the next drink starts on the Friday, the next on Thursday, Wednesday, Tuesday, Monday, and finally Sunday. 
    Then, the simulates starts on a second drink per day in the same order as for the first. 
    The drinks are assumed to have the following properties: 
    
    * Beer: 5.1\% alcohol, 378.77 kcal/L, drunk in 20 minutes, max volume 33 cl before starting the next drink. 
    * Wine: 12\% alcohol, 744.51 kcal/L, drunk in 20 minutes, max volume 15 cl before starting the next drink. 
    * Spirit: 40\% alcohol, 10 kcal/L, drunk in 2 minutes, max volume 4 cl before starting the next drink. 

    The simulation also assumes that the person eats a total amount of 2000/2500 kcal per day, spread as 20\% breakfast at 7:00, 40\% as lunch at 12:00, and 40% as dinner at 18:00. """)

drink_type = st.selectbox("Type of drink (only minor differences)", ["Wine", "Beer", "Spirit"])

n_weeks = st.slider("Number of weeks to simulate the drinking pattern for", 1,12,4,key="n_weeks_anthropometrics")

num_points = st.slider("Number of points to plot on the curve (more takes longer)",10,1000,10,1,key='num_points_anthropometrics')

sim_small_man = st.checkbox("Simulate a small man (65 kg, 170 cm)",False)
sim_large_man = st.checkbox("Simulate a large man (105 kg, 190 cm)",True)
sim_small_woman = st.checkbox("Simulate a small woman (50 kg, 140 cm)",True)
sim_large_woman = st.checkbox("Simulate a large woman (80 kg, 175 cm)",False)

dose_response = {"Ethanol/week (gram)" : np.linspace(1,1000,num_points)}
if sim_small_man:
    dose_response["PEth (small man)"] = []
if sim_large_man:
    dose_response["PEth (large man)"] = []
if sim_small_woman:
    dose_response["PEth (small woman)"] = []
if sim_large_woman:
    dose_response["PEth (large woman)"] = []

for gram in dose_response["Ethanol/week (gram)"]:
    if sim_small_man:
        anthropometrics_small_man = anthropometrics = {"sex": 1, "weight": 65.0, "height": 1.7}
        sim_results = simulate_week(anthropometrics_small_man, drink_type, gram, n_weeks)
        dose_response["PEth (small man)"].append(sim_results["PEth (ng/mL)"].values[-1])
    if sim_large_man:
        anthropometrics_large_man = anthropometrics = {"sex": 1, "weight": 105.0, "height": 1.9}
        sim_results = simulate_week(anthropometrics_large_man, drink_type, gram, n_weeks)
        dose_response["PEth (large man)"].append(sim_results["PEth (ng/mL)"].values[-1])
    if sim_small_woman:
        anthropometrics_small_woman = anthropometrics = {"sex": 0, "weight": 50.0, "height": 1.5}
        sim_results = simulate_week(anthropometrics_small_woman, drink_type, gram, n_weeks)
        dose_response["PEth (small woman)"].append(sim_results["PEth (ng/mL)"].values[-1])
    if sim_large_woman:
        anthropometrics_large_woman = anthropometrics = {"sex": 0, "weight": 80.0, "height": 1.75}
        sim_results = simulate_week(anthropometrics_large_woman, drink_type, gram, n_weeks)
        dose_response["PEth (large woman)"].append(sim_results["PEth (ng/mL)"].values[-1])


fig = go.Figure()
if sim_small_man:
    fig.add_trace(go.Scatter(name="Small man", x = dose_response["Ethanol/week (gram)"], y=dose_response["PEth (small man)"], mode='lines', marker={"line": {"width":0}}))
if sim_large_man:
    fig.add_trace(go.Scatter(name="Large man", x = dose_response["Ethanol/week (gram)"], y=dose_response["PEth (large man)"], mode='lines', marker={"line": {"width":0}}))
if sim_small_woman:
    fig.add_trace(go.Scatter(name="Small woman", x = dose_response["Ethanol/week (gram)"], y=dose_response["PEth (small woman)"], mode='lines', marker={"line": {"width":0}}))
if sim_large_woman:
    fig.add_trace(go.Scatter(name="Large woman", x = dose_response["Ethanol/week (gram)"], y=dose_response["PEth (large woman)"], mode='lines', marker={"line": {"width":0}}))

fig.update_layout(xaxis_title="Ethanol/week (gram)", yaxis_title="Simulated PEth (ng/ml)", 
                        legend=dict(orientation="h", xanchor="center", y=-0.2, x=0.5),
                        margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)
