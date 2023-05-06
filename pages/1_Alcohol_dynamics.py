import os
import json
import pandas as pd
import numpy as np
import streamlit as st

# Install sund in a custom location
import subprocess
import sys
if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", 'https://isbgroup.eu/edu/assets/sund-1.0.1.tar.gz#sha256=669a1d05c5c8b68500086e183d831650277012b3ea57e94356de1987b6e94e3e'])

sys.path.append('./custom_package')
import sund

st.elements.utils._shown_default_value_warning=True # This is not a good solution, but it hides the warning of using default values and sessionstate api

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
    features = model.featurenames
    return model, features

model, model_features = setup_model('alcohol_model')

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

# Start the app

st.title("Simulation of alcohol dynamics")
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

n_drinks = st.slider("Number of drinks:", 1, 15, 1)
extra_time = st.number_input("Additional time to simulate after last drink (h):", 0.0, 100.0, 12.0, 0.1)

drink_times = []
drink_lengths = []
drink_concentrations = []
drink_volumes = []
drink_kcals = []

st.divider()
start_time = 18.0
for i in range(n_drinks):
    st.markdown(f"**Drink {i+1}**")

    drink_times.append(st.number_input("Time of drink (h): ", 0.0, 100.0, start_time, 0.1, key=f"drink_time{i}"))
    drink_lengths.append(st.number_input("Drink length (min): ", 0.0, 240.0, 20.0, 0.1, key=f"drink_length{i}"))
    drink_concentrations.append(st.number_input("Concentration of drink (%): ", 0.0, 100.0, 5.0, 0.01, key=f"drink_concentrations{i}"))
    drink_volumes.append(st.number_input("Volume of drink (L): ", 0.0, 24.0, 0.33, 0.1, key=f"drink_volumes{i}"))
    drink_kcals.append(st.number_input("Kcal of the drink (kcal): ", 0.0, 1000.0, 45.0, 1.0, key=f"drink_kcals{i}"))
    start_time += 1
    st.divider()

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

n_meals = st.slider("Number of (solid) meals:", 0, 15, 1)

for i in range(n_meals):
    st.markdown(f"**Meal {i+1}**")

    meal_times.append(st.number_input("Time of the meal (h): ", 0.0, 100.0, start_time, 0.1, key=f"meal_time{i}"))
    meal_kcals.append(st.number_input("Kcal of the meal (kcal): ", 0.0, 10000.0, 500.0, 1.0, key=f"meal_kcals{i}"))
    start_time += 6
    st.divider()

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
