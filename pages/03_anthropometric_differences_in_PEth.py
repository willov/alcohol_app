import numpy as np
import streamlit as st
import plotly.graph_objects as go

from sidebar_config import setup_sidebar
from functions.ui_helpers import (
    setup_sund_package, setup_model, simulate_week
)

# Setup sund and sidebar
sund = setup_sund_package()
setup_sidebar()

# Setup the model
model, model_features = setup_model('alcohol_model')
model_features = [feature for feature in model_features if feature not in ['Acetate in plasma', 'Gastric volume']]

# Start the app

st.markdown("# Anthropometric differences in PEth")

# Simulating anthropometric differences
st.subheader("Simulating differences based on anthropometrics")
with st.expander("About the simulation"):
    st.markdown(r"""
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
        anthropometrics = {"sex": 1, "weight": 65.0, "height": 1.7}
        sim_results = simulate_week(model, anthropometrics, drink_type, gram, n_weeks)
        dose_response["PEth (small man)"].append(sim_results["PEth (ng/mL)"].values[-1])
    if sim_large_man:
        anthropometrics = {"sex": 1, "weight": 105.0, "height": 1.9}
        sim_results = simulate_week(model, anthropometrics, drink_type, gram, n_weeks)
        dose_response["PEth (large man)"].append(sim_results["PEth (ng/mL)"].values[-1])
    if sim_small_woman:
        anthropometrics = {"sex": 0, "weight": 50.0, "height": 1.5}
        sim_results = simulate_week(model, anthropometrics, drink_type, gram, n_weeks)
        dose_response["PEth (small woman)"].append(sim_results["PEth (ng/mL)"].values[-1])
    if sim_large_woman:
        anthropometrics = {"sex": 0, "weight": 80.0, "height": 1.75}
        sim_results = simulate_week(model, anthropometrics, drink_type, gram, n_weeks)
        dose_response["PEth (large woman)"].append(sim_results["PEth (ng/mL)"].values[-1])


fig = go.Figure()
if sim_small_man:
    fig.add_trace(go.Scatter(name="Small man", x = dose_response["Ethanol/week (gram)"], y=dose_response["PEth (small man)"], mode='lines', line=dict(width=2)))
if sim_large_man:
    fig.add_trace(go.Scatter(name="Large man", x = dose_response["Ethanol/week (gram)"], y=dose_response["PEth (large man)"], mode='lines', line=dict(width=2)))
if sim_small_woman:
    fig.add_trace(go.Scatter(name="Small woman", x = dose_response["Ethanol/week (gram)"], y=dose_response["PEth (small woman)"], mode='lines', line=dict(width=2)))
if sim_large_woman:
    fig.add_trace(go.Scatter(name="Large woman", x = dose_response["Ethanol/week (gram)"], y=dose_response["PEth (large woman)"], mode='lines', line=dict(width=2)))

fig.update_layout(
    xaxis_title="Ethanol/week (gram)", yaxis_title="Simulated PEth (ng/ml)", 
    legend=dict(orientation="h", xanchor="center", y=-0.2, x=0.5),
    margin=dict(l=0, r=0, t=0, b=0)
)
st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
