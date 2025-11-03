import numpy as np
import streamlit as st
import plotly.graph_objects as go

from sidebar_config import setup_sidebar
from functions.ui_helpers import (
    setup_sund_package, setup_model, simulate_week, 
    get_drink_specs, build_stimulus_dict, get_anthropometrics_ui
)

# Setup sund and sidebar
sund = setup_sund_package()
setup_sidebar()

# Setup the model
model, model_features = setup_model('alcohol_model')
model_features = [feature for feature in model_features if feature not in ['Acetate in plasma', 'Gastric volume']]

# Start the app

st.markdown("# Drink type impact on PEth")
st.subheader("Anthropometrics")

anthropometrics = get_anthropometrics_ui(defaults={"sex": "Man", "weight": 104.0, "height": 1.85})

# Simulating anthropometric differences
st.subheader("Simulating differences based on drink")
with st.expander("About the simulation"):
    st.markdown(r"""
    The simulating starts with drinking one (partial) drink on the saturday. Once a full drink is drank on saturday the next drink starts on the Friday, the next on Thursday, Wednesday, Tuesday, Monday, and finally Sunday. 
    Then, the simulates starts on a second drink per day in the same order as for the first. 
    The drinks are assumed to have the following properties: 
    
    * Beer: 5.1\% alcohol, 378.77 kcal/L, drunk in 20 minutes, max volume 33 cl before starting the next drink. 
    * Wine: 12\% alcohol, 744.51 kcal/L, drunk in 20 minutes, max volume 15 cl before starting the next drink. 
    * Spirit: 40\% alcohol, 10 kcal/L, drunk in 2 minutes, max volume 4 cl before starting the next drink. 

    
    The simulation also assumes that the person eats a total amount of 2000/2500 kcal per day, spread as 20\% breakfast at 7:00, 40\% as lunch at 12:00, and 40% as dinner at 18:00. """)

n_weeks = st.slider("Number of weeks to simulate the drinking pattern for", 1,12,4,key="n_weeks_anthropometrics")

num_points = st.slider("Number of points to plot on the curve (more takes longer)",10,1000,10,1,key='num_points_anthropometrics')

dose_response = {"Ethanol/week (gram)" : np.linspace(1,1000,num_points), "Beer":[], "Wine":[], "Spirit":[]}

for gram in dose_response["Ethanol/week (gram)"]:
        sim_results = simulate_week(model, anthropometrics, "Beer", gram, n_weeks)
        dose_response["Beer"].append(sim_results["PEth (ng/mL)"].values[-1])
        sim_results = simulate_week(model, anthropometrics, "Wine", gram, n_weeks)
        dose_response["Wine"].append(sim_results["PEth (ng/mL)"].values[-1])
        sim_results = simulate_week(model, anthropometrics, "Spirit", gram, n_weeks)
        dose_response["Spirit"].append(sim_results["PEth (ng/mL)"].values[-1])

fig = go.Figure()
fig.add_trace(go.Scatter(name="Beer", x = dose_response["Ethanol/week (gram)"], y=dose_response["Beer"], mode='lines', line=dict(width=2)))
fig.add_trace(go.Scatter(name="Spirit", x = dose_response["Ethanol/week (gram)"], y=dose_response["Spirit"], mode='lines', line=dict(width=2)))
fig.add_trace(go.Scatter(name="Wine", x = dose_response["Ethanol/week (gram)"], y=dose_response["Wine"], mode='lines', line=dict(width=2)))

fig.update_layout(
    xaxis_title="Ethanol/week (gram)", yaxis_title="Simulated PEth (ng/ml)", 
    legend=dict(orientation="h", xanchor="center", y=-0.2, x=0.5),
    margin=dict(l=0, r=0, t=0, b=0)
)
st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
