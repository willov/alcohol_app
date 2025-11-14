import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import scipy.interpolate as interp

from sidebar_config import setup_sidebar
from functions.ui_helpers import (
    setup_sund_package, setup_model, simulate_week, 
    get_anthropometrics_ui
)

# Setup sund and sidebar
sund = setup_sund_package()
setup_sidebar()

# Setup the model
model, model_features = setup_model('alcohol_model')
model_features = [feature for feature in model_features if feature not in ['Acetate in plasma', 'Gastric volume']]

# Start the app

st.markdown("# Evaluating reported PEth levels")

# Anthropometrics            
st.subheader("Anthropometrics")

anthropometrics = get_anthropometrics_ui(defaults={"sex": "Man", "weight": 104.0, "height": 1.85, "age": None})

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
    sim_results = simulate_week(model, anthropometrics, drink_type, drink_grams_total, n_weeks)
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
        sim_results = simulate_week(model, anthropometrics, "Beer", gram, n_weeks)
        dose_response["Simulated PEth (low)"].append(sim_results["PEth (ng/mL)"].values[-1]/PEth_scaling)

model.parameter_values = PEth_params["high"]
dose_response["Simulated PEth (high)"] = []
for gram in dose_response["Ethanol/week (gram)"]:
        sim_results = simulate_week(model, anthropometrics, "Beer", gram, n_weeks)
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