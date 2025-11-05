import streamlit as st

from sidebar_config import setup_sidebar
from functions.ui_helpers import (
    setup_sund_package, setup_model, simulate,
    seed_new_items, on_change_time_propagate, lock_all,
    enforce_minimum_time, build_stimulus_dict, 
    create_multi_feature_plot, get_anthropometrics_ui
)

# Setup sund and sidebar
sund = setup_sund_package()
setup_sidebar()

# Setup the model
model, model_features = setup_model('alcohol_model')

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

anthropometrics = get_anthropometrics_ui(defaults={"sex": "Man", "weight": 70.0, "height": 1.72, "age": None})

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
    # Enforce that this drink's time is not before the previous drink's time + previous drink's duration
    enforce_minimum_time(page="02", what="drink", index=index, n=st.session_state.get("n_drinks_02", 1), min_gap=None)
    # Propagate changes to subsequent unlocked drinks
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
    st.checkbox("Lock", key=lock_key, help="Prevent auto-fill changes to this drink time")
    with col2:
        drink_lengths.append(st.number_input("Length (min)", 0.0, 240.0, 20.0, 1.0, key=f"drink_length{i}"))
    with col3:
        drink_concentrations.append(st.number_input("ABV (%)", 0.0, 100.0, 5.0, 0.1, key=f"drink_concentrations{i}"))
    with col4:
        drink_volumes.append(st.number_input("Vol (L)", 0.0, 24.0, 0.33, 0.1, key=f"drink_volumes{i}"))
    with col5:
        drink_kcals.append(st.number_input("kcal", 0.0, 1000.0, 45.0, 1.0, key=f"drink_kcals{i}"))
    start_time += 1

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
    # Enforce that this meal's time is not before the previous meal's time + 10 minutes
    enforce_minimum_time(page="02", what="meal", index=index, n=st.session_state.get("n_meals_02", 0), min_gap=10.0/60.0)  # 10 minutes in hours
    # Propagate changes to subsequent unlocked meals
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
    st.checkbox("Lock", key=lock_key, help="Prevent auto-fill changes to this meal time")
    with col2:
        meal_kcals.append(st.number_input("kcal", 0.0, 10000.0, 500.0, 1.0, key=f"meal_kcals{i}"))
    start_time += 6

if n_meals < 1.0:
    st.divider()

meal_times = [t+(30/60)*on for t in meal_times for on in [0,1]]
meal_kcals = [0]+[m*on for m in meal_kcals for on in [1 , 0]]

# Setup stimulation to the model using helper
stim = build_stimulus_dict(
    drink_times, drink_lengths, drink_concentrations, 
    drink_volumes, drink_kcals, meal_times, meal_kcals
)

# Plotting the drinks

sim_results = simulate(model, anthropometrics, stim, extra_time=extra_time)

st.subheader("Plotting the time course given the alcoholic drinks specified")
selected_features = st.multiselect("Features of the model to plot", model_features, default=[model_features[0]] if model_features else [])

if selected_features:
    fig = create_multi_feature_plot(sim_results, selected_features, drink_starts=drink_times, drink_lengths=drink_lengths)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key=f"plot_02_multi")
else:
    st.info("👆 Select at least one feature to plot.")
