import streamlit as st

from sidebar_config import setup_sidebar
from functions.ui_helpers import (
    setup_sund_package, setup_model, simulate,
    seed_new_items, on_change_duration_validate_next,
    lock_all, enforce_minimum_time, build_stimulus_dict,
    create_multi_feature_plot, get_anthropometrics_ui
)

# Setup sund and sidebar
sund = setup_sund_package()
setup_sidebar()

# Setup the model
model, model_features = setup_model('alcohol_model_2')

# Start the app

st.markdown("# Secondary alcohol metabolites")
st.write("Simulation of secondary alcohol metabolites")
st.markdown("""We have constructed a more detailed mechanistic model that involves secondary metabolites, ethyl glucuronide (EtG), ethyl sulphate (EtS), and urine alcohol concentration (UAC), able to both explain the profiles of these metabolites following consumption of alcohol. 

Below, you can specify one or more alcoholic drinks, and some anthropometrics to get simulations of the concentration of these metabolites in plasma.
""")

# Anthropometrics            
st.subheader("Anthropometrics")

anthropometrics = get_anthropometrics_ui(defaults={"sex": "Man", "weight": 70.0, "height": 1.72, "age": 30})

# Specifying the drinks
st.subheader("Specifying the alcoholic drinks")

def _on_change_n_drinks_07():
    seed_new_items(
        page="07", name="drinks", n=st.session_state.get("n_drinks_07", 1), default_start=18.0, step=1.0,
        seed_key_template="{prefix}_time_{page}_{i}", lock_key_template="{prefix}_time_locked_{page}_{i}", key_prefix="drink"
    )


n_drinks = st.slider("Number of drinks:", 1, 15, 1, key="n_drinks_07", on_change=_on_change_n_drinks_07)
extra_time = st.number_input("Additional time to simulate after last drink (h):", 0.0, 100.0, 12.0, 0.1)

# Lock all / Unlock all controls for drinks
lock_col_a, lock_col_b = st.columns(2)
if lock_col_a.button("Lock all drinks", key="lock_all_drinks_07"):
    lock_all(page="07", what="drink", n=n_drinks, locked=True)
if lock_col_b.button("Unlock all drinks", key="unlock_all_drinks_07"):
    lock_all(page="07", what="drink", n=n_drinks, locked=False)

drink_times = []
drink_lengths = []
drink_concentrations = []
drink_volumes = []
drink_kcals = []

st.divider()
start_time = 18.0

def _on_change_drink_time_07(index):
    enforce_minimum_time(page="07", what="drink", index=index, min_gap=None)
    # Validate that all subsequent drinks still respect their constraints
    # Only adjust if they conflict, don't propagate arbitrary time changes
    n_drinks = st.session_state.get("n_drinks_07", 1)
    for j in range(index + 1, n_drinks):
        enforce_minimum_time(page="07", what="drink", index=j, min_gap=None)

def _on_change_drink_length_07(index):
    on_change_duration_validate_next(page="07", what="drink", index=index, n=st.session_state.get("n_drinks_07", 1), min_gap=None)
    # After updating the next drink if needed, also validate all subsequent drinks
    n_drinks = st.session_state.get("n_drinks_07", 1)
    for j in range(index + 2, n_drinks):
        enforce_minimum_time(page="07", what="drink", index=j, min_gap=None)

# Initialize defaults and locks
for i in range(n_drinks):
    key_time = f"drink_time_07_{i}"
    lock_key = f"drink_time_locked_07_{i}"
    if key_time not in st.session_state:
        st.session_state[key_time] = start_time + i * 1.0
    if lock_key not in st.session_state:
        st.session_state[lock_key] = False

for i in range(n_drinks):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        drink_times.append(st.number_input("Time (h)", 0.0, 100.0, key=f"drink_time_07_{i}", on_change=_on_change_drink_time_07, args=(i,)))
    lock_key = f"drink_time_locked_07_{i}"
    st.checkbox("Lock", key=lock_key, help="Prevent auto-fill changes to this drink time")
    with col2:
        drink_lengths.append(st.number_input("Length (min)", 0.0, 240.0, 20.0, 1.0, key=f"drink_length{i}", on_change=_on_change_drink_length_07, args=(i,)))
    with col3:
        drink_concentrations.append(st.number_input("ABV (%)", 0.0, 100.0, 5.0, 0.1, key=f"drink_concentrations{i}"))
    with col4:
        drink_volumes.append(st.number_input("Vol (L)", 0.0, 2.0, 0.33, 0.1, key=f"drink_volumes{i}"))
    with col5:
        drink_kcals.append(st.number_input("kcal/L", 0.0, 600.0, 45.0, 10.0, key=f"drink_kcals{i}"))
    start_time += 1

# Setup meals
st.subheader(f"Specifying the meals")

start_time = 12.0

meal_times = []
meal_kcals = []

def _on_change_n_meals_07():
    seed_new_items(
        page="07", name="meals", n=st.session_state.get("n_meals_07", 0), default_start=12.0, step=6.0,
        seed_key_template="{prefix}_time_{page}_{i}", lock_key_template="{prefix}_time_locked_{page}_{i}", key_prefix="meal"
    )

n_meals = st.slider("Number of (solid) meals:", 0, 15, 1, key="n_meals_07", on_change=_on_change_n_meals_07)

# Lock all / Unlock all controls for meals
lockm_a, lockm_b = st.columns(2)
if lockm_a.button("Lock all meals", key="lock_all_meals_07"):
    lock_all(page="07", what="meal", n=n_meals, locked=True)
if lockm_b.button("Unlock all meals", key="unlock_all_meals_07"):
    lock_all(page="07", what="meal", n=n_meals, locked=False)

st.divider()

def _on_change_meal_time_07(index):
    # Enforce that this meal's time is not before the previous meal's time + 10 minutes
    enforce_minimum_time(page="07", what="meal", index=index, min_gap=10.0/60.0)  # 10 minutes in hours
    # Validate that all subsequent meals still respect their constraints
    # Only adjust if they conflict, don't propagate arbitrary time changes
    n_meals = st.session_state.get("n_meals_07", 0)
    for j in range(index + 1, n_meals):
        enforce_minimum_time(page="07", what="meal", index=j, min_gap=10.0/60.0)

# Initialize meal defaults and locks
for i in range(n_meals):
    key_time = f"meal_time_07_{i}"
    lock_key = f"meal_time_locked_07_{i}"
    if key_time not in st.session_state:
        st.session_state[key_time] = start_time + i * 6.0
    if lock_key not in st.session_state:
        st.session_state[lock_key] = False

for i in range(n_meals):
    col1, col2 = st.columns(2)
    with col1:
        meal_times.append(st.number_input("Time (h)", 0.0, 100.0, key=f"meal_time_07_{i}", on_change=_on_change_meal_time_07, args=(i,)))
    lock_key = f"meal_time_locked_07_{i}"
    st.checkbox("Lock", key=lock_key, help="Prevent auto-fill changes to this meal time")
    with col2:
        meal_kcals.append(st.number_input("kcal", 0.0, 10000.0, 500.0, 25.0, key=f"meal_kcals{i}"))
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
selected_features = st.multiselect("Features of the model to plot", model_features, default=[model_features[0]] if model_features else [], key="plot_features_07")

if selected_features:
    fig = create_multi_feature_plot(sim_results, selected_features, drink_starts=drink_times, drink_lengths=drink_lengths)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key=f"plot_07_multi")
else:
    st.info("👆 Select at least one feature to plot.")
