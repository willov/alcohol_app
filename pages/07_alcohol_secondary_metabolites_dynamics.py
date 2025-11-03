import streamlit as st

from sidebar_config import setup_sidebar
from functions.ui_helpers import (
    setup_sund_package, setup_model, simulate,
    seed_new_items, on_change_time_propagate, lock_all,
    draw_drink_timeline_plotly, enforce_minimum_time,
    init_anthropometrics, build_stimulus_dict
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

# Initialize anthropometrics with helper
anthropometrics = init_anthropometrics(defaults={"sex": "Man", "weight": 70.0, "height": 1.72, "age": 30})
anthropometrics["sex"] = st.selectbox("Sex:", ["Man", "Woman"], ["Man", "Woman"].index(st.session_state['sex']), key="sex")
anthropometrics["weight"] = st.number_input("Weight (kg):", 0.0, 200.0, st.session_state.weight, 1.0, key="weight")
anthropometrics["height"] = st.number_input("Height (m):", 0.0, 2.5, st.session_state.height, key="height")
anthropometrics["age"] = st.number_input("Age (years):", 0, 120, st.session_state.age, key="age")

anthropometrics["sex"] = float(anthropometrics["sex"].lower() in ["male", "man", "men", "boy", "1", "chap", "guy"]) # Converts to a numerical representation

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
    enforce_minimum_time(page="07", what="drink", index=index, n=st.session_state.get("n_drinks_07", 1), min_gap=None)
    on_change_time_propagate(page="07", what="drink", index=index, n=st.session_state.get("n_drinks_07", 1), step=1.0)

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
    lock_label = "Lock 🔒" if st.session_state.get(lock_key, False) else "Lock"
    st.checkbox(lock_label, key=lock_key, help="Prevent auto-fill changes to this drink time")
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
    enforce_minimum_time(page="07", what="meal", index=index, n=st.session_state.get("n_meals_07", 0), min_gap=10.0/60.0)  # 10 minutes in hours
    on_change_time_propagate(page="07", what="meal", index=index, n=st.session_state.get("n_meals_07", 0), step=6.0)

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
    lock_label = "Lock 🔒" if st.session_state.get(lock_key, False) else "Lock"
    st.checkbox(lock_label, key=lock_key, help="Prevent auto-fill changes to this meal time")
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
feature = st.selectbox("Feature of the model to plot", model_features)

# Try to render an interactive Plotly timeline with drink duration rectangles. Fall back to line_chart if plotly
try:
    fig = draw_drink_timeline_plotly(sim_results, feature, drink_times, drink_lengths, title=f"{feature} over time")
    st.plotly_chart(fig, use_container_width=True, key=f"plot_07_{feature}")
except Exception:
    st.line_chart(sim_results, x="Time", y=feature)
