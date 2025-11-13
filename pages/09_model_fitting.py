import streamlit as st
import numpy as np
import pandas as pd

from sidebar_config import setup_sidebar
from functions.ui_helpers import (
    setup_sund_package, setup_model, simulate,
    build_stimulus_dict,
    create_multi_feature_plot, get_anthropometrics_ui,
    drink_selector_cards
)

# Setup sund and sidebar
sund = setup_sund_package()
setup_sidebar()

# Setup the model
model, model_features = setup_model('alcohol_model_2')

# Feature units mapping
feature_units = {
    "Blood alcohol concentration (mg/dL)": "mg/dL",
    "Urine alcohol concentration (mg/dL)": "mg/dL",
    "Ethyl glucuronide (mg/dL)": "mg/dL",
    "Ethyl sulphate (mg/dL)": "mg/dL"
}

st.markdown("""
# Model Fitting Challenge: Evaluating Drinking Patterns

In this demonstration, you take the role of a scientist trying to find a drinking pattern that is consistent with measured biomarker data.

The interactive tool allows you to enter custom data, and challenges you to find a drinking pattern that matches measured data.

1. **Input your data points** (Time and measured value)
2. **Specify anthropometrics** for your subject
3. **Design a drinking pattern** to simulate
4. **Run the simulation** to see if it matches the data

The simulation is considered successful if all data points are within the chosen tolerance (default ±5%) of the predicted values. 

Can you find a drinking pattern that fits the data?
""")

st.divider()

# === DATA INPUT SECTION ===
st.header("Defining measured data")
st.markdown("Enter measured data points (time in hours, measured value). You can add data for **BAC, UAC, EtG, and/or EtS** - all are optional.")

# Feature selection - allow multiple features
available_features = [f for f in model_features if f in [
    "Blood alcohol concentration (mg/dL)", 
    "Urine alcohol concentration (mg/dL)", 
    "Ethyl glucuronide (mg/dL)", 
    "Ethyl sulphate (mg/dL)"
    ]
]
selected_features = st.multiselect(
    "Select measured features:", 
    available_features, 
    default=[available_features[0]] if available_features else [],
    key="selected_features"
)

if not selected_features:
    st.warning("Please select at least one feature to measure.")
    selected_features = []

# Data input method
data_input_method = st.radio(
    "How would you like to input data?", 
    ["Manual Entry", "Paste CSV"], 
    horizontal=True
)

# Initialize data_points from session state or as empty
data_points = {feature: [] for feature in selected_features}

if data_input_method == "Manual Entry":
    st.markdown("#### Manual Data Entry")
    
    # Initialize session state for data points if not already done
    if 'num_data_points' not in st.session_state:
        st.session_state['num_data_points'] = 3
    
    num_points = st.number_input("Number of data points:", 1, 20, st.session_state['num_data_points'], key="num_data_points_input")
    st.session_state['num_data_points'] = num_points
    
    # Create input fields for each data point and each feature
    for feature in selected_features:
        st.markdown(f"##### Data for {feature}")
        
        # Create input fields for each data point
        data_cols = st.columns([1, 2, 2, 0.5])
        with data_cols[0]:
            st.markdown("**#**")
        with data_cols[1]:
            st.markdown("**Time (h)**")
        with data_cols[2]:
            unit = feature_units.get(feature, "")
            st.markdown(f"**{feature} Value ({unit})**")
        
        # Set default values based on feature
        feature_defaults = {
            "Blood alcohol concentration (mg/dL)": 50.0,
            "Urine alcohol concentration (mg/dL)": 70.0,
            "Ethyl glucuronide (mg/dL)": 0.1,
            "Ethyl sulphate (mg/dL)": 0.03
        }
        default_value = feature_defaults.get(feature, 50.0)
        
        for i in range(num_points):
            cols = st.columns([1, 2, 2, 0.5])
            with cols[0]:
                st.write(f"{i+1}")
            with cols[1]:
                time_key = f"time_{i}_{feature}"
                time_val = st.number_input(f"Time {i}_{feature}", 0.0, 100.0, 1.0 + 2.0 * i, 0.1, label_visibility="collapsed", key=time_key)
            with cols[2]:
                value_key = f"value_{i}_{feature}"
                value_val = st.number_input(f"Value {i}_{feature}", 0.0, 10000.0, default_value * (3 - i) / 2, 0.01 if feature in ["Ethyl glucuronide (mg/dL)", "Ethyl sulphate (mg/dL)"] else 1.0, label_visibility="collapsed", key=value_key)
            with cols[3]:
                st.write("")
            
            # Store values from session state (which persists across reruns)
            if time_val >= 0 and value_val >= 0:
                data_points[feature].append({"time": float(time_val), "value": float(value_val)})

else:  # CSV paste
    st.markdown("#### Paste CSV Data")
    st.markdown(f"Paste data as CSV with columns: `Time`, and one or more of: {', '.join(selected_features)}")
    st.markdown("(comma or tab-separated, first row should be headers)")
    st.markdown("**Unit specifications for features:**")
    for feature in selected_features:
        unit = feature_units.get(feature, "")
        st.markdown(f"  - **{feature}**: {unit}")
    st.markdown("- **Time**: hours (h)")
    
    csv_input = st.text_area("Paste your data here:", height=150, placeholder="Time,BAC,UAC,EtG,EtS\n0.0,100,0,0,0\n1.0,80,5,2,1\n2.0,60,10,5,3")
    
    if csv_input.strip():
        try:
            # Try to parse the CSV
            from io import StringIO
            df = pd.read_csv(StringIO(csv_input), sep=r'[,\t]', engine='python')
            
            # Find time column
            time_col = None
            for col in df.columns:
                col_lower = col.lower().strip()
                if 'time' in col_lower:
                    time_col = col
                    break
            
            if not time_col:
                st.error("Could not find 'Time' column. Please use 'Time' as a column header.")
            else:
                # Process each row and extract all available features
                for idx, row in df.iterrows():
                    try:
                        time_val = row[time_col]
                        
                        # Skip if time is NaN or None
                        if pd.isna(time_val):
                            continue
                        if isinstance(time_val, str) and time_val.lower() in ['none', 'nan', '']:
                            continue
                        
                        time_val = float(time_val)
                        
                        # For each selected feature, check if it exists in this row
                        for feature in selected_features:
                            feature_col = None
                            for col in df.columns:
                                if col.strip().upper() == feature.upper():
                                    feature_col = col
                                    break
                            
                            if feature_col:
                                value_val = row[feature_col]
                                
                                # Skip if value is NaN or None or empty string
                                if pd.isna(value_val):
                                    continue
                                if isinstance(value_val, str) and value_val.lower() in ['none', 'nan', '']:
                                    continue
                                
                                try:
                                    value_val = float(value_val)
                                    data_points[feature].append({
                                        "time": time_val,
                                        "value": value_val
                                    })
                                except (ValueError, TypeError):
                                    # Skip values that can't be converted to float
                                    continue
                    except (ValueError, TypeError):
                        # Skip rows that can't be processed
                        continue
                
                total_points = sum(len(points) for points in data_points.values())
                if total_points > 0:
                    st.success(f"✅ Loaded {total_points} data points for {len([f for f in selected_features if data_points[f]])} features")
        except Exception as e:
            st.error(f"Error parsing CSV: {str(e)}")

# Display parsed data
if any(data_points.values()):
    total_points = sum(len(points) for points in data_points.values())
    st.markdown(f"#### Total number of data points loaded: {total_points}")
    
    for feature in selected_features:
        if data_points[feature]:
            st.markdown(f"**{feature}** ({len(data_points[feature])} points)")
            data_df = pd.DataFrame(data_points[feature])
            # Rename columns with units
            unit = feature_units.get(feature, "")
            data_df = data_df.rename(columns={'time': 'Time (h)', 'value': f'{feature} Value ({unit})'})
            st.dataframe(data_df, width='stretch', hide_index=True)
else:
    st.warning("No valid data points entered. Please add data above.")

st.divider()

# === ANTHROPOMETRICS SECTION ===
st.header("Subject anthropometrics")

anthropometrics = get_anthropometrics_ui(defaults={"sex": "Man", "weight": 70.0, "height": 1.72, "age": 30})

st.divider()

# === DRINKING PATTERN SECTION ===
st.header("Design your drinking pattern")

drink_times, drink_lengths, drink_concentrations, drink_volumes, drink_kcals = drink_selector_cards(page_number="09", trigger_simulation_update=True, mark_update=True)


st.divider()

# === SIMULATION SECTION ===
st.header("Simulation settings")

extra_time = st.number_input("Time to simulate after last drink (h):", 0.0, 100.0, 5.0, 0.5, key="extra_time_09")

data_sem = st.number_input("Data uncertainty (%):", 0.1, 100.0, 5.0, 0.1, key="data_sem_09", help="Standard Error of the Mean tolerance for data points (±%)")

# Validate drink arrays before building stimulus dict
if len(drink_times) > 0 and len(drink_lengths) > 0:
    # Check all arrays have same length
    if not (len(drink_times) == len(drink_lengths) == len(drink_concentrations) == len(drink_volumes) == len(drink_kcals)):
        st.error("Error: All drink parameter arrays should have the same length. Please refresh the page.")
    else:
        # No meals in this demo, so prepare empty meal arrays in the right format
        meal_times = []
        meal_kcals = [0]
        
        # Build stimulus dictionary with properly formatted meal arrays
        stim = build_stimulus_dict(
            drink_times, drink_lengths, drink_concentrations, 
            drink_volumes, drink_kcals, meal_times, meal_kcals
        )
        
    # Run simulation button
    if not any(data_points.values()):
        st.error("Please enter at least one data point before running simulation.")
    else:
        with st.spinner("Running simulation..."):
            try:
                sim_results = simulate(model, anthropometrics, stim, extra_time=extra_time)
                st.session_state['sim_results_09'] = sim_results
                st.session_state['data_points_09'] = data_points
                st.session_state['selected_features_09'] = selected_features
                st.session_state['anthropometrics_09'] = anthropometrics.copy()
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
else:
    st.warning("Please configure at least one drink for the simulation to be run.")

# Display results if simulation has been run
if drink_times:
    sim_results = st.session_state['sim_results_09']
    data_points_dict = st.session_state['data_points_09']
    selected_features = st.session_state['selected_features_09']
    
    st.divider()
    st.header("Comparison: simulation vs experimental Data")
    
    # Check if there's any data to compare
    if not any(data_points_dict.values()):
        st.warning("No data points to compare against.")
    else:
        # Create combined multi-feature plot with data points and drink patches
        fig = create_multi_feature_plot(
            sim_results, selected_features, 
            data_points=data_points_dict,
            drink_starts=drink_times,
            drink_lengths=drink_lengths,
            data_sem=data_sem
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="multi_feature_plot_09")
        
        # Extract simulation data for fit analysis
        sim_times = sim_results['Time'].values
        
        # Process each selected feature for fit analysis
        all_fits = []
        
        for feature in selected_features:
            data_points_plot = data_points_dict.get(feature, [])
            
            if not data_points_plot:
                st.info(f"No data points for {feature}")
                continue
            
            if feature not in sim_results.columns:
                st.error(f"Feature '{feature}' not found in simulation results.")
                continue
            
            sim_values = sim_results[feature].values
            
            # === TOLERANCE CHECK ===
            st.markdown(f"#### Fit Analysis for {feature}")
            
            # Extract data times and values
            data_times = [float(dp['time']) for dp in data_points_plot]
            data_values = [float(dp['value']) for dp in data_points_plot]
            
            # Interpolate simulation values at data point times
            interpolated_values = np.interp(data_times, sim_times, sim_values)
            
            # Check tolerance for each point
            tolerance = data_sem / 100.0  # Convert percentage to decimal
            fits = []
            errors = []
            
            for i, data_point in enumerate(data_points_plot):
                data_value = float(data_point['value'])
                interp_value = interpolated_values[i]
                
                # Calculate relative error
                if interp_value != 0:
                    relative_error = abs(data_value - interp_value) / interp_value
                else:
                    relative_error = float('inf') if data_value != 0 else 0
                
                # Check if within tolerance
                within_tolerance = relative_error <= tolerance
                fits.append(within_tolerance)
                errors.append(relative_error * 100)
            
            # Display results table
            results_df = pd.DataFrame({
                'Time (h)': data_times,
                'Experimental': data_values,
                'Simulated': interpolated_values,
                f'Within ±{data_sem}%': ['✅' if fit else '❌' for fit in fits]
            })
            
            st.dataframe(results_df, width='stretch', hide_index=True)
            
            # Overall assessment for this feature
            num_within_tolerance = sum(fits)
            total_points = len(fits)
            
            if all(fits):
                st.success(
                    f"✅ **{feature}: Drinking pattern describes data** ✅\n\n"
                    f"All {total_points} data points are within ±{data_sem}% tolerance!"
                )
            else:
                st.warning(
                    f"**{feature}: Not quite there yet**\n\n"
                    f"{num_within_tolerance}/{total_points} data points are within ±{data_sem}% tolerance."
                )
            
            all_fits.extend(fits)
        
        # Overall success message if all features and points are within tolerance
        if all_fits and len(all_fits) > 0 and all(all_fits):
            st.success(
                f"**OVERALL SUCCESS! Drinking pattern describes all data**\n\n"
                f"All data points across all measured features are within ±{data_sem}% tolerance!"
            )

st.divider()

st.markdown("""
### Tips for Finding the Right Drinking Pattern:

1. **Start simple**: Begin with a single drink and gradually add more if needed
2. **Vary timing**: Adjust the timing of drinks to match peaks and valleys in the data
3. **Adjust volume/strength**: Change drink volume and ABV to match data magnitude
4. **Consider duration**: The drinking period affects the concentration profile
5. **Use the lock feature**: Lock successful drink parameters when experimenting with others

Good luck!
""")
