import streamlit as st
from sidebar_config import setup_sidebar

# Setup sidebar
setup_sidebar()

# Main content
st.markdown("""
## Predicting Real-life Drinking Scenarios through a Physiological Digital Twin Incorporating Secondary Alcohol Markers

Secondary alcohol metabolites such as ethyl glucuronide (EtG), ethyl sulphate (EtS), and urine alcohol concentration (UAC) are important biomarkers for detecting recent alcohol consumption. These metabolites have different time profiles compared to blood alcohol concentration and can be detected for longer periods after alcohol consumption.

We have extended our physiologically based digital twin model to include detailed dynamics of these secondary metabolites. The model can predict the time course of EtG, EtS, and UAC following alcohol consumption, accounting for individual differences in anthropometrics and drinking patterns.

Using this extended model, we have constructed interactive use-cases where you can simulate the dynamics of secondary metabolites and evaluate the plausibility of claimed drinking patterns against measured biomarker levels. The use-cases can be reached from the left sidebar, or using the links below.

[The first use-case allows you to simulate drinks and observe the dynamics of secondary alcohol metabolites](Alcohol_secondary_metabolities_dynamics). 
Here you can specify anthropometrics and define a set of drinks and meals to simulate the time course of EtG, EtS, UAC, and other metabolites.

[The second use-case enables you to evaluate the plausibility of claimed alcohol consumption](Plausability_of_behaviour) by comparing model predictions with measured biomarker data. You can see how the model's uncertainty bounds compare with actual measurements, and explore how different drinking scenarios affect the predicted metabolite profiles.

We hope that this model and these applications can provide insights into secondary metabolite dynamics and support more informed interpretation of biomarker measurements.

**👈 Select a demo from the sidebar** to explore the interactive demonstrations!

### Interactive demos

This section contains two interactive demonstrations:

- **Alcohol secondary metabolites dynamics** - Simulate drinks and observe the time course of EtG, EtS, and UAC
- **Plausibility of behavior** - Evaluate claimed drinking patterns against measured biomarker levels and model uncertainty

---

### Citation

This application is a companion to the upcoming publication on secondary alcohol metabolites modeling.

If you use this application, please check back for citation details.
""")
