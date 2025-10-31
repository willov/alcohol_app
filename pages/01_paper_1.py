import streamlit as st
from sidebar_config import setup_sidebar

# Setup sidebar
setup_sidebar()

# Main content
st.markdown("""
## A Physiologically Based Digital Twin for Alcohol Consumption – Predicting Real-life Drinking Responses and Long-term Plasma PEth

Current models for alcohol dynamics lack detailed dynamics of e.g., gastric emptying. 
We have constructed a more detailed model able to both explain differing dynamics of different types of drinks, as well as differences based on anthropometrics. 
Using the model, we can construct interactive use-cases where you can both simulate the dynamics of different drinks, and long-term clinical markers, personalized based on anthropometrics. The use-cases can be reached from the left sidebar, or using the links below. 

[The first use-case revolves around simulating a number of drinks, and observing the dynamics of e.g., ethanol in the blood and breath.](Alcohol_dynamics) 
Here you can specify anthropometrics, and then define a set of drinks and meals to simulate the dynamics of alcohol. 

The three other use-cases revolve around making long-term predictions of the clinical marker of alcohol use phosphatidyl ethanol (PEth). 
[Firstly to evaluate if self-reported weekly use of alcohol and measured PEth levels are in agreement](Evaluating_reported_PEth_levels). Secondly, to showcase [how weekly use of alcohol leads to different values of PEth affected by anthropometrics](Anthropometric_differences_in_PEth). Finally, [how different types of drinks affect the PEth levels](Drink_type_impact_on_PEth).

We hope that this model and the applications can be used to make more informed decisions on alcohol consumption.  

**👈 Select a demo from the sidebar** to explore the interactive demonstrations!

### Interactive demos

This section contains four interactive demonstrations:

- **Alcohol dynamics** - Simulate drinks and observe dynamics of ethanol in blood and breath
- **Anthropometric differences in PEth** - See how body characteristics affect PEth levels
- **Drink type impact on PEth** - Compare how different drink types affect PEth
- **Evaluating reported PEth levels** - Check if self-reported alcohol use matches PEth measurements

---

### Citation

This application is a companion to the publication "A Physiologically Based Digital Twin for Alcohol Consumption – Predicting Real-life Drinking Responses and Long-term Plasma PEth.", 
[available at npj Digital Medicine](https://doi.org/10.1038/s41746-024-01089-6).

If you use this application, please cite as:

> A Physiologically Based Digital Twin for Alcohol Consumption – Predicting Real-life Drinking Responses and Long-term Plasma PEth.
> Henrik Podéus, Christian Simonsson, Patrik Nasr, Mattias Ekstedt, Stergios Kechagias, Peter Lundberg, William Lövfors, Gunnar Cedersund.
> Npj Digital Medicine, 7(1), 1–18. doi: https://doi.org/10.1038/s41746-024-01089-6
""")
