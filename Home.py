
import streamlit as st
from sidebar_config import setup_sidebar

# Setup sidebar
setup_sidebar()

st.sidebar.success("Select a demo above.")

# Main content
st.write("# Welcome! 👋")
st.markdown(
"""
## A physiological-twin for alcohol consumption and use cases revolving alcohol markers.

This tool presents interactive demonstrations based on our research on alcohol consumption and its physiological effects. These demonstrations are organized under two main sections (see left sidebar): Paper 1 and Paper 2. These sections provide insights into our findings and methodologies, which are presented in the papers listed below.

**👈 Select a demo from the sidebar** to explore interactive demonstrations!

### Paper 1: Interactive demos

The Paper 1 section contains four interactive demonstrations:
- **Alcohol dynamics** - Simulate drinks and observe dynamics of ethanol in blood and breath
- **Evaluating reported PEth levels** - Check if self-reported alcohol use matches PEth measurements
- **Anthropometric differences in PEth** - See how body characteristics affect PEth levels
- **Drink type impact on PEth** - Compare how different drink types affect PEth

### Paper 2: Interactive demos

The Paper 2 section contains two interactive demonstrations focusing on secondary alcohol metabolites:
- **Alcohol secondary metabolites dynamics** - Simulate drinks and observe the time course of EtG, EtS, and UAC
- **Plausibility of behavior** - Evaluate claimed drinking patterns against measured biomarker levels and model uncertainty

---

**Please note:** This application is only for research and visualization purposes, 
and should not be used to make medical decisions.

### Citation

#### Paper 1

This application is a companion to the publication "A Physiologically Based Digital Twin for Alcohol Consumption – Predicting Real-life Drinking Responses and Long-term Plasma PEth.", 
[available at npj Digital Medicine](https://doi.org/10.1038/s41746-024-01089-6).

If you use this application, please cite as:

> A Physiologically Based Digital Twin for Alcohol Consumption – Predicting Real-life Drinking Responses and Long-term Plasma PEth.
> Henrik Podéus, Christian Simonsson, Patrik Nasr, Mattias Ekstedt, Stergios Kechagias, Peter Lundberg, William Lövfors, Gunnar Cedersund.
> Npj Digital Medicine, 7(1), 1–18. doi: https://doi.org/10.1038/s41746-024-01089-6

#### Paper 2

Citation to paper 2 will be added here when available.
"""
)