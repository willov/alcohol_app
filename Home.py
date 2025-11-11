
import streamlit as st
from sidebar_config import setup_sidebar

# Setup sidebar
setup_sidebar()

st.sidebar.success("Select a demo above.")

# Main content
st.markdown(
"""
## A digital twin for alcohol consumption 

This project provides an open, interactive implementation of a physiologically based digital twin for alcohol consumption. The apps below let researchers and practitioners explore how different drinking patterns and individual characteristics influence short-term biomarkers (blood alcohol, breath alcohol, urine alcohol) and longer-term clinical markers (plasma PEth), and how secondary metabolites (EtG, EtS) behave after drinking.

Explore the interactive demos from the left sidebar or the links below. The demos are designed for research, teaching, and reproducible exploration — not for medical decision-making.

The demonstrations have originally been developed as a companion to two academic publications (papers). Below are brief summaries of each paper along with quick links to the corresponding demos.

---

### Paper 1 — A Physiologically Based Digital Twin for Alcohol Consumption – Predicting Real-life Drinking Responses and Long-term Plasma PEth

This paper presents a mechanistic, physiologically based digital twin that reproduces real-life ethanol kinetics across drink types and individuals. It describes how the blood alcohol concentration is affected by pairing the alcohol with different drinks and meals. It also links short-term ethanol exposure to the formation of long-term PEth, enabling personalized simulations and comparisons against measured PEth levels. The paper was published in npj Digital Medicine in 2024.

Additional information is available in the [*About* section](paper_1) for *Paper 1* in the sidebar.
#### Citation

This application is a companion to the publication "A Physiologically Based Digital Twin for Alcohol Consumption – Predicting Real-life Drinking Responses and Long-term Plasma PEth.", 
[available at npj Digital Medicine](https://doi.org/10.1038/s41746-024-01089-6).

If you use this application, please cite as:

> Podéus H, Simonsson C, Nasr P, Ekstedt M, Kechagias S, Lundberg P, Lövfors W, Cedersund G (2024) A physiologically-based digital twin for alcohol consumption — predicting real-life drinking responses and long-term plasma PEth. npj Digital Medicine 7:112. https://doi.org/10.1038/s41746-024-01089-6

---

### Paper 2 — Predicting Real-life Drinking Scenarios through a Physiological Digital Twin Incorporating Secondary Alcohol Markers

This extended model incorporates secondary metabolites (EtG, EtS) and urine alcohol concentration, highlighting their different time windows and utility for detecting recent drinking. The paper focuses on how these metabolites behave after typical drinking scenarios, and how the model can be used to assist in forensic interpretation, particularly in the context of drunk driving. The paper have been submitted for peer-review in November 2025.

**Quick link paper 2:**

- Paper 2: Secondary metabolites and plausibility checks ([open Paper 2 demos](paper_2))

#### Citation

Citation to paper 2 will be added here when available.

---

"""
)