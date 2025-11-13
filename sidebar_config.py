import streamlit as st

# Setup sidebar
def setup_sidebar():
    st.sidebar.header("Alcohol consumption digital twin demos")
    st.sidebar.page_link("Home.py", label="Home")
    st.sidebar.divider()
    st.sidebar.header("Paper 1 - real-life drinking & PEth")
    st.sidebar.page_link("pages/01-alcohol-dynamics.py", label="About")
    st.sidebar.page_link("pages/02-alcohol-dynamics-simulation.py", label="Alcohol dynamics")
    st.sidebar.page_link("pages/03-anthropometric-differences-in-PEth.py", label="Anthropometric differences")
    st.sidebar.page_link("pages/04-drink-type-impact-on-PEth.py", label="Drink type impact")
    st.sidebar.page_link("pages/05-evaluating-reported-PEth-levels.py", label="Evaluating reported PEth levels")

    st.sidebar.divider()
    st.sidebar.header("Paper 2 - secondary metabolites & forensic interpretation")
    st.sidebar.page_link("pages/06-secondary-metabolites.py", label="About")
    st.sidebar.page_link("pages/07-secondary-metabolites-simulation.py", label="Secondary metabolites")
    st.sidebar.page_link("pages/08-plausibility-of-behavior.py", label="Plausibility checks")
    st.sidebar.page_link("pages/09-model-fitting.py", label="Evaluate drinking against data")
    st.sidebar.divider()