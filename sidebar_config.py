import streamlit as st

# Setup sidebar
def setup_sidebar():
    st.sidebar.header("Alcohol consumption digital twin demos")
    st.sidebar.page_link("Home.py", label="🍺 Home")
    st.sidebar.divider()
    st.sidebar.header("📄 Paper 1")
    st.sidebar.page_link("pages/01_paper_1.py", label="👉 About")
    st.sidebar.page_link("pages/02_alcohol_dynamics.py", label="🍺 Alcohol dynamics")
    st.sidebar.page_link("pages/03_anthropometric_differences_in_PEth.py", label="👤 Anthropometric differences")
    st.sidebar.page_link("pages/04_drink_type_impact_on_PEth.py", label="🍷 Drink type impact")
    st.sidebar.page_link("pages/05_evaluating_reported_PEth_levels.py", label="🧾 Evaluating reported PEth levels")

    st.sidebar.divider()
    st.sidebar.header("📄 Paper 2")
    st.sidebar.page_link("pages/06_paper_2.py", label="👉 About")
    st.sidebar.page_link("pages/07_alcohol_secondary_metabolites_dynamics.py", label="⚗️ Secondary metabolites")
    st.sidebar.page_link("pages/08_plausibility_of_behavior.py", label="📊 Plausibility checks")
    st.sidebar.page_link("pages/09_model_fitting.py", label="📈 Evaluate drinking against data")
    st.sidebar.divider()