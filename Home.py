
import streamlit as st

# Install sund in a custom location
import subprocess
import sys
import os 

if "sund" not in os.listdir('./custom_package'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target=./custom_package", 'https://isbgroup.eu/edu/assets/sund-1.0.1.tar.gz#sha256=669a1d05c5c8b68500086e183d831650277012b3ea57e94356de1987b6e94e3e'])


st.title("A physiological-twin for alcohol consumption: connecting short-term drinking habits to plasma PEth")
st.markdown("""Current models for alcohol dynamics lack detailed dynamics of e.g., gastric emptying. 
We have constructed a more detailed model able to both explain differing dynamics of different types of drinks, as well as differences based on anthropometrics. 
Using the model, we can construct interactive use-cases where you can both simulate the dynamics of different drinks, and long-term clinical markers, personalized based on anthropometrics. The use-cases can be reached from the left sidebar, or using the links below. 

[The first use-case revolves around simulating a number of drinks, and observing the dynamics of e.g., ethanol in the blood and breath.](Alcohol_dynamics) 
Here you can specify anthropometrics, and then define a set of drinks and meals to simulate the dynamics of alcohol. 

The three other use-cases revolve around making long-term predictions of the clinical marker of alcohol use phosphatidyl ethanol (PEth). 
[Firstly to evaluate if self-reported weekly use of alcohol and measured PEth levels are in agreement](Evaluating_reported_PEth_levels). Secondly, to showcase [how weekly use of alcohol leads to different values of PEth affected by anthropometrics](Anthropometric_differences_in_PEth). Finally, [how different types of drinks affect the PEth levels](Drink_type_impact_on_PEth).

We hope that this model and the applications can be used to make more informed decisions on alcohol consumption.  

Please note that this application is only for research and visualization purposes, and should not be used to make medical decisions. 
            
This application is a companion-application to the publication titled \"A physiological-twin for alcohol consumption: connecting short-term drinking habits to plasma PEth\", [available as a preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2023.08.18.553836). 
            
If you use this application, please cite as: 

> A Physiologically Based Digital Twin for Alcohol Consumption – Predicting Real-life Drinking Responses and Long-term Plasma PEth.
Henrik Podéus, Christian Simonsson, Patrik Nasr, Mattias Ekstedt, Stergios Kechagias, Peter Lundberg, William Lövfors, Gunnar Cedersund.
bioRxiv 2023.08.18.553836; doi: https://doi.org/10.1101/2023.08.18.553836 

""")

         