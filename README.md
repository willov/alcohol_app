# A physiologically based digital twin for alcohol consumption – predicting real-life drinking responses and long-term plasma PEth

<a href="https://doi.org/10.1038/s41746-024-01089-6">
	<img src="./article-frontpage.png" alt="article-frontpage" width="50%">
</a>

This repository contains the code for the eHealth application prototype developed in the publication titled \"A physiologically based digital twin for alcohol consumption – predicting real-life drinking responses and long-term plasma PEth\", [published in npj Digital Medicine](https://doi.org/10.1038/s41746-024-01089-6).

The prototype is hosted at [https://alcohol.streamlit.app](https://alcohol.streamlit.app), but can also be run locally. To do that, install the required packages listed in the `pyproject.toml` file: 

```bash
uv sync
```

Then run the application by running `uv run streamlit run Home.py` in the terminal. 

Please note that the application can take a few minutes to start up, primarily when running at [https://alcohol.streamlit.app](https://alcohol.streamlit.app), but also locally. Also note that a valid C-compiler is necessary for running the application locally.

The app was tested with Python 3.13, with the dependencies listed in `pyproject.toml` and `uv.lock`. 

## How to cite

If you use this application, please cite as:

```text
Podéus H, Simonsson C, Nasr P, Ekstedt M, Kechagias S, Lundberg P, Lövfors W, Cedersund G (2024) A physiologically-based digital twin for alcohol consumption — predicting real-life drinking responses and long-term plasma PEth. npj Digital Medicine 7:112. https://doi.org/10.1038/s41746-024-01089-6
```

Or use the following bibtex entry:

```bibtex
@article{podeus_2024,
	title = {A physiologically-based digital twin for alcohol consumption — predicting real-life drinking responses and long-term plasma {PEth}},
	volume = {7},
	issn = {2398-6352},
	url = {https://doi.org/10.1038/s41746-024-01089-6},
	doi = {10.1038/s41746-024-01089-6},
	number = {1},
	journal = {npj Digital Medicine},
	author = {Podéus, Henrik and Simonsson, Christian and Nasr, Patrik and Ekstedt, Mattias and Kechagias, Stergios and Lundberg, Peter and Lövfors, William and Cedersund, Gunnar},
	month = may,
	year = {2024},
	pages = {112},
}
```
