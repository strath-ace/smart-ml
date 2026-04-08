# GENEPY

Finding solutions in dynamical systems via generative flow-based models.

## Installing packages

Packages can be installed via pip with
```
pip install -r requirements.txt
```
or via conda with
```
conda create --name <env> --file requirements-conda.txt
```

## Additional Data
Data for the publication "Generating Stable and Metastable Critical Points in Uncertain Systems via Flow-Based Models" are available [here](https://doi.org/10.15129/ecc01ad5-efce-4127-9232-b887b6a2a907).

## File descriptions
 - [es-journal-results.ipynb](es-journal-results.ipynb) - notebook for reproducing and processing publication results
 - [optdist.py](optdist.py) - functions and classes for generic optimisation test cases
 - [potentialdist.py](potentialdist.py) - functions and classes for dynamical system potential test cases
 - tfFlow/ - package for building and training flow-based models
 - scripts/ - scripts for training and testing models
 - mindata/ - files containing locations of minima/equilibirium points for some functions

## Citation

```
@article{wilson_generating_2026,
    author = {Wilson, Callum and Vasile, Massimiliano},
    title = {Generating Stable and Metastable Critical Points in Uncertain Systems via Flow-Based Models},
    journal = {Expert Systems},
    volume = {43},
    number = {3},
    pages = {e70196},
    keywords = {conditional generation, dynamical systems, flow-based models, metastability},
    doi = {https://doi.org/10.1111/exsy.70196},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/exsy.70196},
    eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/exsy.70196},
    year = {2026}
}
```
