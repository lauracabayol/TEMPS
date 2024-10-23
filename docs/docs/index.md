# TEMPS documentation!
Welcome to the documentation for TEMPS! 
This repository contains a neural network to predict photometric redshifts. The neural network incorporates domain adaptation, a methodology to mitigate the impact of sample bias in the spectroscopic training samples. 

The training and validation data are not available in the repository, but the model can be trained with new data. The model is also deployed and available [here](https://huggingface.co/spaces/lauracabayol/TEMPS). The model in production enables making predictions for new galaxies with the pretrained models. 

Details on the data and the pa


## Table of Contents

- [Prerequisites](##Prerequisites)
- [Installation](##installation)
- [Usage](##usage)
- [Deployed model](##Accessing-the-LSTM-depolyed-model)
- [License](##license)

## Prerequisites

Before proceeding, ensure that the following software is installed on your system:

- Python 3.10
- [pip](https://pip.pypa.io/en/stable/installation/)
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

You will also need to clone the repository to your local environment by executing the following commands:

```bash
git clone https://github.com/lauracabayol/TEMPS
cd TEMPS
```
## Installation

We recommend using a conda environment with Python 3.10 by executing the following commands:
```
conda create -n temps -c conda-forge python=3.10
conda activate temps
```
Once your environment is ready, proceed with the installation of the package:

```
pip install -e .
```
This will already install the dependencies. 


## Deployed model
Alternatively, one can access the deployed models at [HuggigFace](https://huggingface.co/spaces/lauracabayol/TEMPS). This enbles making predictions from a file with photometric measurements. The format should be a csv file with the following band photometries in this order: G,R,I,Z,Y,H,J. 


## Notebooks
The repository contains notebooks to reproduce the figures in the [Paper](paper)
The notebooks are loaded on GitHub as .py files. To convert them to .ipynb use <jupytext>

```bash
jupytext --to ipynb notebooks/*.py
```

## Training the Model
The model can be trained using the train.py script at the repo main directory. 

```
python train.py --config-file data/config.yml
```
One only needs to modify the config file to point at the input files. Make sure to also specify the photometric bands naming, and the spectroscopic and photometric redshift columns. 
Input catalogs must be in .fits or .csv formats and these should already contain clean photometry. 

If extinction_corr is set to True, one must specify the column namings of the E_VB corrections in the config file. 

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project as long as you adhere to the license terms.