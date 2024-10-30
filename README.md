---
title: Photo-z predictor
emoji: 🌌
colorFrom: blue
colorTo: red
sdk: docker
python_version: 3.11
pinned: false
---

[![Python application](https://img.shields.io/badge/python_application-passing-success)](https://github.com/lauracabayol/TEMPS/actions)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗_Spaces-deployed-blue)](https://huggingface.co/spaces/lauracabayol/TEMPS)
[![Mkdocs](https://img.shields.io/badge/mkdocs-passing-success)](https://github.com/lauracabayol/TEMPS/actions)

This repository contains a neural network to predict photometric redshifts. The neural network incorporates domain adaptation, a methodology to mitigate the impact of sample bias in the spectroscopic training samples.

The model is deployed and available [here](https://huggingface.co/spaces/lauracabayol/TEMPS) (WiP). The model in production enables making predictions for new galaxies with the pretrained models. One can also run the code locally
```bash
python app.py
```
This opens a Gradio interface to make predictions for new galaxies.

Documentation is available [here](https://lauracabayol.github.io/TEMPS/).

## Installation

You will also need to clone the repository to your local environment by executing the following commands:

```bash
git clone https://github.com/lauracabayol/TEMPS
cd TEMPS
```
## Installation

We recommend using a conda environment with Python 3.10 by executing the following commands:
```bash
conda create -n temps -c conda-forge python=3.10
conda activate temps
```
Once your environment is ready, proceed with the installation of the package:

```bash
pip install -e .
```
This will already install the dependencies.

## Notebooks

The repository contains notebooks to reproduce the figures in the paper (TBD)
The notebooks are loaded on GitHub as .py files. To convert them to .ipynb use <jupytext>

```bash
jupytext --to ipynb notebooks/*.py
```

## Usage

The model can be trained using the train.py script at the repo main directory.

```bash
python scripts/train.py --config-file data/config.yml
```
More information on the training script can be found in the [docs](https://lauracabayol.github.io/TEMPS/docs/docs/index.html)

To make predictions for new galaxies, you can use the predict.py script (TBD).

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project as long as you adhere to the license terms.
