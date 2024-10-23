# PENGUINS CLASSIFIER documentation!
Welcome to the documentation for PENGUINS CLASSIFIER! This repository uses data from the open Penguin data set available [here](https://www.kaggle.com/datasets/satyajeetrai/palmer-penguins-dataset-for-eda). A script to download the data is available at scripts/download_data.py. This script requires setting up the [Kaggle API](https://www.kaggle.com/docs/api).

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
git clone https://github.com/lauracabayol/LCG-TestDataScience-1
cd LCG-TestDataScience-1
```
## Installation

#### Installation and Environment Setup
We recommend using a virtual environment to install the project dependencies and maintain an isolated workspace.
#### Setting Up a Virtual Environment:
To create a virtual environment using <venv>, run the following commands:
```bash
python -m venv venv
source venv/bin/activate  
```
#### 2. Setting Up a Conda Environment:
Alternatively, you can create a Conda environment with Python 3.10 by executing the following commands:
```
conda create -n TempForecast -c conda-forge python=3.10
conda activate TempForecast
```
The required python modules are in the <requirements.txt> file although these will automatically install when installing the repository.

Once your environment is ready, proceed with the installation of the package:

```
pip install -e .
``` 
#### Optional: Configuring MLflow
For advanced users interested in tracking experiments and using MLflow, please follow the official MLflow setup [guide](https://mlflow.org/docs/latest/getting-started/index.html) to configure the tracking server.

## Usage

#### Running the Models
This project supports four algorithms:

- Gradient Boosting
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
You can choose any of these algorithms for training and predictions.

#### Training the Model
To train a model, use the following command, specifying the desired algorithm:
```bash
python VitalMetrics/modeling/train.py --model-type <algorithm name>
```
Replace <algorithm name> with one of the following options:
- gradient_boosting
- random_forest
- logistic_regression
- svm

#### Making Predictions
Once the model is trained, you can use the trained model to make predictions. Run the following command:
```bash
python VitalMetrics/modeling/predict.py <test features file>
```

#### Accessing the notebooks
The notebooks are loaded on GitHub as .py files. To convert them to .ipynb use <jupytext>

```bash
jupytext --to ipynb notebooks/*.py
```
## Accessing Models
There are three ways of accessing the models.

### Accessing models from MLFlow
The </notebooks/Make_predictions.ipynb> access the model registery in MLFlow to make predictions, compare them, and run the necessary plots and metrics. This option is only available for those having the MLFlow logs, which are not uploaded to GitHub.
```bash
import mlflow.pytorch

# Define the model name and version
model_name = "PENGUINS CLASSIFIER"
model_version = 16 #select the version
model_uri = f"models:/{model_name}/{model_version}"

# Load the model from MLflow
deployed_model = mlflow.sklearn.load_model(model_uri)
```
### Deployed Gradient Boosting model
**The available deployed model as of today is the Gradient Boosting**. There are two different ways of accessing the deployed model:

#### Through weights and biases:
We have deployed the model in the WaB platform and it is available [here](https://huggingface.co/spaces/lauracabayol/PENGUINS_CLASSIFIER). This allows the user to make single predictions from a set of features. It is publicly available for everyone and userfriendly, but does not support making predictions for more than one sample simultaneously. 

#### Model in a Docker container:

For convenience, we have created a Docker container that includes the Gradient Boosting model along with all necessary dependencies. This allows you to run the model without needing access to MLflow or the associated logs.

##### Instructions to Run the Docker Container:

**Build the Docker Image**: First, clone the repository and navigate to the project directory. Then, build the Docker image:
```bash
docker build -t penguin-classificaton:latest .
```
**Run the Docker Container**: Once the image is built, run the container using the following command:
```
docker run -p 9999:9999 penguin-classificaton:latest
```
This will start a Jupyter notebook where you can interact with the pre-trained best-performing model.

**Access the Jupyter Notebook**: Open your web browser and go to:
```bash
http://localhost:9999
```
**The access token is 12345**
The notebook is pre-configured to load and run the best model, so you can use it without needing to access MLflow.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project as long as you adhere to the license terms.