from __future__ import annotations  # This should actually be the first import
import argparse
import logging
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
import os

from temps.temps_arch import EncoderPhotometry, MeasureZ
from temps.temps import TempsModule
from temps.constants import PROJ_ROOT

logger = logging.getLogger(__name__)


def get_model_path() -> Path:
    """Get the appropriate model path for both local and Docker/HF environments"""
    if os.environ.get("SPACE_ID"):
        # HuggingFace Spaces - models will be in the root directory
        logger.info("Running on HuggingFace Spaces")
        return Path("data/models")  # Absolute path to models in HF Spaces
    else:
        return PROJ_ROOT / "data/models/"


def load_models(model_path: Path):
    logger.info(f"Loading models from {model_path}")
    nn_features = EncoderPhotometry()
    nn_z = MeasureZ(num_gauss=6)

    nn_features.load_state_dict(
        torch.load(
            model_path / "modelF_DA.pt",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    nn_z.load_state_dict(
        torch.load(
            model_path / "modelZ_DA.pt",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )

    return nn_features, nn_z


def predict(input_file_path: Path):
    global LOADED_MODELS
    if LOADED_MODELS is None:
        logger.error("Models not loaded!")
        return "Error: Models not initialized"

    nn_features, nn_z = LOADED_MODELS

    # Rest of your predict function, but use the pre-loaded models
    try:
        fluxes = pd.read_csv(input_file_path, sep=",", header=0)
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return f"Error loading file: {e}"

    colors = fluxes.values[:, :-1] / fluxes.values[:, 1:]

    temps_module = TempsModule(nn_features, nn_z)

    try:
        z, pz, odds = temps_module.get_pz(
            input_data=torch.Tensor(colors), return_pz=True, return_flag=True
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return f"Error during prediction: {e}"

    return (z.tolist(),)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    parser.add_argument(
        "--server-address",  # Changed from server-name
        default="127.0.0.1",  # Changed default to match launch
        type=str,
    )

    parser.add_argument(
        "--input-file-path",
        type=Path,
        help="Path to the input CSV file",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
    )

    return parser.parse_args()


interface = gr.Interface(
    fn=predict,
    inputs=[gr.File(label="Upload CSV file", file_types=[".csv"], type="filepath")],
    outputs=[gr.JSON(label="Predictions")],
    title="Photometric Redshift Prediction",
    description="Upload a CSV file containing flux measurements to get redshift predictions, posterior probabilities, and odds.",
)

if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.log_level)

    # Load models before creating the interface
    try:
        # model_path = PROJ_ROOT / "data/models/"
        model_path = get_model_path()
        logger.info("Loading models...")
        LOADED_MODELS = load_models(model_path)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    interface.launch()
