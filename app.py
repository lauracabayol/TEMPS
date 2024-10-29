from __future__ import annotations
import argparse
import logging
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

from temps.temps_arch import EncoderPhotometry, MeasureZ
from temps.temps import TempsModule

logger = logging.getLogger(__name__)

# Define the prediction function that will be called by Gradio
def predict(input_file_path: Path):
    model_path = Path("models/")

    logger.info("Loading data and converting fluxes to colors")

    # Load the input data file (CSV)
    try:
        fluxes = pd.read_csv(input_file_path, sep=",", header=0)
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return f"Error loading file: {e}"

    # Assuming that the model expects "colors" as input
    colors = fluxes.iloc[:, :-1] / fluxes.iloc[:, 1:]

    logger.info("Loading model...")

    # Load the neural network models from the given model path
    nn_features = EncoderPhotometry()
    nn_z = MeasureZ(num_gauss=6)

    try:
        nn_features.load_state_dict(
            torch.load(model_path / "modelF.pt", map_location=torch.device("cpu"))
        )
        nn_z.load_state_dict(
            torch.load(model_path / "modelZ.pt", map_location=torch.device("cpu"))
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return f"Error loading model: {e}"

    temps_module = TempsModule(nn_features, nn_z)

    # Run predictions
    try:
        z, pz, odds = temps_module.get_pz(
            input_data=torch.Tensor(colors.values), return_pz=True, return_flag=True
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return f"Error during prediction: {e}"

    # Return the predictions as a dictionary
    result = {
        "redshift (z)": z.tolist(),
        "posterior (pz)": pz.tolist(),
        "odds": odds.tolist(),
    }
    return result


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    parser.add_argument(
        "--server-address",  # Changed from server-name
        default="0.0.0.0",  # Changed default to match launch
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
    interface.launch(server_name=args.server_address, server_port=args.port, share=True)
