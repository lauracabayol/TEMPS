from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from temps.archive import Archive
from temps.temps_arch import EncoderPhotometry, MeasureZ

logger = logging.getLogger(__name__)

# Define the prediction function that will be called by Gradio
def predict(input_file_path: Path, model_path: Path):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(levelname)s:%(message)s",
    )

    logger.info("Loading data and converting fluxes to colors")

    # Load the input data file (CSV)
    try:
        fluxes = pd.read_csv(input_file_path, sep=',', header=0)
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
        nn_features.load_state_dict(torch.load(model_path / 'modelF.pt', map_location=torch.device('cpu')))
        nn_z.load_state_dict(torch.load(model_path / 'modelZ.pt', map_location=torch.device('cpu')))
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return f"Error loading model: {e}"

    temps_module = TempsModule(nn_features, nn_z)

    # Run predictions
    try:
        z, pz, odds = temps_module.get_pz(input_data=torch.Tensor(colors.values), 
                                          return_pz=True,
                                          return_flag=True)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return f"Error during prediction: {e}"

    # Return the predictions as a dictionary
    result = {
        "redshift (z)": z.tolist(),
        "posterior (pz)": pz.tolist(),
        "odds": odds.tolist()
    }
    return result


# Gradio app
def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = get_args()

    # Define the Gradio interface
    gr.Interface(
        fn=predict,  # the function that Gradio will call
        inputs=[
            gr.inputs.File(label="Upload your input CSV file"),  # file input for the data
            gr.inputs.Textbox(label="Model path", default=str(args.model_path)),  # text input for model path
        ],
        outputs="json",  # return the results as JSON
        live=False,
        title="Prediction App",
        description="Upload a CSV file with your data to get predictions.",
    ).launch(server_name=args.server_name, server_port=args.port)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        type=str,
    )

    parser.add_argument(
        "--input-file-path",
        type=Path,
        help="Path to the input CSV file",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to the model files",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
