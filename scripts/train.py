import torch
import argparse
from loguru import logger
from pathlib import Path

from temps.archive import Archive
from temps.temps import TempsModule
from temps.temps_arch import EncoderPhotometry, MeasureZ


def train(config: dict) -> None:
    """
    Trains the TempsModule using photometry data.

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing paths, model hyperparameters, and settings.

    Returns:
    --------
    None
    """

    # Paths
    path_calib = Path(config["path_calib"])
    path_valid = Path(config["path_valid"])
    output_model = Path(config["output_model"])

    # Initialize neural network modules for photometry features and redshift measurement
    nn_features = EncoderPhotometry()
    nn_z = MeasureZ(num_gauss=6)  # Example for Gaussian mixture model with 6 components

    # Initialize the TempsModule with the defined neural networks
    temps_module = TempsModule(nn_features, nn_z)

    # Retrieve photometry and spectroscopic data for training
    photoz_archive = Archive(
        path_calib=path_calib,
        path_valid=path_valid,
        drop_stars=False,
        clean_photometry=False,
        only_zspec=config["only_zs"],
        columns_photometry=config["bands"],
    )

    f, specz, VIS_mag, f_DA, z_DA = photoz_archive.get_training_data()

    # Train the TempsModule
    logger.info("Starting model training...")
    temps_module.train(
        input_data=f,
        input_data_da=f_DA,
        target_data=specz,
        nepochs=config["hyperparams"]["nepochs"],
        step_size=config["hyperparams"]["nepochs"],
        val_fraction=0.1,  # Validation fraction of 10%
        lr=config["hyperparams"]["learning_rate"],
    )
    logger.info("Model training complete.")

    # Save the trained models
    logger.info("Saving trained models...")
    torch.save(temps_module.modelF.state_dict(), output_model / "modelF_zs_test.pt")
    torch.save(temps_module.modelZ.state_dict(), output_model / "modelZ_zs_test.pt")
    logger.info("Models saved at: {}", output_model)


def main() -> None:
    """
    Main entry point for the training script.

    Reads the configuration file, calls the `train` function, and handles logging.
    """
    # Get command-line arguments
    args = get_args()

    # Load the configuration from the provided path
    config_path = args.config_path
    logger.info("Loading configuration from {}", config_path)

    # Read the configuration file (assuming YAML format)
    config = read_config(config_path)

    # Call the train function
    train(config)


def get_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns:
    --------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Training script for TempsModule")

    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to the configuration file (YAML format)",
    )

    return parser.parse_args()


def read_config(config_path: Path) -> dict:
    """
    Reads the configuration from a YAML file.

    Parameters:
    -----------
    config_path : Path
        Path to the configuration YAML file.

    Returns:
    --------
    dict
        Parsed configuration dictionary.
    """
    import yaml

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":
    main()
