import os
import logging
import numpy as np
import polars as pl
import h5py
from sklearn.mixture import GaussianMixture
from config import CHECKPOINT_FILE


def get_logger():
    return logging.getLogger(__name__)


def save_checkpoint_polars_df(group, key: str, df: pl.DataFrame):
    """
    Helper function to save Polars DataFrame to HDF5 group.
    """
    np_data = df.to_numpy()
    columns = df.columns
    group.create_dataset(key, data=np_data)
    # Save column names as string array
    group.create_dataset(
        f"{key}_columns", data=[col.encode("utf-8") for col in columns]
    )


def load_checkpoint_polars_df(group, key: str) -> pl.DataFrame:
    """
    Helper function to load Polars DataFrame from HDF5 group.
    """
    data = group[key][:]
    columns_bytes = group[f"{key}_columns"][:]
    columns = [col.decode("utf-8") for col in columns_bytes]
    return pl.DataFrame(data, schema=columns)


def save_checkpoint(checkpoint_data: dict, step: str):
    """
    Save checkpoint data to HDF5 file.

    Args:
        checkpoint_data: Dictionary with data to save
        step: String identifying the current step
    """
    logger = get_logger()
    logger.info(f"Saving checkpoint at step: {step}")

    with h5py.File(CHECKPOINT_FILE, "a") as f:
        # Create or update step group
        if step in f:
            del f[step]  # Remove existing step data

        step_group = f.create_group(step)

        # Save each piece of data
        for key, value in checkpoint_data.items():
            if isinstance(value, np.ndarray):
                step_group.create_dataset(key, data=value)
            elif isinstance(value, (int, float, str)):
                step_group.attrs[key] = value
            elif isinstance(value, list):
                step_group.create_dataset(key, data=np.array(value))
            elif isinstance(value, dict):
                # Save dictionaries as attributes
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, str)):
                        step_group.attrs[f"{key}_{sub_key}"] = sub_value
            elif hasattr(value, "to_numpy"):  # Polars DataFrame
                # Use helper function for Polars DataFrames
                save_checkpoint_polars_df(step_group, key, value)

        # Save GMM model if present
        if "gmm_model" in checkpoint_data and checkpoint_data["gmm_model"] is not None:
            gmm = checkpoint_data["gmm_model"]
            gmm_group = step_group.create_group("gmm_model")
            gmm_group.create_dataset("weights", data=gmm.weights_)
            gmm_group.create_dataset("means", data=gmm.means_)
            gmm_group.create_dataset("covariances", data=gmm.covariances_)
            gmm_group.attrs["n_components"] = gmm.n_components
            gmm_group.attrs["covariance_type"] = gmm.covariance_type
            gmm_group.attrs["converged"] = gmm.converged_

        # Update progress metadata
        f.attrs["last_completed_step"] = step
        f.attrs["timestamp"] = np.bytes_(str(np.datetime64("now")))

    logger.debug(f"Checkpoint saved successfully at step: {step}")


def load_checkpoint(step: str = None) -> dict:
    """
    Load checkpoint data from HDF5 file.

    Args:
        step: Specific step to load, if None loads the last completed step

    Returns:
        Dictionary with loaded data, or empty dict if no checkpoint exists
    """
    logger = get_logger()

    if not os.path.exists(CHECKPOINT_FILE):
        logger.info("No checkpoint file found, starting from scratch")
        return {}

    try:
        with h5py.File(CHECKPOINT_FILE, "r") as f:
            if step is None:
                if "last_completed_step" in f.attrs:
                    step = f.attrs["last_completed_step"]
                    if isinstance(step, bytes):
                        step = step.decode("utf-8")
                else:
                    logger.info("No completed steps found in checkpoint")
                    return {}

            if step not in f:
                logger.info(f"Step '{step}' not found in checkpoint")
                return {}

            logger.info(f"Loading checkpoint from step: {step}")
            step_group = f[step]

            checkpoint_data = {}

            # Load datasets
            for key in step_group.keys():
                if key == "gmm_model":
                    # Reconstruct GMM model
                    gmm_group = step_group[key]
                    gmm = GaussianMixture(
                        n_components=gmm_group.attrs["n_components"],
                        covariance_type=gmm_group.attrs["covariance_type"].decode(
                            "utf-8"
                        )
                        if isinstance(gmm_group.attrs["covariance_type"], bytes)
                        else gmm_group.attrs["covariance_type"],
                    )
                    gmm.weights_ = gmm_group["weights"][:]
                    gmm.means_ = gmm_group["means"][:]
                    gmm.covariances_ = gmm_group["covariances"][:]
                    gmm.converged_ = gmm_group.attrs["converged"]
                    gmm.n_components = gmm_group.attrs["n_components"]
                    checkpoint_data["gmm_model"] = gmm
                elif key.endswith("_columns"):
                    # Skip column metadata, handled with main dataset
                    continue
                elif f"{key}_columns" in step_group:
                    # Reconstruct Polars DataFrame using helper function
                    checkpoint_data[key] = load_checkpoint_polars_df(step_group, key)
                else:
                    # Regular numpy array
                    checkpoint_data[key] = step_group[key][:]

            # Load attributes
            for key in step_group.attrs.keys():
                if key not in [
                    "last_completed_step",
                    "timestamp",
                ]:  # Skip metadata attributes
                    value = step_group.attrs[key]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    checkpoint_data[key] = value

            logger.info(f"Checkpoint loaded successfully from step: {step}")
            return checkpoint_data

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return {}


def get_completed_steps() -> list:
    """Get list of completed steps from checkpoint file."""
    logger = get_logger()

    if not os.path.exists(CHECKPOINT_FILE):
        return []

    try:
        with h5py.File(CHECKPOINT_FILE, "r") as f:
            return list(f.keys())
    except Exception as e:
        logger.error(f"Error reading checkpoint steps: {e}")
        return []
