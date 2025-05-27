import logging
import polars as pl
from pathlib import Path
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from config import (
    METADATA_PATTERN,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_OVERLAP_THRESHOLD,
)
from checkpoint_manager import get_completed_steps, load_checkpoint, save_checkpoint


def get_logger():
    return logging.getLogger(__name__)


def extract_metadata_from_filename(filename: str) -> dict:
    """Extract metadata from filename using regex pattern."""
    match = METADATA_PATTERN.search(filename)
    if not match:
        raise ValueError(f"File name {filename} does not match the expected pattern.")
    return {
        key: float(value.replace("p", ".")) for key, value in match.groupdict().items()
    }


def load_all_data(path: str) -> pl.DataFrame:
    """Load all .mat files from the specified path and return as Polars DataFrame."""
    logger = get_logger()

    # Check if data loading step is already completed
    completed_steps = get_completed_steps()
    if "data_loading" in completed_steps:
        logger.info("Data loading step found in checkpoint, loading from checkpoint...")
        checkpoint_data = load_checkpoint("data_loading")
        return checkpoint_data["raw_data"]

    logger.info("Loading data from scratch...")
    dfs = []
    for directory in Path(path).iterdir():
        if not directory.is_dir():
            continue
        for file_path in directory.rglob("*.mat"):
            try:
                metadata = extract_metadata_from_filename(file_path.name)
                mat = loadmat(file_path)["rconst"][0]
                I, Q = mat.real, mat.imag

                df = pl.DataFrame(
                    {
                        "I": I,
                        "Q": Q,
                        "Distance": metadata["Distance"],
                        "Power": metadata["Power"],
                        "Spacing": metadata["Spacing"],
                        "OSNR": metadata["OSNR"],
                    }
                )
                dfs.append(df)
                logger.debug(f"Loaded file: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load file {file_path.name}: {e}")

    raw_data = pl.concat(dfs)

    # Save checkpoint
    save_checkpoint({"raw_data": raw_data, "n_files_loaded": len(dfs)}, "data_loading")

    return raw_data


def filter_overlap(
    df: pl.DataFrame, threshold: float = DEFAULT_OVERLAP_THRESHOLD
) -> pl.DataFrame:
    """Filter data based on spectral overlapping threshold."""
    logger = get_logger()

    # Check if filtering step is already completed
    completed_steps = get_completed_steps()
    if "data_filtering" in completed_steps:
        logger.info(
            "Data filtering step found in checkpoint, loading from checkpoint..."
        )
        checkpoint_data = load_checkpoint("data_filtering")
        return checkpoint_data["filtered_data"]

    logger.info(f"Filtering entries with Spectral Overlapping (< {threshold} GHz)")
    filtered_data = df.filter(pl.col("Spacing") < threshold)

    # Save checkpoint
    save_checkpoint(
        {
            "filtered_data": filtered_data,
            "original_size": len(df),
            "filtered_size": len(filtered_data),
            "threshold": threshold,
        },
        "data_filtering",
    )

    return filtered_data


def train_test_split_data(
    df: pl.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """Split the data into training and test sets."""
    logger = get_logger()

    # Check if split step is already completed
    completed_steps = get_completed_steps()
    if "data_splitting" in completed_steps:
        logger.info(
            "Data splitting step found in checkpoint, loading from checkpoint..."
        )
        checkpoint_data = load_checkpoint("data_splitting")
        return (
            checkpoint_data["train_df"],
            checkpoint_data["test_df"],
            checkpoint_data["X_train"],
            checkpoint_data["X_test"],
        )

    logger.info(
        f"Splitting data: {int((1 - test_size) * 100)}% train, {int(test_size * 100)}% test"
    )

    # Convert to numpy for sklearn train_test_split
    X = df.select(["I", "Q"]).to_numpy()
    metadata = df.select(["Distance", "Power", "Spacing", "OSNR"]).to_numpy()

    # Split the data
    X_train, X_test, meta_train, meta_test = train_test_split(
        X, metadata, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Convert back to Polars DataFrames
    train_df = pl.DataFrame(
        {
            "I": X_train[:, 0],
            "Q": X_train[:, 1],
            "Distance": meta_train[:, 0],
            "Power": meta_train[:, 1],
            "Spacing": meta_train[:, 2],
            "OSNR": meta_train[:, 3],
        }
    )

    test_df = pl.DataFrame(
        {
            "I": X_test[:, 0],
            "Q": X_test[:, 1],
            "Distance": meta_test[:, 0],
            "Power": meta_test[:, 1],
            "Spacing": meta_test[:, 2],
            "OSNR": meta_test[:, 3],
        }
    )

    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Save checkpoint
    save_checkpoint(
        {
            "train_df": train_df,
            "test_df": test_df,
            "X_train": X_train,
            "X_test": X_test,
            "test_size": test_size,
            "random_state": random_state,
        },
        "data_splitting",
    )

    return train_df, test_df, X_train, X_test
