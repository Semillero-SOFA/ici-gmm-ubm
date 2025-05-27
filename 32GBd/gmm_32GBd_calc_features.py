import logging
import os
import pickle
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from scipy.io import loadmat
from sklearn.mixture import GaussianMixture

# ===========================
# Globals and Configuration
# ===========================

# Create a logger for this script
FILENAME = os.path.basename(__file__)[:-3]


def setup_logger(name: str) -> logging.Logger:
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(f"{name}.log"),
        ],
    )
    _logger = logging.getLogger(name)
    return _logger


logger = setup_logger(FILENAME)


def find_root() -> Path:
    """
    Find the root directory of the Git project.

    Returns:
        Path: The absolute path of the Git root directory, or None if not found.
    """
    try:
        # Run 'git rev-parse --show-toplevel' to get the root directory
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check if the command was successful
        if result.returncode == 0:
            return Path(result.stdout.strip())
        # Git command failed, print the error message
        logger.error(result.stderr.strip())
        return None

    except Exception as e:
        # Handle exceptions, e.g., subprocess.CalledProcessError
        logger.error(str(e))
        return None


LOCAL_ROOT = find_root()
GLOBAL_ROOT = LOCAL_ROOT.parent
DATABASE_DIR = f"{GLOBAL_ROOT}/databases"
GLOBAL_RESULTS_DIR = f"{GLOBAL_ROOT}/results"

# Create results directory if it doesn't exist
RESULTS_DIR = f"{GLOBAL_RESULTS_DIR}/gmm_ubm_32GBd/features"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# Regex to extract metadata from file names
METADATA_PATTERN = re.compile(
    # Match the prefix (e.g., Song1_X or Song1_Y)
    r"Song\d+_[XY]_"
    # Match OSNR (integer or decimal with 'p')
    r"(?P<OSNR>[\d]+(?:p[\d]+)?)dB_"
    # Match Spacing (integer or decimal with 'p')
    r"(?P<Spacing>[\d]+(?:p[\d]+)?)GHz_"
    # Match Distance (integer or decimal with 'p')
    r"(?P<Distance>[\d]+(?:p[\d]+)?)km_"
    # Match Power (integer or decimal with 'p')
    r"(?P<Power>[\d]+(?:p[\d]+)?)dBm"
)

# ===========================
# Functions
# ===========================


def extract_metadata_from_filename(filename: str) -> dict:
    match = METADATA_PATTERN.search(filename)
    if not match:
        logger.error(f"File name {filename} does not match the expected pattern.")
        raise ValueError(f"File name {filename} does not match the expected pattern.")
    return {
        key: float(value.replace("p", ".")) for key, value in match.groupdict().items()
    }


def load_all_data(path: str) -> pl.DataFrame:
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
    return pl.concat(dfs)


def filter_overlap(df: pl.DataFrame, threshold: float = 35.2) -> pl.DataFrame:
    logger.info(f"Filtering entries with Spectral Overlapping (< {threshold} GHz)")
    return df.filter(pl.col("Spacing") < threshold)


def select_best_gmm(
    X: np.ndarray, max_components: int = 16
) -> tuple[GaussianMixture, list]:
    lowest_bic = np.inf
    best_gmm = None
    bic_scores = []

    logger.info("Evaluating GMM models with different component counts")
    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=n_components, covariance_type="full", random_state=42
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_scores.append(bic)
        logger.debug(f"GMM with {n_components} components: BIC = {bic:.2f}")
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    logger.info(
        f"Best GMM has {best_gmm.n_components} components with BIC = {lowest_bic:.2f}"
    )
    return best_gmm, bic_scores


def plot_bic_curve(bic_scores: list, output_path: str):
    logger.info("Plotting BIC curve")
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(bic_scores) + 1), bic_scores, marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("BIC")
    plt.title("Optimal BIC curve for GMM model selection")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.debug(f"BIC plot saved to {output_path}")


# ===========================
# Main Execution
# ===========================


def main():
    logger.info("Loading data...")
    df_32gbd = load_all_data(Path(DATABASE_DIR) / "Estimation" / "32GBd")

    logger.info("Filtering scenarios with spectral overlapping...")
    overlap_df = filter_overlap(df_32gbd)

    logger.info("Extracing I/Q features...")
    X = overlap_df.select(["I", "Q"]).to_numpy()

    logger.info("Training GMM models...")
    best_gmm, bic_scores = select_best_gmm(X)

    logger.info("Plotting and saving BIC curve...")
    plot_bic_curve(bic_scores, os.path.join(RESULTS_DIR, "gmm_bic_plot.png"))

    logger.info("Saving best GMM and data...")
    gmm_output = {
        "model": best_gmm,
        "features": X,
        "metadata": overlap_df.select(
            ["Distance", "Power", "Spacing", "OSNR"]
        ).to_pandas(),
        "best_n_components": best_gmm.n_components,
        "bic_scores": bic_scores,
    }

    with open(os.path.join(RESULTS_DIR, "gmm_overlap_model.pkl"), "wb") as f:
        pickle.dump(gmm_output, f)

    logger.info(
        f"GMM model trained with {best_gmm.n_components} components (least BIC)."
    )

if __name__ == "__main__":
    main()
