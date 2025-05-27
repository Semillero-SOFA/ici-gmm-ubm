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
from sklearn.model_selection import train_test_split

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


def train_test_split_data(df: pl.DataFrame, test_size: float = 0.1, random_state: int = 42):
    """
    Split the data into training and test sets.
    
    Args:
        df: Polars DataFrame with the data
        test_size: Proportion of data for testing (default 0.1 for 10%)
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (train_df, test_df, X_train, X_test)
    """
    logger.info(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    
    # Convert to numpy for sklearn train_test_split
    X = df.select(["I", "Q"]).to_numpy()
    metadata = df.select(["Distance", "Power", "Spacing", "OSNR"]).to_numpy()
    
    # Split the data
    X_train, X_test, meta_train, meta_test = train_test_split(
        X, metadata, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # Convert back to Polars DataFrames
    train_df = pl.DataFrame({
        "I": X_train[:, 0],
        "Q": X_train[:, 1],
        "Distance": meta_train[:, 0],
        "Power": meta_train[:, 1],
        "Spacing": meta_train[:, 2],
        "OSNR": meta_train[:, 3],
    })
    
    test_df = pl.DataFrame({
        "I": X_test[:, 0],
        "Q": X_test[:, 1],
        "Distance": meta_test[:, 0],
        "Power": meta_test[:, 1],
        "Spacing": meta_test[:, 2],
        "OSNR": meta_test[:, 3],
    })
    
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return train_df, test_df, X_train, X_test


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


def calculate_likelihood_statistics(gmm: GaussianMixture, X_test: np.ndarray) -> dict:
    """
    Calculate likelihood statistics for the test set.
    
    Args:
        gmm: Trained Gaussian Mixture Model
        X_test: Test data
    
    Returns:
        dict: Dictionary with likelihood statistics
    """
    logger.info("Calculating likelihood statistics on test set...")
    
    # Calculate log-likelihood for each sample
    log_likelihoods = gmm.score_samples(X_test)
    
    # Calculate total log-likelihood
    total_log_likelihood = gmm.score(X_test)
    
    # Calculate statistics
    stats = {
        "total_log_likelihood": total_log_likelihood,
        "mean_log_likelihood": np.mean(log_likelihoods),
        "std_log_likelihood": np.std(log_likelihoods),
        "min_log_likelihood": np.min(log_likelihoods),
        "max_log_likelihood": np.max(log_likelihoods),
        "median_log_likelihood": np.median(log_likelihoods),
        "n_test_samples": len(X_test),
        "individual_log_likelihoods": log_likelihoods
    }
    
    logger.info(f"Test set likelihood statistics:")
    logger.info(f"  Total log-likelihood: {stats['total_log_likelihood']:.4f}")
    logger.info(f"  Mean log-likelihood: {stats['mean_log_likelihood']:.4f}")
    logger.info(f"  Std log-likelihood: {stats['std_log_likelihood']:.4f}")
    logger.info(f"  Min log-likelihood: {stats['min_log_likelihood']:.4f}")
    logger.info(f"  Max log-likelihood: {stats['max_log_likelihood']:.4f}")
    logger.info(f"  Median log-likelihood: {stats['median_log_likelihood']:.4f}")
    
    return stats


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


def plot_likelihood_distribution(log_likelihoods: np.ndarray, output_path: str):
    """
    Plot the distribution of log-likelihoods for the test set.
    """
    logger.info("Plotting likelihood distribution")
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(log_likelihoods, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel("Log-likelihood")
    plt.ylabel("Frequency")
    plt.title("Distribution of Log-likelihoods (Test Set)")
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(log_likelihoods, vert=True)
    plt.ylabel("Log-likelihood")
    plt.title("Box Plot of Log-likelihoods")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.debug(f"Likelihood distribution plot saved to {output_path}")


# ===========================
# Main Execution
# ===========================


def main():
    logger.info("Loading data...")
    df_32gbd = load_all_data(Path(DATABASE_DIR) / "Estimation" / "32GBd")

    logger.info("Filtering scenarios with spectral overlapping...")
    overlap_df = filter_overlap(df_32gbd)

    logger.info("Splitting data into train/test sets...")
    train_df, test_df, X_train, X_test = train_test_split_data(overlap_df)

    logger.info("Training GMM models on training set...")
    best_gmm, bic_scores = select_best_gmm(X_train)

    logger.info("Calculating likelihood on test set...")
    likelihood_stats = calculate_likelihood_statistics(best_gmm, X_test)

    logger.info("Plotting and saving BIC curve...")
    plot_bic_curve(bic_scores, os.path.join(RESULTS_DIR, "gmm_bic_plot.png"))

    logger.info("Plotting likelihood distribution...")
    plot_likelihood_distribution(
        likelihood_stats["individual_log_likelihoods"], 
        os.path.join(RESULTS_DIR, "likelihood_distribution.png")
    )

    logger.info("Saving best GMM, data, and results...")
    gmm_output = {
        "model": best_gmm,
        "train_features": X_train,
        "test_features": X_test,
        "train_metadata": train_df.select(["Distance", "Power", "Spacing", "OSNR"]).to_pandas(),
        "test_metadata": test_df.select(["Distance", "Power", "Spacing", "OSNR"]).to_pandas(),
        "best_n_components": best_gmm.n_components,
        "bic_scores": bic_scores,
        "likelihood_stats": likelihood_stats,
    }

    with open(os.path.join(RESULTS_DIR, "gmm_overlap_model.pkl"), "wb") as f:
        pickle.dump(gmm_output, f)

    logger.info(f"GMM model trained with {best_gmm.n_components} components (least BIC).")
    logger.info(f"Test set mean log-likelihood: {likelihood_stats['mean_log_likelihood']:.4f}")
    
    # Interpretation of results
    if likelihood_stats['mean_log_likelihood'] > -10:  # Threshold can be adjusted
        logger.info("✓ High likelihood detected - GMM adapted well to the scenario")
    else:
        logger.info("⚠ Low likelihood detected - Consider model adjustment")


if __name__ == "__main__":
    main()
