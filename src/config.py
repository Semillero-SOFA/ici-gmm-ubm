import os
import re
import subprocess
import logging
from pathlib import Path


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
        print(f"Git error: {result.stderr.strip()}")
        return None

    except Exception as e:
        # Handle exceptions, e.g., subprocess.CalledProcessError
        print(f"Error finding root: {str(e)}")
        return None


def setup_logger(name: str) -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(f"{name}.log"),
        ],
    )
    return logging.getLogger(name)


# ===========================
# Global Configuration
# ===========================

# File and directory paths
LOCAL_ROOT = find_root()
GLOBAL_ROOT = LOCAL_ROOT.parent if LOCAL_ROOT else Path.cwd()
DATABASE_DIR = f"{GLOBAL_ROOT}/databases"
GLOBAL_RESULTS_DIR = f"{GLOBAL_ROOT}/results"
RESULTS_DIR = f"{GLOBAL_RESULTS_DIR}/gmm_ubm_32GBd/features"

# Create results directory if it doesn't exist
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# HDF5 checkpoint file
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "gmm_training_checkpoint.h5")

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

# Default parameters
DEFAULT_TEST_SIZE = 0.1
DEFAULT_RANDOM_STATE = 42
DEFAULT_OVERLAP_THRESHOLD = 35.2
DEFAULT_MAX_GMM_COMPONENTS = 16
