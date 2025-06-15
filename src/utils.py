import re
import logging
from pathlib import Path

# ==== General Configuration ====

# Static Paths
DB_PATH = Path("../../databases/Demodulation/Processed")
RESULTS_DIR = Path("../../results/gmm_ubm_16GBd")
LOGS_DIR = Path(RESULTS_DIR / "logs")

# Sampling Options
SAMPLE_SIZES = [1_000, 5_000, 10_000, 20_000, 50_000]
TEST_SIZE = 500  # Samples per scenario
COMPONENTS_LIST = [2**n for n in range(1, 7)]
RANDOM_SEED = 15

# Extra Options
HDF5_FILENAME = "gmm_ubm_results_overlap.h5"
MAX_BACKUPS = 2
OVERLAP_SPACING_THRESHOLD_16GBD = 17.6  # GHz
OVERLAP_SPACING_THRESHOLD_32GBD = 35.2  # GHz


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Initializes and returns a logger with both console and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # --- Console handler ---
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # --- File handler ---
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / f"{name}.log"
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.info(f"Logger initialized. Log file: {log_file}")

    return logger


def ensure_dir(path: Path) -> None:
    """Creates a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def extract_spacing_from_dirname(name: str) -> float:
    """Extracts the spacing value from the directory name."""
    match = re.search(r"(\d+(\.\d+)?)GHz", name)
    if match:
        return float(match.group(1))
    # If spacing is not found in the name, assume 50 GHz as a default
    return 50.0


def extract_osnr_from_filename(name: str) -> float:
    """Extracts the OSNR value from the file name."""
    match = re.search(r"consY(\d+(?:\.\d+)?)dB", name)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract OSNR from: {name}")


# ==== Mandatory Directory Initialization ====

ensure_dir(RESULTS_DIR)
