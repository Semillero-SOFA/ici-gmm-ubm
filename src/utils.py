import re
import logging
from pathlib import Path

# ==== Configuración general ====

# Rutas estáticas
DB_PATH = Path("../../databases/Demodulation/Processed")
RESULTS_DIR = Path("../../results/gmm_ubm_16GBd/results")
CHECKPOINT_DIR = Path(RESULTS_DIR.parent / "checkpoints")

# Opciones de muestreo
SAMPLE_SIZES = [1_000, 5_000, 10_000, 20_000, 50_000]
COMPONENTS_LIST = [2**n for n in range(1, 6)]
RANDOM_SEED = 15

# Opciones extra
HDF5_FILENAME = "gmm_ubm_results_overlap.h5"
MAX_BACKUPS = 2
OVERLAP_SPACING_THRESHOLD_16GBD = 17.6  # GHz
OVERLAP_SPACING_THRESHOLD_32GBD = 35.2  # GHz

# ==== Logger ====


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Inicializa y devuelve un logger con formato estándar."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


# ==== Utilidades generales ====


def ensure_dir(path: Path) -> None:
    """Crea un directorio si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def extract_spacing_from_dirname(name: str) -> float:
    """Extrae el valor de spacing desde el nombre del directorio."""
    match = re.search(r"(\d+(\.\d+)?)GHz", name)
    if match:
        return float(match.group(1))
    # Si no encuentra el espaciamiento en el nombre, se asume 50 GHz
    return 50.0


def extract_osnr_from_filename(name: str) -> float:
    """Extrae el OSNR desde el nombre del archivo."""
    match = re.search(r"consY(\d+(?:\.\d+)?)dB", name)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract OSNR from: {name}")


# ==== Inicialización de directorios obligatorios ====

ensure_dir(CHECKPOINT_DIR)
ensure_dir(RESULTS_DIR)
