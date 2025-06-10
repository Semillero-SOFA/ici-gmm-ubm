import os
import polars as pl
import h5py
import numpy as np

from pathlib import Path

from utils import (
    DB_PATH,
    RESULTS_DIR,
    MAX_BACKUPS,
    extract_spacing_from_dirname,
    extract_osnr_from_filename,
    get_logger,
)

logger = get_logger(__name__)


def load_database_16gbd(n_samples: int, seed: int = 15) -> pl.DataFrame:
    """
    Reads all CSVs from the database, adds 'Spacing' and 'OSNR' columns,
    and returns a random sample of size n_samples.
    """
    logger.info(f"Reading database with n_samples={n_samples}, seed={seed}")
    all_dataframes = []

    for directory in DB_PATH.iterdir():
        # Ignore the TX file
        if not directory.is_dir():
            continue

        # Extract spectral spacing from the directory name
        try:
            spacing = extract_spacing_from_dirname(directory.name)

        except Exception as e:
            logger.warning(f"Could not extract spacing from {directory.name}: {e}")
            continue

        for file in directory.glob("*.csv"):
            try:
                osnr = extract_osnr_from_filename(file.name)
                df = pl.read_csv(file)
                df = df.with_columns(
                    [
                        pl.lit(osnr).alias("OSNR"),
                        pl.lit(spacing).alias("Spacing"),
                    ]
                )
                all_dataframes.append(df.sample(n=n_samples, seed=seed, shuffle=True))
            except Exception as e:
                logger.warning(f"Error while processing {file}: {e}")
                continue

    if not all_dataframes:
        logger.error("No data was loaded due to an error.")
        raise ValueError("No data was loaded due to an error.")

    full_df = pl.concat(all_dataframes)
    logger.info(f"Total samples available: {len(full_df)}")

    return full_df


# ========== Checkpoint Management ==========


def _get_backup_filename(base_filename: str, backup_number: int) -> str:
    """
    Generates the backup file name, such as "result.h5.bak1".
    """
    if not isinstance(backup_number, int) or backup_number < 1:
        raise ValueError("The backup number must be a positive integer.")
    base, ext = os.path.splitext(base_filename)
    return f"{base}.bak{backup_number}{ext}"


def _rotate_backups(file_path: Path) -> None:
    """
    Performs backup rotation: .bak(N) ← .bak(N-1) ← ... ← original.
    """
    if not file_path.exists():
        return  # Nothing to back up

    for i in reversed(range(1, MAX_BACKUPS)):
        src = file_path.parent / _get_backup_filename(file_path.name, i)
        dst = file_path.parent / _get_backup_filename(file_path.name, i + 1)
        if src.exists():
            src.rename(dst)

    # Current backup ← original
    first_backup = file_path.parent / _get_backup_filename(file_path.name, 1)
    file_path.rename(first_backup)
    logger.info(f"Backup created: {first_backup.name}")


def save_result_to_hdf5(
    filename: str,
    *,
    n_samples: int,
    n_components: int,
    mean_log_likelihood_overlap: float,
    std_log_likelihood_overlap: float,
    mean_log_likelihood: float,
    std_log_likelihood: float,
    aic: float,  # New parameter
    bic: float,  # New parameter
) -> None:
    """
    Saves a row of results to an HDF5 file in a flat/tabular format.

    Applies log rotation if the file already exists before writing.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RESULTS_DIR / filename

    row = np.array(
        [
            (
                n_samples,
                n_components,
                mean_log_likelihood_overlap,
                std_log_likelihood_overlap,
                mean_log_likelihood,
                std_log_likelihood,
                aic,
                bic,
            )
        ],
        dtype=[
            ("n_samples", "i4"),
            ("n_components", "i4"),
            ("mean_log_likelihood_overlap", "f4"),
            ("std_log_likelihood_overlap", "f4"),
            ("mean_log_likelihood", "f4"),
            ("std_log_likelihood", "f4"),
            ("aic", "f4"),
            ("bic", "f4"),
        ],
    )

    with h5py.File(file_path, "a") as f:
        # Check if the 'results' dataset exists *inside* the file
        if "results" in f:
            # --- APPEND DATA ---
            dset = f["results"]
            old_size = dset.shape[0]
            dset.resize((old_size + 1,))
            dset[old_size] = row[0]
            logger.info(f"Appended to existing file: {file_path.name}")
        else:
            # --- CREATE DATASET ---
            # The dataset doesn't exist, so create it for the first time.
            f.create_dataset(
                "results",
                data=row,
                maxshape=(None,),
                chunks=True,
            )
            logger.info(f"New results file created: {file_path.name}")


def read_results_from_hdf5(filename: str) -> pl.DataFrame:
    """
    Reads the HDF5 results file and returns a Polars DataFrame.
    The DataFrame will now include 'aic' and 'bic' columns.
    """
    file_path = RESULTS_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        data = f["results"][:]

    df = pl.DataFrame(data)
    logger.info(f"Read {df.shape[0]} rows from {file_path.name}")
    return df
