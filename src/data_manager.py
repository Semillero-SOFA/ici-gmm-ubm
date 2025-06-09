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
    Lee todos los CSV desde la base de datos, añade columnas 'Spacing' y 'OSNR',
    y devuelve una muestra aleatoria de tamaño n_samples.
    """
    logger.info(f"Reading database with n_samples={n_samples}, seed={seed}")
    all_dataframes = []

    for directory in DB_PATH.iterdir():
        # Ignorar el archivo TX
        if not directory.is_dir():
            continue

        # Extraer el espaciamiento espectral del nombre del directorio
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
        logger.err("No data was loaded due to an error.")
        raise ValueError("No data was loaded due to an error.")

    full_df = pl.concat(all_dataframes)
    logger.info(f"Total samples available: {len(full_df)}")

    return full_df


# ========== Checkpoint Management ==========


def _get_backup_filename(base_filename: str, backup_number: int) -> str:
    """
    Genera el nombre del archivo de backup, como "result.h5.bak1"
    """
    if not isinstance(backup_number, int) or backup_number < 1:
        raise ValueError("The backup number must be a positive integer.")
    base, ext = os.path.splitext(base_filename)
    return f"{base}.bak{backup_number}{ext}"


def _rotate_backups(file_path: Path) -> None:
    """
    Realiza una rotación de backups: .bak(N) ← .bak(N-1) ← ... ← original
    """
    if not file_path.exists():
        return  # No hay nada que respaldar

    for i in reversed(range(1, MAX_BACKUPS)):
        src = file_path.parent / _get_backup_filename(file_path.name, i)
        dst = file_path.parent / _get_backup_filename(file_path.name, i + 1)
        if src.exists():
            src.rename(dst)

    # Backup actual ← original
    first_backup = file_path.parent / _get_backup_filename(file_path.name, 1)
    file_path.rename(first_backup)
    logger.info(f"Backup creado: {first_backup.name}")


def save_result_to_hdf5(
    filename: str,
    *,
    n_samples: int,
    n_components: int,
    mean_log_likelihood_overlap: float,
    std_log_likelihood_overlap: float,
    mean_log_likelihood: float,
    std_log_likelihood: float,
) -> None:
    """
    Guarda una fila de resultados en un archivo HDF5 en formato plano/tabular.

    Aplica log rotation si el archivo ya existe antes de escribir.
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
            )
        ],
        dtype=[
            ("n_samples", "i4"),
            ("n_components", "i4"),
            ("mean_log_likelihood_overlap", "f4"),
            ("std_log_likelihood_overlap", "f4"),
            ("mean_log_likelihood", "f4"),
            ("std_log_likelihood", "f4"),
        ],
    )

    with h5py.File(file_path, "a") as f:
        # Check if the 'results' dataset exists *inside* the file
        if "results" in f:
            # --- APPEND PATH ---
            dset = f["results"]
            old_size = dset.shape[0]
            dset.resize((old_size + 1,))
            dset[old_size] = row[0]
            logger.info(f"Appended to existing file: {file_path.name}")
        else:
            # --- CREATE PATH ---
            # The dataset doesn't exist, so create it for the first time.
            f.create_dataset(
                "results",
                data=row,
                maxshape=(None,),  # Allows resizing along the first axis
                chunks=True,  # Enables efficient resizing
            )
            logger.info(f"New results file created: {file_path.name}")


def read_results_from_hdf5(filename: str) -> pl.DataFrame:
    """
    Lee el archivo de resultados HDF5 y devuelve un DataFrame de Polars.
    """
    file_path = RESULTS_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    with h5py.File(file_path, "r") as f:
        data = f["results"][:]

    df = pl.DataFrame(data)
    logger.info(f"Leídas {df.shape[0]} filas desde {file_path.name}")
    return df
