from utils import (
    SAMPLE_SIZES,
    COMPONENTS_LIST,
    RANDOM_SEED,
    HDF5_FILENAME,
    OVERLAP_SPACING_THRESHOLD_16GBD,
)
from gmm_ubm import train_gmm, evaluate_log_likelihood
from data_manager import load_database_16gbd, save_result_to_hdf5
from utils import get_logger

import polars as pl

logger = get_logger(__name__)


def main() -> None:
    logger.info("Inicio del proceso de entrenamiento GMM y evaluación")

    for n_samples in SAMPLE_SIZES:
        logger.info(f"Preparing data for n_samples = {n_samples}")
        X = load_database_16gbd(n_samples=n_samples, seed=RANDOM_SEED)

        X_overlap = X.filter(
            pl.col("Spacing") < OVERLAP_SPACING_THRESHOLD_16GBD
        ).to_numpy()
        train_size = int(0.9 * len(X_overlap))
        X_overlap_train = X_overlap[:train_size]
        X_overlap_test = X_overlap[train_size + 1:]

        X_non_overlap = X.filter(
            pl.col("Spacing") >= OVERLAP_SPACING_THRESHOLD_16GBD
        ).to_numpy()
        X_non_overlap = X_non_overlap[: X_overlap_test.shape[0]]

        for n_components in COMPONENTS_LIST:
            logger.info(f"Entrenando GMM con n_components = {n_components}")

            gmm = train_gmm(
                X_overlap_train, n_components=n_components, seed=RANDOM_SEED
            )

            mean_ll, std_ll = evaluate_log_likelihood(gmm, X_non_overlap)
            mean_ll_overlap, std_ll_overlap = evaluate_log_likelihood(
                gmm, X_overlap_test
            )

            logger.info(
                f"Log-Likelihood (non-overlap): mean = {mean_ll:.3f}, std = {std_ll:.3f}"
            )
            logger.info(
                f"Log-Likelihood (overlap): mean = {mean_ll_overlap:.3f}, std = {std_ll_overlap:.3f}"
            )

            save_result_to_hdf5(
                HDF5_FILENAME,
                n_samples=n_samples,
                n_components=n_components,
                mean_log_likelihood_overlap=mean_ll_overlap,
                std_log_likelihood_overlap=std_ll_overlap,
                mean_log_likelihood=mean_ll,
                std_log_likelihood=std_ll,
            )

    logger.info("Proceso finalizado correctamente")


if __name__ == "__main__":
    main()
