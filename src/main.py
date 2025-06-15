from utils import (
    RESULTS_DIR,
    SAMPLE_SIZES,
    COMPONENTS_LIST,
    RANDOM_SEED,
    HDF5_FILENAME,
    OVERLAP_SPACING_THRESHOLD_16GBD,
    TEST_SIZE,
    get_logger,
)
from gmm_ubm import train_gmm, evaluate_gmm
from data_manager import (
    load_database_16gbd,
    save_result_to_hdf5,
    read_results_from_hdf5,
)
from plotting import (
    plot_likelihood_vs_samples,
    plot_aic_bic_vs_samples,
)

import polars as pl

logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting GMM training and evaluation process.")

    # Attempt to read existing results to continue or avoid re-running
    try:
        results = read_results_from_hdf5(HDF5_FILENAME)
        logger.info(f"Loaded {results.shape[0]} existing results from {HDF5_FILENAME}")
    except FileNotFoundError:
        logger.info(f"No existing results file found: {HDF5_FILENAME}. Starting fresh.")
        # Create an empty DataFrame with the expected schema if no file exists
        results = pl.DataFrame(
            {
                "n_samples": pl.Series(dtype=pl.Int32),
                "n_components": pl.Series(dtype=pl.Int32),
                "mean_log_likelihood_overlap": pl.Series(dtype=pl.Float32),
                "std_log_likelihood_overlap": pl.Series(dtype=pl.Float32),
                "mean_log_likelihood": pl.Series(dtype=pl.Float32),
                "std_log_likelihood": pl.Series(dtype=pl.Float32),
                "aic": pl.Series(dtype=pl.Float32),
                "bic": pl.Series(dtype=pl.Float32),
            }
        )

    for n_samples in SAMPLE_SIZES:
        logger.info(
            f"Loading training and test data for n_samples = {n_samples}, test_size = {TEST_SIZE}"
        )
        train_df, test_df = load_database_16gbd(
            n_samples=n_samples, test_size=TEST_SIZE, seed=RANDOM_SEED
        )

        feature_cols = train_df.columns[0:2]

        X_overlap_train = (
            train_df.filter(pl.col("Spacing") <= OVERLAP_SPACING_THRESHOLD_16GBD)
            .select(feature_cols)
            .to_numpy()
        )
        X_overlap_test = (
            test_df.filter(pl.col("Spacing") <= OVERLAP_SPACING_THRESHOLD_16GBD)
            .select(feature_cols)
            .to_numpy()
        )
        X_non_overlap_test = (
            test_df.filter(pl.col("Spacing") > OVERLAP_SPACING_THRESHOLD_16GBD)
            .select(feature_cols)
            .to_numpy()
        )

        # Log shapes of each subset
        logger.info(
            f"Sample sizes after filtering: "
            f"X_overlap_train = {X_overlap_train.shape}, "
            f"X_overlap_test = {X_overlap_test.shape}, "
            f"X_non_overlap_test = {X_non_overlap_test.shape}"
        )

        # Debug unique scenarios
        n_overlap_train_scenarios = (
            train_df.filter(pl.col("Spacing") <= OVERLAP_SPACING_THRESHOLD_16GBD)
            .select(["Spacing", "OSNR"])
            .unique()
            .height
        )
        n_overlap_test_scenarios = (
            test_df.filter(pl.col("Spacing") <= OVERLAP_SPACING_THRESHOLD_16GBD)
            .select(["Spacing", "OSNR"])
            .unique()
            .height
        )
        n_non_overlap_test_scenarios = (
            test_df.filter(pl.col("Spacing") > OVERLAP_SPACING_THRESHOLD_16GBD)
            .select(["Spacing", "OSNR"])
            .unique()
            .height
        )

        logger.debug(
            f"Unique scenarios per set: "
            f"overlap_train = {n_overlap_train_scenarios}, "
            f"overlap_test = {n_overlap_test_scenarios}, "
            f"non_overlap_test = {n_non_overlap_test_scenarios}"
        )

        for n_components in COMPONENTS_LIST:
            logger.info(f"Training GMM with n_components = {n_components}")

            if (
                "n_samples" in results.columns
                and "n_components" in results.columns
                and not (
                    results.filter(pl.col("n_samples") == n_samples)
                    .filter(pl.col("n_components") == n_components)
                    .is_empty()
                )
            ):
                logger.info("Results already exist for this combination. Skipping...")
                continue

            gmm_overlap = train_gmm(
                X_overlap_train, n_components=n_components, seed=RANDOM_SEED
            )

            mean_ll_overlap, std_ll_overlap, aic_overlap, bic_overlap = evaluate_gmm(
                gmm_overlap, X_overlap_test
            )

            (
                mean_ll_non_overlap,
                std_ll_non_overlap,
                aic_non_overlap_eval,
                bic_non_overlap_eval,
            ) = evaluate_gmm(gmm_overlap, X_non_overlap_test)

            logger.info(
                f"Log-Likelihood (non-overlap): mean = {mean_ll_non_overlap:.3f}, std = {std_ll_non_overlap:.3f}, "
                f"AIC = {aic_non_overlap_eval:.2f}, BIC = {bic_non_overlap_eval:.2f}"
            )
            logger.info(
                f"Log-Likelihood (overlap): mean = {mean_ll_overlap:.3f}, std = {std_ll_overlap:.3f}, "
                f"AIC = {aic_overlap:.2f}, BIC = {bic_overlap:.2f}"
            )

            save_result_to_hdf5(
                HDF5_FILENAME,
                n_samples=n_samples,
                n_components=n_components,
                mean_log_likelihood_overlap=mean_ll_overlap,
                std_log_likelihood_overlap=std_ll_overlap,
                mean_log_likelihood=mean_ll_non_overlap,
                std_log_likelihood=std_ll_non_overlap,
                aic=aic_overlap,
                bic=bic_overlap,
            )

            results = read_results_from_hdf5(HDF5_FILENAME)

    logger.info("Process completed successfully.")

    # --- Plotting Results ---
    # Ensure all latest results are loaded before plotting
    final_results_df = read_results_from_hdf5(HDF5_FILENAME)

    # Plot Mean Log-Likelihood vs. Number of Samples
    plot_likelihood_vs_samples(
        final_results_df, RESULTS_DIR / "likelihood_vs_samples.svg"
    )
    logger.info(f"Saved likelihood plot to {RESULTS_DIR / 'likelihood_vs_samples.svg'}")

    # Plot AIC and BIC vs. Number of Samples
    plot_aic_bic_vs_samples(final_results_df, RESULTS_DIR / "aic_bic_vs_samples.svg")
    logger.info(f"Saved AIC/BIC plot to {RESULTS_DIR / 'aic_bic_vs_samples.svg'}")


if __name__ == "__main__":
    main()
