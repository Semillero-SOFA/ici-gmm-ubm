from utils import (
    RESULTS_DIR,
    SAMPLE_SIZES,
    COMPONENTS_LIST,
    RANDOM_SEED,
    HDF5_FILENAME,
    OVERLAP_SPACING_THRESHOLD_16GBD,
    get_logger,
)
from gmm_ubm import train_gmm, evaluate_log_likelihood
from data_manager import (
    load_database_16gbd,
    save_result_to_hdf5,
    read_results_from_hdf5,
)
from plotting import plot_likelihood_vs_samples, plot_aic_bic_vs_components # Import the new plotting function

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
        results = pl.DataFrame({
            "n_samples": pl.Series(dtype=pl.Int32),
            "n_components": pl.Series(dtype=pl.Int32),
            "mean_log_likelihood_overlap": pl.Series(dtype=pl.Float32),
            "std_log_likelihood_overlap": pl.Series(dtype=pl.Float32),
            "mean_log_likelihood": pl.Series(dtype=pl.Float32),
            "std_log_likelihood": pl.Series(dtype=pl.Float32),
            "aic": pl.Series(dtype=pl.Float32),
            "bic": pl.Series(dtype=pl.Float32),
        })

    for n_samples in SAMPLE_SIZES:
        logger.info(f"Preparing data for n_samples = {n_samples}")
        # Load the full dataset (it samples internally to n_samples)
        X_full_df = load_database_16gbd(n_samples=n_samples, seed=RANDOM_SEED)

        # Assuming features are the first two columns (e.g., 'I', 'Q')
        # ***IMPORTANT: Adjust 'feature_cols' if your actual feature columns are different.***
        feature_cols = X_full_df.columns[0:2] 

        X_overlap_data = X_full_df.filter(
            pl.col("Spacing") < OVERLAP_SPACING_THRESHOLD_16GBD
        ).select(feature_cols).to_numpy()

        # Split into training and test sets for overlap data
        # TODO: Use the big dataset to get the train and test data
        # Instead of splitting the subdataset into train and test
        train_size_overlap = int(0.9 * len(X_overlap_data))
        X_overlap_train = X_overlap_data[:train_size_overlap]
        X_overlap_test = X_overlap_data[train_size_overlap:]

        X_non_overlap_data = X_full_df.filter(
            pl.col("Spacing") >= OVERLAP_SPACING_THRESHOLD_16GBD
        ).select(feature_cols).to_numpy()
        
        # Ensure X_non_overlap_test has a consistent size for comparison if needed
        # (e.g., same size as X_overlap_test, or a fixed ratio)
        X_non_overlap_test = X_non_overlap_data[: X_overlap_test.shape[0]]


        for n_components in COMPONENTS_LIST:
            logger.info(f"Training GMM with n_components = {n_components}")

            # Skip if results for this n_samples and n_components already exist
            if (
                "n_samples" in results.columns and "n_components" in results.columns and
                not (
                    results.filter(pl.col("n_samples") == n_samples)
                    .filter(pl.col("n_components") == n_components)
                    .is_empty()
                )
            ):
                logger.info("Results already exist for this combination. Skipping...")
                continue

            # Train GMM on the OVERLAP training data
            gmm_overlap = train_gmm(
                X_overlap_train, n_components=n_components, seed=RANDOM_SEED
            )
            
            # Evaluate the OVERLAP-trained GMM on the OVERLAP test set
            mean_ll_overlap, std_ll_overlap, aic_overlap, bic_overlap = evaluate_log_likelihood(
                gmm_overlap, X_overlap_test
            )

            # Evaluate the *same OVERLAP-trained* GMM on the NON-OVERLAP test set
            # This assesses how well a model trained for overlap performs on non-overlap data.
            mean_ll_non_overlap, std_ll_non_overlap, aic_non_overlap_eval, bic_non_overlap_eval = evaluate_log_likelihood(
                gmm_overlap, X_non_overlap_test
            )
            # Note: aic_non_overlap_eval and bic_non_overlap_eval are AIC/BIC *calculated for the OVERLAP model
            # when evaluated on NON-OVERLAP data*. For saving, we're still saving the 'model's' AIC/BIC (`aic_overlap`/`bic_overlap`).

            logger.info(
                f"Log-Likelihood (non-overlap eval with overlap model): mean = {mean_ll_non_overlap:.3f}, std = {std_ll_non_overlap:.3f}, "
                f"AIC = {aic_non_overlap_eval:.2f}, BIC = {bic_non_overlap_eval:.2f}"
            )
            logger.info(
                f"Log-Likelihood (overlap eval with overlap model): mean = {mean_ll_overlap:.3f}, std = {std_ll_overlap:.3f}, "
                f"AIC = {aic_overlap:.2f}, BIC = {bic_overlap:.2f}"
            )

            # Save results to HDF5
            # AIC and BIC saved here (`aic_overlap`, `bic_overlap`) are for the model trained on X_overlap_train.
            save_result_to_hdf5(
                HDF5_FILENAME,
                n_samples=n_samples,
                n_components=n_components,
                mean_log_likelihood_overlap=mean_ll_overlap,
                std_log_likelihood_overlap=std_ll_overlap,
                mean_log_likelihood=mean_ll_non_overlap, # This is the likelihood of OVERLAP-trained model on NON-OVERLAP data
                std_log_likelihood=std_ll_non_overlap,   # This is the std of OVERLAP-trained model on NON-OVERLAP data
                aic=aic_overlap, # AIC for the GMM *trained* on OVERLAP data
                bic=bic_overlap, # BIC for the GMM *trained* on OVERLAP data
            )
            
            # Re-read results to ensure the 'results' DataFrame is up-to-date for the next iteration's skip logic
            results = read_results_from_hdf5(HDF5_FILENAME)


    logger.info("Process completed successfully.")

    # --- Plotting Results ---
    # Ensure all latest results are loaded before plotting
    final_results_df = read_results_from_hdf5(HDF5_FILENAME)

    # Plot Mean Log-Likelihood vs. Number of Samples
    plot_likelihood_vs_samples(final_results_df, RESULTS_DIR / "likelihood_vs_samples.svg")
    logger.info(f"Saved likelihood plot to {RESULTS_DIR / 'likelihood_vs_samples.svg'}")

    # Plot AIC and BIC vs. Number of Components
    plot_aic_bic_vs_components(final_results_df, RESULTS_DIR / "aic_bic_vs_components.svg")
    logger.info(f"Saved AIC/BIC plot to {RESULTS_DIR / 'aic_bic_vs_components.svg'}")


if __name__ == "__main__":
    main()
