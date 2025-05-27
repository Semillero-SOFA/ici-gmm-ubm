import logging
import numpy as np
from sklearn.mixture import GaussianMixture

from config import DEFAULT_MAX_GMM_COMPONENTS
from checkpoint_manager import get_completed_steps, load_checkpoint, save_checkpoint


def get_logger():
    return logging.getLogger(__name__)


def select_best_gmm(
    X: np.ndarray, max_components: int = DEFAULT_MAX_GMM_COMPONENTS
) -> tuple[GaussianMixture, list]:
    """
    Train GMM models with different component counts and select the best one based on BIC.

    Args:
        X: Training data
        max_components: Maximum number of components to test

    Returns:
        Tuple of (best_gmm_model, bic_scores_list)
    """
    logger = get_logger()

    # Check if GMM training is already completed
    completed_steps = get_completed_steps()
    if "gmm_training" in completed_steps:
        logger.info("GMM training step found in checkpoint, loading from checkpoint...")
        checkpoint_data = load_checkpoint("gmm_training")
        return checkpoint_data["gmm_model"], checkpoint_data["bic_scores"]

    # Check if partial GMM training exists
    checkpoint_data = load_checkpoint("gmm_training_partial")
    if checkpoint_data:
        logger.info("Resuming GMM training from checkpoint...")
        completed_components = checkpoint_data.get("completed_components", 0)
        bic_scores = checkpoint_data.get("bic_scores", [])
        best_gmm = checkpoint_data.get("gmm_model", None)
        lowest_bic = checkpoint_data.get("lowest_bic", np.inf)
    else:
        logger.info("Starting GMM training from scratch...")
        completed_components = 0
        bic_scores = []
        best_gmm = None
        lowest_bic = np.inf

    logger.info("Evaluating GMM models with different component counts")

    for n_components in range(completed_components + 1, max_components + 1):
        logger.info(f"Training GMM with {n_components} components...")

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

        # Save partial checkpoint after each component
        save_checkpoint(
            {
                "completed_components": n_components,
                "bic_scores": bic_scores,
                "gmm_model": best_gmm,
                "lowest_bic": lowest_bic,
                "current_n_components": n_components,
                "current_bic": bic,
            },
            "gmm_training_partial",
        )

        logger.info(
            f"Completed {n_components}/{max_components} components, current best BIC: {lowest_bic:.2f}"
        )

    # Save final checkpoint
    save_checkpoint(
        {
            "gmm_model": best_gmm,
            "bic_scores": bic_scores,
            "best_n_components": best_gmm.n_components,
            "final_bic": lowest_bic,
            "max_components_tested": max_components,
        },
        "gmm_training",
    )

    logger.info(
        f"Best GMM has {best_gmm.n_components} components with BIC = {lowest_bic:.2f}"
    )
    return best_gmm, bic_scores


def calculate_likelihood_statistics(gmm: GaussianMixture, X_test: np.ndarray) -> dict:
    """
    Calculate likelihood statistics for the test set.

    Args:
        gmm: Trained GMM model
        X_test: Test data

    Returns:
        Dictionary with likelihood statistics
    """
    logger = get_logger()

    # Check if likelihood calculation is already completed
    completed_steps = get_completed_steps()
    if "likelihood_calculation" in completed_steps:
        logger.info(
            "Likelihood calculation step found in checkpoint, loading from checkpoint..."
        )
        checkpoint_data = load_checkpoint("likelihood_calculation")
        return checkpoint_data

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
        "individual_log_likelihoods": log_likelihoods,
    }

    logger.info(f"Test set likelihood statistics:")
    logger.info(f"  Total log-likelihood: {stats['total_log_likelihood']:.4f}")
    logger.info(f"  Mean log-likelihood: {stats['mean_log_likelihood']:.4f}")
    logger.info(f"  Std log-likelihood: {stats['std_log_likelihood']:.4f}")
    logger.info(f"  Min log-likelihood: {stats['min_log_likelihood']:.4f}")
    logger.info(f"  Max log-likelihood: {stats['max_log_likelihood']:.4f}")
    logger.info(f"  Median log-likelihood: {stats['median_log_likelihood']:.4f}")

    # Save checkpoint
    save_checkpoint(stats, "likelihood_calculation")

    return stats
