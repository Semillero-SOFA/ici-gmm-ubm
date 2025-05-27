import os
import pickle
import logging
from sklearn.mixture import GaussianMixture
import numpy as np
import polars as pl

from config import RESULTS_DIR


def get_logger():
    return logging.getLogger(__name__)


def save_model(
    gmm_model: GaussianMixture,
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_metadata: pl.DataFrame,
    test_metadata: pl.DataFrame,
    bic_scores: list,
    likelihood_stats: dict,
    filename: str = "gmm_overlap_model.pkl",
) -> str:
    """
    Save the complete GMM model and associated data.

    Args:
        gmm_model: Trained GMM model
        X_train: Training features
        X_test: Test features
        train_metadata: Training metadata
        test_metadata: Test metadata
        bic_scores: BIC scores from model selection
        likelihood_stats: Likelihood statistics
        filename: Output filename

    Returns:
        Path to saved model file
    """
    logger = get_logger()

    gmm_output = {
        "model": gmm_model,
        "train_features": X_train,
        "test_features": X_test,
        "train_metadata": train_metadata,
        "test_metadata": test_metadata,
        "best_n_components": gmm_model.n_components,
        "bic_scores": bic_scores,
        "likelihood_stats": likelihood_stats,
    }

    output_path = os.path.join(RESULTS_DIR, filename)

    with open(output_path, "wb") as f:
        pickle.dump(gmm_output, f)

    logger.info(f"Model saved to: {output_path}")
    return output_path


def load_model(filename: str = "gmm_overlap_model.pkl") -> dict:
    """
    Load a saved GMM model and associated data.

    Args:
        filename: Model filename to load

    Returns:
        Dictionary with loaded model and data
    """
    logger = get_logger()

    model_path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    logger.info(f"Model loaded from: {model_path}")
    return model_data


def evaluate_model_performance(gmm_model: GaussianMixture, X_test: np.ndarray) -> dict:
    """
    Evaluate GMM model performance on test data.

    Args:
        gmm_model: Trained GMM model
        X_test: Test data

    Returns:
        Dictionary with performance metrics
    """
    logger = get_logger()

    # Calculate various performance metrics
    log_likelihood = gmm_model.score(X_test)
    sample_log_likelihoods = gmm_model.score_samples(X_test)

    # Calculate AIC and BIC
    n_params = (
        gmm_model.n_components
        * (
            X_test.shape[1]  # means
            + X_test.shape[1]
            * (X_test.shape[1] + 1)
            // 2  # covariances (assuming full)
        )
        + gmm_model.n_components
        - 1
    )  # weights

    aic = -2 * log_likelihood * X_test.shape[0] + 2 * n_params
    bic = gmm_model.bic(X_test)

    performance = {
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "n_components": gmm_model.n_components,
        "n_parameters": n_params,
        "converged": gmm_model.converged_,
        "mean_sample_log_likelihood": np.mean(sample_log_likelihoods),
        "std_sample_log_likelihood": np.std(sample_log_likelihoods),
    }

    logger.info("Model performance evaluation:")
    for key, value in performance.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    return performance


def predict_probabilities(gmm_model: GaussianMixture, X: np.ndarray) -> np.ndarray:
    """
    Predict component probabilities for input data.

    Args:
        gmm_model: Trained GMM model
        X: Input data

    Returns:
        Array of component probabilities
    """
    return gmm_model.predict_proba(X)


def predict_components(gmm_model: GaussianMixture, X: np.ndarray) -> np.ndarray:
    """
    Predict most likely component for input data.

    Args:
        gmm_model: Trained GMM model
        X: Input data

    Returns:
        Array of predicted components
    """
    return gmm_model.predict(X)


def get_model_summary(gmm_model: GaussianMixture) -> dict:
    """
    Get a summary of the GMM model parameters.

    Args:
        gmm_model: Trained GMM model

    Returns:
        Dictionary with model summary
    """
    summary = {
        "n_components": gmm_model.n_components,
        "covariance_type": gmm_model.covariance_type,
        "converged": gmm_model.converged_,
        "n_iter": getattr(gmm_model, "n_iter_", "Unknown"),
        "weights": gmm_model.weights_.tolist(),
        "means_shape": gmm_model.means_.shape,
        "covariances_shape": gmm_model.covariances_.shape,
    }

    return summary
