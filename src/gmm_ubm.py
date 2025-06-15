import numpy as np
from sklearn.mixture import GaussianMixture
import math

from utils import get_logger

logger = get_logger(__name__)


def _get_n_parameters(n_components: int, n_features: int, cov_type: str) -> int:
    """
    Calculates the number of free parameters in a GMM model for AIC/BIC.

    Parameters:
    - n_components: Number of Gaussian components (M)
    - n_features: Dimensionality of the data (D)
    - cov_type: Type of covariance ('full', 'tied', 'diag', 'spherical')

    Returns:
    - Number of parameters (k)
    """
    M = n_components
    D = n_features

    # Parameters for means: M * D
    k_means = M * D

    # Parameters for mixing weights: M - 1 (since they sum to 1)
    k_weights = M - 1

    # Parameters for covariances: depends on cov_type
    if cov_type == "full":
        # Each component has D * (D + 1) / 2 unique parameters (upper triangle of symmetric matrix)
        k_covariances = M * (D * (D + 1) // 2)
    elif cov_type == "tied":
        # One shared D * (D + 1) / 2 unique parameters for all components
        k_covariances = D * (D + 1) // 2
    elif cov_type == "diag":
        # Each component has D parameters (diagonal elements)
        k_covariances = M * D
    elif cov_type == "spherical":
        # Each component has 1 parameter (a single variance)
        k_covariances = M * 1
    else:
        # Raise an error for an unknown covariance type to prevent incorrect calculations
        raise ValueError(f"Unknown covariance_type: {cov_type}")

    k = k_means + k_covariances + k_weights

    # Ensure k is at least 1, though for any reasonable GMM it will be larger
    return max(1, k)


def train_gmm(
    X: np.ndarray, n_components: int, cov_type: str = "full", seed: int = 15
) -> GaussianMixture:
    """
    Trains a Gaussian Mixture Model with the given data X.

    Parameters:
    - X: Input data (n_samples, n_features)
    - n_components: Number of Gaussian components
    - cov_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
    - seed: Random seed for reproducibility

    Returns:
    - Trained GMM model
    """
    logger.info(
        f"Training GMM with {n_components} components, {X.shape[0]} samples, "
        f"{X.shape[1]} features, and covariance_type='{cov_type}'..."
    )
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        random_state=seed,
        max_iter=200,
        verbose=1,
    )
    gmm.fit(X)
    logger.info("Training complete!")
    return gmm


def evaluate_gmm(
    gmm: GaussianMixture, X: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Evaluates the log-likelihood of the given data X under the GMM model
    and calculates AIC and BIC.

    Parameters:
    - gmm: Trained GMM model
    - X: Input data (n_samples, n_features)

    Returns:
    - Mean log-likelihood
    - Standard deviation of log-likelihood
    - AIC value
    - BIC value
    """
    # Calculate log-likelihood for each sample
    log_likelihoods_per_sample = gmm.score_samples(X)
    mean_ll = float(np.mean(log_likelihoods_per_sample))
    std_ll = float(np.std(log_likelihoods_per_sample))

    # AIC and BIC formulas use the total log-likelihood (sum over all samples)
    total_log_likelihood = gmm.score(X)

    # Extract model parameters needed for AIC/BIC
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_components = gmm.n_components
    cov_type = gmm.covariance_type

    # Calculate the number of free parameters (k)
    n_parameters = _get_n_parameters(n_components, n_features, cov_type)

    # AIC = 2k - 2 * log_likelihood
    aic = 2 * n_parameters - 2 * total_log_likelihood

    # BIC = k * ln(N) - 2 * log_likelihood
    bic = n_parameters * math.log(n_samples) - 2 * total_log_likelihood

    logger.info(f"Mean Log-likelihood: {mean_ll:.2f}, std: {std_ll:.2f}")
    logger.info(f"AIC: {aic:.2f}, BIC: {bic:.2f}")

    return mean_ll, std_ll, aic, bic
