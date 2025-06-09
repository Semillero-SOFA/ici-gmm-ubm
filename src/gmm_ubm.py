# gmm_ubm.py

import numpy as np
from sklearn.mixture import GaussianMixture

from utils import get_logger

logger = get_logger(__name__)


def train_gmm(
    X: np.ndarray, n_components: int, cov_type: str = "full", seed: int = 15
) -> GaussianMixture:
    """
    Entrena un modelo Gaussian Mixture con los datos X.

    Parámetros:
    - X: Datos de entrada (n_samples, n_features)
    - n_components: Número de componentes gaussianas
    - seed: Semilla aleatoria para reproducibilidad

    Retorna:
    - Modelo GMM entrenado
    """
    logger.info(
        f"Training GMM with {n_components} components and {X.shape[0]} samples..."
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


def evaluate_log_likelihood(gmm: GaussianMixture, X: np.ndarray) -> tuple[float, float]:
    """
    Evalúa la log-verosimilitud de los datos dados X bajo el modelo GMM.

    Retorna:
    - Media de la log-verosimilitud
    - Desviación estándar de la log-verosimilitud
    """
    log_likelihoods = gmm.score_samples(X)
    mean_ll = float(np.mean(log_likelihoods))
    std_ll = float(np.std(log_likelihoods))
    logger.info(f"Mean Log-likelihood: {mean_ll:.2f}, std: {std_ll:.2f}")
    return mean_ll, std_ll

