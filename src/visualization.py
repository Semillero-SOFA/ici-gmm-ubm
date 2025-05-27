import logging
import numpy as np
import matplotlib.pyplot as plt


def get_logger():
    return logging.getLogger(__name__)


def plot_bic_curve(bic_scores: list, output_path: str):
    """
    Plot BIC curve for GMM model selection.

    Args:
        bic_scores: List of BIC scores for different component counts
        output_path: Path to save the plot
    """
    logger = get_logger()
    logger.info("Plotting BIC curve")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(bic_scores) + 1), bic_scores, marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("BIC")
    plt.title("Optimal BIC curve for GMM model selection")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.debug(f"BIC plot saved to {output_path}")


def plot_likelihood_distribution(log_likelihoods: np.ndarray, output_path: str):
    """
    Plot the distribution of log-likelihoods for the test set.

    Args:
        log_likelihoods: Array of log-likelihood values
        output_path: Path to save the plot
    """
    logger = get_logger()
    logger.info("Plotting likelihood distribution")

    plt.figure(figsize=(10, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(log_likelihoods, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    plt.xlabel("Log-likelihood")
    plt.ylabel("Frequency")
    plt.title("Distribution of Log-likelihoods (Test Set)")
    plt.grid(True, alpha=0.3)

    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(log_likelihoods, vert=True)
    plt.ylabel("Log-likelihood")
    plt.title("Box Plot of Log-likelihoods")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.debug(f"Likelihood distribution plot saved to {output_path}")


def plot_data_distribution(train_df, test_df, output_path: str):
    """
    Plot distribution of I/Q data for training and test sets.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        output_path: Path to save the plot
    """
    logger = get_logger()
    logger.info("Plotting data distribution")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training I component
    axes[0, 0].hist(
        train_df["I"].to_numpy(), bins=50, alpha=0.7, color="blue", label="Train I"
    )
    axes[0, 0].set_title("Training Set - I Component")
    axes[0, 0].set_xlabel("I Values")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, alpha=0.3)

    # Training Q component
    axes[0, 1].hist(
        train_df["Q"].to_numpy(), bins=50, alpha=0.7, color="red", label="Train Q"
    )
    axes[0, 1].set_title("Training Set - Q Component")
    axes[0, 1].set_xlabel("Q Values")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # Test I component
    axes[1, 0].hist(
        test_df["I"].to_numpy(), bins=50, alpha=0.7, color="lightblue", label="Test I"
    )
    axes[1, 0].set_title("Test Set - I Component")
    axes[1, 0].set_xlabel("I Values")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)

    # Test Q component
    axes[1, 1].hist(
        test_df["Q"].to_numpy(), bins=50, alpha=0.7, color="lightcoral", label="Test Q"
    )
    axes[1, 1].set_title("Test Set - Q Component")
    axes[1, 1].set_xlabel("Q Values")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.debug(f"Data distribution plot saved to {output_path}")


def plot_scatter_iq(train_df, test_df, output_path: str, sample_size: int = 5000):
    """
    Plot I vs Q scatter plot for both training and test sets.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        output_path: Path to save the plot
        sample_size: Number of samples to plot (for performance)
    """
    logger = get_logger()
    logger.info("Plotting I vs Q scatter plot")

    # Sample data for visualization performance
    train_sample = train_df.sample(min(sample_size, len(train_df)))
    test_sample = test_df.sample(min(sample_size, len(test_df)))

    plt.figure(figsize=(12, 5))

    # Training data scatter
    plt.subplot(1, 2, 1)
    plt.scatter(
        train_sample["I"].to_numpy(),
        train_sample["Q"].to_numpy(),
        alpha=0.5,
        s=1,
        color="blue",
    )
    plt.xlabel("I Component")
    plt.ylabel("Q Component")
    plt.title("Training Set - I vs Q")
    plt.grid(True, alpha=0.3)

    # Test data scatter
    plt.subplot(1, 2, 2)
    plt.scatter(
        test_sample["I"].to_numpy(),
        test_sample["Q"].to_numpy(),
        alpha=0.5,
        s=1,
        color="red",
    )
    plt.xlabel("I Component")
    plt.ylabel("Q Component")
    plt.title("Test Set - I vs Q")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.debug(f"I vs Q scatter plot saved to {output_path}")
