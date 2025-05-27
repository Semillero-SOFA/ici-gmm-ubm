#!/usr/bin/env python3
"""
Main script for GMM training with checkpoint system.
This script orchestrates the complete training pipeline.
"""

import os
from pathlib import Path

# Import modular components
from config import setup_logger, DATABASE_DIR, RESULTS_DIR, DEFAULT_MAX_GMM_COMPONENTS
from checkpoint_manager import get_completed_steps, save_checkpoint
from data_processor import load_all_data, filter_overlap, train_test_split_data
from gmm_trainer import select_best_gmm, calculate_likelihood_statistics
from visualization import (
    plot_bic_curve,
    plot_likelihood_distribution,
    plot_data_distribution,
    plot_scatter_iq,
)
from model_utils import save_model, get_model_summary


def main():
    """Main execution function."""
    # Setup logging
    logger = setup_logger("gmm_training")

    logger.info("=== Starting GMM Training with Checkpoint System ===")

    # Check overall progress
    completed_steps = get_completed_steps()
    logger.info(
        f"Found {len(completed_steps)} completed checkpoint steps: {completed_steps}"
    )

    try:
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        df_32gbd = load_all_data(Path(DATABASE_DIR) / "Estimation" / "32GBd")
        logger.info(f"Loaded {len(df_32gbd)} total samples")

        # Step 2: Filter data
        logger.info("Step 2: Filtering scenarios with spectral overlapping...")
        overlap_df = filter_overlap(df_32gbd)
        logger.info(f"Filtered to {len(overlap_df)} samples with spectral overlapping")

        # Step 3: Split data
        logger.info("Step 3: Splitting data into train/test sets...")
        train_df, test_df, X_train, X_test = train_test_split_data(overlap_df)

        # Step 4: Train GMM
        logger.info("Step 4: Training GMM models on training set...")
        best_gmm, bic_scores = select_best_gmm(X_train, DEFAULT_MAX_GMM_COMPONENTS)

        # Step 5: Calculate likelihood statistics
        logger.info("Step 5: Calculating likelihood on test set...")
        likelihood_stats = calculate_likelihood_statistics(best_gmm, X_test)

        # Step 6: Generate visualizations
        logger.info("Step 6: Generating plots and final outputs...")

        # Check if visualization step is completed
        if "visualization" not in completed_steps:
            logger.info("Creating visualizations...")

            # BIC curve
            plot_bic_curve(bic_scores, os.path.join(RESULTS_DIR, "gmm_bic_plot.png"))

            # Likelihood distribution
            plot_likelihood_distribution(
                likelihood_stats["individual_log_likelihoods"],
                os.path.join(RESULTS_DIR, "likelihood_distribution.png"),
            )

            # Data distribution plots
            plot_data_distribution(
                train_df, test_df, os.path.join(RESULTS_DIR, "data_distribution.png")
            )

            # I vs Q scatter plot
            plot_scatter_iq(
                train_df, test_df, os.path.join(RESULTS_DIR, "iq_scatter.png")
            )

            # Mark visualization as complete
            save_checkpoint(
                {
                    "bic_plot_saved": True,
                    "likelihood_plot_saved": True,
                    "data_distribution_plot_saved": True,
                    "iq_scatter_plot_saved": True,
                },
                "visualization",
            )

        # Step 7: Save final results
        if "final_results" not in completed_steps:
            logger.info("Saving final results...")

            # Save the complete model
            model_path = save_model(
                gmm_model=best_gmm,
                X_train=X_train,
                X_test=X_test,
                train_metadata=train_df.select(
                    ["Distance", "Power", "Spacing", "OSNR"]
                ),
                test_metadata=test_df.select(["Distance", "Power", "Spacing", "OSNR"]),
                bic_scores=bic_scores,
                likelihood_stats=likelihood_stats,
            )

            # Get model summary
            model_summary = get_model_summary(best_gmm)

            # Save final checkpoint
            save_checkpoint(
                {
                    "final_model_saved": True,
                    "model_path": model_path,
                    "model_summary": model_summary,
                },
                "final_results",
            )

        # Final summary
        logger.info("=== Training Complete ===")
        logger.info(
            f"GMM model trained with {best_gmm.n_components} components (lowest BIC)."
        )
        logger.info(
            f"Test set mean log-likelihood: {likelihood_stats['mean_log_likelihood']:.4f}"
        )

        # Interpretation of results
        likelihood_threshold = -10  # This can be adjusted based on domain knowledge
        if likelihood_stats["mean_log_likelihood"] > likelihood_threshold:
            logger.info("✓ High likelihood detected - GMM adapted well to the scenario")
        else:
            logger.info("⚠ Low likelihood detected - Consider model adjustment")

        # Summary of checkpoint system
        final_steps = get_completed_steps()
        logger.info(f"Total completed steps: {len(final_steps)}")
        logger.info("All results saved to: " + RESULTS_DIR)

        return {
            "model": best_gmm,
            "likelihood_stats": likelihood_stats,
            "bic_scores": bic_scores,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        }

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    results = main()
