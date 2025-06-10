import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import logging  # Import logging to use it here

# Set a visually appealing style for the plots
sns.set_theme(style="whitegrid")

# Get the logger (assuming utils.get_logger is correctly set up)
logger = logging.getLogger(__name__)


def plot_likelihood_vs_samples(
    df: pl.DataFrame,
    output_path: (str | Path) | None = None,
) -> None:
    """
    Plots the mean log-likelihood vs. the number of samples,
    with separate lines for each number of components, and faceted by overlap status.
    Includes standard deviation as error bands.
    X-axis ticks are set to show all unique n_samples values.

    Args:
        df (pl.DataFrame): DataFrame containing the results.
        output_path (Optional[Union[str, Path]]): If provided, the plot will be saved to this path.
    """
    # Define original column names for means and stds
    mean_ov_col = "mean_log_likelihood_overlap"
    mean_non_ov_col = "mean_log_likelihood"
    std_ov_col = "std_log_likelihood_overlap"
    std_non_ov_col = "std_log_likelihood"

    # --- Data Preparation for Seaborn (Long Format) ---
    df_plot_long = (
        df.with_columns(
            # First, add a column for `n_components` as string for categorical mapping
            pl.col("n_components").cast(pl.Utf8).alias("n_components_str")
        )
        .melt(
            id_vars=["n_samples", "n_components", "n_components_str"],
            value_vars=[mean_ov_col, mean_non_ov_col, std_ov_col, std_non_ov_col],
            variable_name="Metric",
        )
        .with_columns(
            pl.when(pl.col("Metric").str.contains("overlap"))
            .then(pl.lit("Overlapped"))
            .otherwise(pl.lit("Non-Overlapped"))
            .alias("Overlap Type"),
            pl.when(pl.col("Metric").str.contains("mean"))
            .then(pl.col("value"))
            .alias("Mean Log-Likelihood"),
            pl.when(pl.col("Metric").str.contains("std"))
            .then(pl.col("value"))
            .alias("Std Log-Likelihood"),
        )
        .drop("Metric", "value")
        .group_by(["n_samples", "n_components", "n_components_str", "Overlap Type"])
        .agg(
            pl.col("Mean Log-Likelihood").drop_nulls().first(),
            pl.col("Std Log-Likelihood").drop_nulls().first(),
        )
        .sort(["n_samples", "n_components", "Overlap Type"])
    )
    # --- End of Data Preparation ---

    # Get unique n_samples values for tick placement
    x_tick_values = sorted(df_plot_long["n_samples"].unique().to_list())

    # Use seaborn.relplot for faceting by "Overlap Type"
    g = sns.relplot(
        data=df_plot_long,
        x="n_samples",
        y="Mean Log-Likelihood",
        hue="n_components_str",
        style="n_components_str",
        markers=True,
        kind="line",
        errorbar=("sd"),
        col="Overlap Type",
        col_wrap=2,
        height=8,
        aspect=1.1,
        facet_kws={"sharey": True, "sharex": False},
        palette="viridis",
    )

    # Apply log scale and custom ticks to the x-axis for each subplot
    for ax in g.axes.flat:
        ax.set_xscale("log")  # Keep log scale

        # Set specific x-axis ticks to the unique n_samples values
        ax.set_xticks(x_tick_values)
        # Set specific x-axis tick labels to the unique n_samples values
        ax.set_xticklabels(
            [f"{x:,.0f}" for x in x_tick_values]
        )  # Format as integers with commas

        ax.set_xlabel("Number of Samples", fontsize=12)
        ax.set_ylabel("Mean Log-Likelihood", fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Adjust titles for clarity
    g.set_titles("{col_name}")
    g.fig.suptitle("Mean Log-Likelihood vs. Number of Samples", y=1.02, fontsize=16)

    # Improve legend title and placement
    g.add_legend(
        title="Number of Components",
        bbox_to_anchor=(1.02, 0.7),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


# TODO: Change x-axis to number of samples
def plot_aic_bic_vs_components(
    df: pl.DataFrame,
    output_path: (str | Path) | None = None,
) -> None:
    """
    Plots AIC and BIC vs. the number of components,
    with separate lines for each number of samples, and faceted by overlap status.
    A lower AIC/BIC generally indicates a better model.

    Args:
        df (pl.DataFrame): DataFrame containing the results including 'aic' and 'bic'.
        output_path (Optional[Union[str, Path]]): If provided, the plot will be saved to this path.
    """
    # --- DEBUGGING START ---
    logger.info(
        f"DEBUG: Columns available in DataFrame for AIC/BIC plotting: {df.columns}"
    )
    logger.info(f"DEBUG: Schema of DataFrame for AIC/BIC plotting: {df.schema}")
    # --- DEBUGGING END ---

    # --- Data Preparation for Seaborn (Long Format) ---
    df_plot_long = (
        df.with_columns(
            # Add 'n_samples' as string for categorical mapping (hue/style)
            pl.col("n_samples").cast(pl.Utf8).alias("n_samples_str")
        )
        .melt(
            id_vars=["n_components", "n_samples", "n_samples_str"],
            value_vars=["aic", "bic"],  # Using the saved 'aic' and 'bic' columns
            variable_name="Criterion Type",
            value_name="Criterion Value",
        )
        .sort(["n_components", "n_samples", "Criterion Type"])
    )

    # Get unique n_components values for tick placement on X-axis
    x_tick_values = sorted(df_plot_long["n_components"].unique().to_list())

    # Use seaborn.relplot for faceting by "Criterion Type" (AIC vs BIC)
    g = sns.relplot(
        data=df_plot_long,
        x="n_components",
        y="Criterion Value",
        hue="n_samples_str",  # Color by number of samples
        style="n_samples_str",  # Line style by number of samples
        markers=True,  # Add markers for each point
        kind="line",
        col="Criterion Type",  # Separate AIC and BIC into different columns
        col_wrap=2,
        height=6,
        aspect=1.3,
        facet_kws={"sharey": False, "sharex": True},  # Don't share Y, but share X
        palette="viridis",
    )

    # Apply custom ticks to the x-axis for each subplot
    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)  # n_components often scale by powers of 2
        ax.set_xticks(x_tick_values)
        ax.set_xticklabels(
            [f"{int(x)}" for x in x_tick_values]
        )  # Ensure integer labels

        ax.set_xlabel("Number of Components", fontsize=12)
        ax.set_ylabel("Criterion Value (Lower is Better)", fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Adjust titles for clarity
    g.set_titles("{col_name}")
    g.fig.suptitle("AIC and BIC vs. Number of Components", y=1.02, fontsize=16)

    # Improve legend title and placement
    g.add_legend(
        title="Number of Samples",
        bbox_to_anchor=(1.02, 0.7),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
