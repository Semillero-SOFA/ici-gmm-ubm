import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Set a visually appealing style for the plots
sns.set_theme(style="whitegrid")


def plot_likelihood_vs_components(
    hdf5_path: str | Path,
    output_path: (str | Path) | None = None,
    plot_overlap: bool = False,
) -> None:
    """
    Reads results from an HDF5 file and plots the log-likelihood vs. the number of components,
    with separate lines for each number of samples.

    Args:
        hdf5_path (Union[str, Path]): Path to the input HDF5 file.
        output_path (Optional[Union[str, Path]]): If provided, the plot will be saved to this path.
                                                  Otherwise, the plot will be displayed.
        plot_overlap (bool): If True, plots 'mean_log_likelihood_overlap' instead of the standard one.
    """
    file_path = Path(hdf5_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"The file was not found at the specified path: {file_path}"
        )

    # --- 1. Read HDF5 data and convert to Pandas DataFrame ---
    with h5py.File(file_path, "r") as f:
        if "results" not in f:
            raise KeyError("HDF5 file must contain a dataset named 'results'.")
        # Read the structured array from the dataset
        results_data = f["results"][:]

    df = pd.DataFrame(results_data)

    if df.empty:
        print("Warning: The 'results' dataset is empty. No plot will be generated.")
        return

    # Determine which columns to use for the y-axis and its error
    y_col = "mean_log_likelihood_overlap" if plot_overlap else "mean_log_likelihood"
    y_err_col = "std_log_likelihood_overlap" if plot_overlap else "std_log_likelihood"

    # --- 2. Create the Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # Group data by 'n_samples' to plot a separate line for each
    for n_samples, group in df.groupby("n_samples"):
        # Sort by n_components to ensure the line connects points correctly
        group = group.sort_values("n_components")

        ax.errorbar(
            group["n_components"],
            group[y_col],
            yerr=group[y_err_col],
            marker="o",
            linestyle="-",
            capsize=4,  # Adds caps to the error bars
            label=f"n_samples = {n_samples}",
        )

    # --- 3. Customize and Finalize the Plot ---
    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Mean Log-Likelihood", fontsize=12)
    plot_title = y_col.replace("_", " ").title()
    ax.set_title(
        f"Model Performance: {plot_title} vs. Number of Components",
        fontsize=14,
        weight="bold",
    )

    ax.legend(title="Sample Sizes")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Make sure x-axis ticks are integers if components are integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()

    # Set number of components axis as logarithmic
    ax.set_xscale("log", base=2)

    # --- 4. Save or Display the Plot ---
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
