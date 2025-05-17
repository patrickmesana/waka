import os
import pickle
import numpy as np




def order_points_by_distance(training_data, centers, centers_distances=None):
    """
    Order the points based on their distance to the cluster centers.
    
    Parameters:
    - training_data: Array containing the data points
    - centers: List containing centers of clusters
    
    Returns:
    - sorted_indices: List of arrays containing sorted indices for each cluster
    """

    if centers_distances is None:
        centers_distances = cdist(training_data, centers, metric='euclidean').T

    sorted_indices = []
    for i, _ in enumerate(centers):
        sorted_indices_cluster = np.argsort(centers_distances[i], axis=0).flatten()
        sorted_indices.append(sorted_indices_cluster)
        
    return sorted_indices


from numba import njit, prange
@njit(parallel=True)
def parallel_order_points_by_distance(training_data, centers, centers_distances=None):
    """
    Order the points based on their distance to the cluster centers.
    
    Parameters:
    - training_data: Array containing the data points
    - centers: List containing centers of clusters
    
    Returns:
    - sorted_indices: List of arrays containing sorted indices for each cluster
    """

    if centers_distances is None:
        centers_distances = cdist(training_data, centers, metric='euclidean').T

    sorted_indices = []
    for i, in prange(len(centers)):
        sorted_indices_cluster = np.argsort(centers_distances[i], axis=0).flatten()
        sorted_indices.append(sorted_indices_cluster)
        
    return sorted_indices

def load_or_compute_and_save(task_name, file_path, compute_function, overwrite=False):
    """
    Load data from a file if it exists, otherwise compute using the provided function and save the results.

    Parameters:
    - task_name (str): A name for the task, used in logging messages.
    - file_path (str): The path to the file.
    - compute_function (callable): The function to call if the data needs to be computed. It should return the data to be saved.
    - overwrite (bool, optional): Whether to overwrite the file if it exists and needs to be recomputed. Defaults to False.

    Returns:
    - The loaded or computed data.
    """
    # Check if the file exists and we should load it
    if os.path.exists(file_path) and not overwrite:
        with open(file_path, "rb") as f:
            print(f"{task_name}: Loading data from existing file.")
            return pickle.load(f)
    else:
        print(f"{task_name}: Computing and saving data.")
        
        # Compute the data using the provided function
        data = compute_function()
        
        # Ensure the directory for the file exists, create if not
        folder = os.path.dirname(file_path)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"{task_name}: Created directory {folder}.")
        
        # Save the computed data to the file
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
            print(f"Data saved to {file_path}.")
        
        return data

import time

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time before the function runs
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # End time after the function runs
        duration = end_time - start_time  # Calculate the duration
        print(f"Function {func.__name__} took {duration:.2f} seconds to execute.")
        return result
    return wrapper


def check_number(value):
    try:
        # Convert the value to a float if possible
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("The value cannot be converted to a float")

    if np.isnan(value):
        raise ValueError("The value is NaN")
    if np.isinf(value) or np.isneginf(value):
        raise ValueError("The value is infinity or negative infinity")

    return value

def cumsum(arr):
    result = []
    total = 0
    for num in arr:
        total += num
        result.append(total)
    return result


def plot_distribution(
    values, hp_str, figure_filepath, value_label="Values", bins=100
) -> None:
    """
    Plots the distribution of the given values and saves the plot as a PNG file.

    Parameters:
    -----------
    values : array-like
        The values to plot.
    hp_str : str
        A string to include in the plot title and filename.
    figure_filepath : str
        The path where the plot will be saved.
    value_label : str, optional
        The label for the x-axis. Default is "Values".
    bins : int, optional
        The number of bins for the histogram. Default is 100.
    """
    plt.figure()
    plt.hist(values, bins=bins)
    plt.xlabel(value_label)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {value_label} for KNN {hp_str}")
    plt.savefig(figure_filepath)
    plt.close()
    
    
def plot_distribution_with_labels(
    values,
    labels,
    file_path,
    title,
    xlabel,
    ylabel="Density",
    remove_outliers=False,
    quantile_range=(0.01, 0.99),
    bins=100,
):
    """
    Plot the distribution of values with histograms conditioned on labels, ensuring consistent bins.

    Parameters:
    -----------
    values : array-like
        Array of values to plot (1D array).
    labels : array-like
        Array of corresponding labels (1D array, same length as values).
    file_path : str
        Path to save the plot.
    title : str
        Title of the plot.
    xlabel : str
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis. Default is "Density".
    remove_outliers : bool, optional
        Whether to remove outliers for visualization purposes. Default is False.
    quantile_range : tuple, optional
        Tuple specifying lower and upper quantiles for trimming. Default is (0.01, 0.99).
    bins : int, optional
        Number of bins to use. Default is 100.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(values) != len(labels):
        raise ValueError("The length of 'values' and 'labels' must be the same.")

    if remove_outliers:
        # Determine the lower and upper bounds based on quantiles
        lower_bound, upper_bound = np.quantile(values, quantile_range)
        mask = (values >= lower_bound) & (values <= upper_bound)
        values = values[mask]
        labels = labels[mask]

    # Determine common bin edges
    bin_edges = np.histogram_bin_edges(values, bins=bins)

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Get unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Plot conditional distributions for each label
    for label, count in zip(unique_labels, counts):
        # Filter values for the current label
        label_mask = labels == label
        label_values = values[label_mask]

        sns.histplot(
            label_values,
            bins=bin_edges,
            kde=False,
            label=f"Label {label} ({count})",
            stat="density",
            alpha=0.6,
        )

    # Add title and labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title="Conditioned on Label", fontsize=10)
    plt.grid(True, alpha=0.4)

    # Save and close the plot
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    
    
    
def resample_to_percentages(values, old_percentages, new_percentages):
    """
    Resample values to match new percentage points using linear interpolation.

    Args:
        values: Array-like of values to resample
        old_percentages: Original percentage points
        new_percentages: Target percentage points to resample to

    Returns:
        Resampled values matching new_percentages
    """
    if isinstance(values, dict):
        # Handle dictionary case (like utility_random_score_means)
        return {
            k: np.interp(new_percentages, old_percentages, v) for k, v in values.items()
        }
    elif isinstance(values, list) and values and isinstance(values[0], dict):
        # Handle list of dictionaries case (like results from value-based methods)
        metrics = values[0].keys()
        resampled_values = []
        for metric in metrics:
            metric_values = [v[metric] for v in values]
            interpolated = np.interp(new_percentages, old_percentages, metric_values)
            for i, p in enumerate(new_percentages):
                if len(resampled_values) <= i:
                    resampled_values.append({})
                resampled_values[i][metric] = interpolated[i]
        return resampled_values
    else:
        # Handle simple array case
        return np.interp(new_percentages, old_percentages, values)
    
    

def label_ratio_by_indices(
    percentages, ordered_indices, training_labels, is_removal=True
):
    """
    Compute the ratio of positive labels for selected indices at each percentage.

    Parameters:
    -----------
    percentages : array-like
        Percentages of data to consider.
    ordered_indices : array-like or dict
        Indices of data points, or dict mapping percentages to indices.
    training_labels : array-like
        Labels of the training data.
    is_removal : bool, optional
        Whether the operation is removal (True) or acquisition (False). Default is True.

    Returns:
    --------
    list
        List of label ratios for each percentage.
    """
    label_ratios = []
    for percent in percentages:
        if isinstance(ordered_indices, dict):
            selected_indices_value_based = ordered_indices[percent]
        else:
            if is_removal:
                # For removal: take first cutoff indices (100% -> 0%)
                cutoff = int(len(ordered_indices) * (percent / 100.0))
                selected_indices_value_based = ordered_indices[:cutoff]
            else:
                # For acquisition: take first num_points indices (0% -> 100%)
                num_points = int(len(ordered_indices) * percent / 100)
                selected_indices_value_based = ordered_indices[:num_points]

        reduced_training_labels_value_based = training_labels[
            selected_indices_value_based
        ]

        label_ratio = np.sum(reduced_training_labels_value_based == 1) / len(
            reduced_training_labels_value_based
        )
        if label_ratio is None:
            print("label_ratio is None")
        label_ratios.append(label_ratio)

    return label_ratios



def plot_label_ratio_evolution(
    percentages,
    label_ratios_dict,
    initial_ratio,
    figure_filepath,
    title="Label Ratio Evolution",
    xlabel="Percentage of Dataset Preserved",
    ylabel="Ratio of Positive Labels",
    result_colors=None,
):
    """
    Plot the evolution of label ratios compared to the initial ratio.

    Parameters:
    - percentages: Array of percentage values
    - label_ratios_dict: Dictionary of method names and their corresponding label ratios
    - initial_ratio: The initial ratio of positive labels in the full dataset
    - figure_filepath: Path to save the plot
    - title: Title of the plot
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - result_colors: Optional dictionary mapping method names to colors
    """
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({"font.size": 16})

    linestyle_utility = "-"

    # Plot horizontal line for initial ratio
    plt.axhline(y=initial_ratio, color="gray", linestyle="--", label="Initial Ratio")

    # Plot each method's ratio evolution
    for method_name, ratios in label_ratios_dict.items():
        color = result_colors.get(method_name) if result_colors else None
        plt.plot(
            percentages,
            ratios,
            label=method_name,
            color=color,
            linestyle=linestyle_utility,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.legend(
        prop={"size": 10}, bbox_to_anchor=(0.5, 1.15), loc="upper center", ncol=2
    )
    plt.grid(True)

    # Invert x-axis for removal scenario (100% -> 0%)
    plt.gca().invert_xaxis()

    plt.savefig(figure_filepath, bbox_inches="tight")
    plt.close()