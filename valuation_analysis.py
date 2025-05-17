import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from tqdm import tqdm
from rayPlus import parallel_loop
from utils import time_it


def random_data_minimization(
    utility_fct,
    training_features,
    training_labels,
    validation_features,
    validation_labels,
    global_seed,
    utility_fct_names,
    percentages,
    num_random_partitions=50,
    only_training_indices=False,
    hyperparams=None,
    ray_parallelization=None,
    acquisition_mode=False,
):
    """
    Generate random subsets of training data and evaluate utility function performance.
    
    Parameters:
    -----------
    utility_fct : callable
        Function that evaluates model performance on a subset of data
    training_features : array-like
        Feature matrix for training data
    training_labels : array-like
        Labels for training data
    validation_features : array-like
        Feature matrix for validation data
    validation_labels : array-like
        Labels for validation data
    global_seed : int
        Random seed for reproducibility
    utility_fct_names : list
        Names of utility functions to evaluate
    percentages : list
        Percentages of data to use in subsets
    num_random_partitions : int, default=50
        Number of random partitions to generate for each percentage
    only_training_indices : bool, default=False
        If True, only pass training indices to utility function
    hyperparams : dict, default=None
        Hyperparameters for utility function
    ray_parallelization : dict, default=None
        Configuration for Ray parallelization
    acquisition_mode : bool, default=False
        If True, start from empty set and add data; if False, start from full set and remove data
        
    Returns:
    --------
    tuple:
        - random_score_means: dict mapping utility function names to mean scores
        - random_score_stds: dict mapping utility function names to standard deviations
        - random_score_values_all_details: dict with detailed scores per utility function
    """
    indices = np.arange(0, len(training_features))
    if ray_parallelization is not None:
        import ray
        from rayPlus import parallel_loop
        ray.init()
        # Put training data in object store when using Ray
        training_features_ref = ray.put(training_features)
        training_labels_ref = ray.put(training_labels)
    else:
        training_features_ref = training_features
        training_labels_ref = training_labels

    np.random.seed(global_seed)

    random_score_means = {}
    random_score_stds = {}
    random_score_values_all_details = {}  # Store detailed scores for each utility function

    for utility_fct_name in utility_fct_names:
        random_score_values_all = [] 

        # @time_it
        def iteration(i):
            partition_random_scores = np.zeros(len(percentages))
            shuffled_indices = np.random.permutation(indices)  # Shuffle indices for each partition

            # Get training data from object store if using Ray
            if ray_parallelization is not None:
                training_features_local = ray.get(training_features_ref)
                training_labels_local = ray.get(training_labels_ref)
            else:
                training_features_local = training_features_ref
                training_labels_local = training_labels_ref

            for j, percent in enumerate(percentages):
                if acquisition_mode:
                    # For acquisition, start from 0 and add data points
                    cutoff = int(len(shuffled_indices) * (percent / 100.0))
                else:
                    # For removal, start from 100% and remove data points
                    cutoff = int(len(shuffled_indices) * (percent / 100.0))
                
                selected_indices_random = shuffled_indices[:cutoff]

                # Select the subset of the data (random)
                reduced_training_features_random = training_features_local[selected_indices_random]
                reduced_training_labels_random = training_labels_local[selected_indices_random]

                if only_training_indices:
                    utility_result_random = utility_fct(
                        reduced_training_indices=selected_indices_random,
                        validation_labels = validation_labels,
                        hyperparams=hyperparams,
                    )
                else:
                    # Fit and evaluate the model (random)
                    utility_result_random = utility_fct(
                        reduced_training_features_random,
                        reduced_training_labels_random,
                        validation_features,
                        validation_labels,
                        seed=global_seed,
                        hyperparams=hyperparams,
                    )
                partition_random_scores[j] = utility_result_random.score[utility_fct_name]
            return partition_random_scores
        
        if ray_parallelization is None:
            random_score_values_all = [iteration(i) for i in range(num_random_partitions)]
        else:
            random_score_values_all = parallel_loop(list(range(num_random_partitions)), iteration, return_results=True, n_tasks=ray_parallelization["n_tasks"], use_object_store=ray_parallelization["use_object_store"], init_and_shutdown_ray=False)
                

        random_score_values_all = np.array(random_score_values_all)
        # Compute mean and standard deviation of scores for random subsets
        random_score_means[utility_fct_name] = np.mean(random_score_values_all, axis=0)
        random_score_stds[utility_fct_name] = np.std(random_score_values_all, axis=0)
        random_score_values_all_details[utility_fct_name] = random_score_values_all  # Store detailed scores

    if ray_parallelization is not None:
        ray.shutdown()

    return random_score_means, random_score_stds, random_score_values_all_details


def plot_random_minimization(
    utility_fct_names,
    percentages,
    random_score_means,
    random_score_stds,
    dataset_folder_name="results",
    complete_plot_name='',
    short_plot_name='',
    show_plot=True,
):
    """
    Plot the results of random data minimization experiments.
    
    Parameters:
    -----------
    utility_fct_names : list
        Names of utility functions to plot
    percentages : list
        Percentages of data used in the experiments
    random_score_means : dict
        Mean scores for each utility function at each percentage
    random_score_stds : dict
        Standard deviations of scores for each utility function
    dataset_folder_name : str, default="results"
        Folder to save plots
    complete_plot_name : str, default=''
        Full title for the plot
    short_plot_name : str, default=''
        Short name to append to saved plot filename
    show_plot : bool, default=True
        Whether to display plots
    """
    for utility_fct_name in utility_fct_names:
        # Plot the mean MSE for random subsets
        plt.figure(figsize=(10, 6))
        plt.plot(
            percentages,
            random_score_means[utility_fct_name],
            marker="x",
            label="Random (mean)",
            color="black",
        )

        # Fill between mean ± std deviation for random subsets
        plt.fill_between(
            percentages,
            random_score_means[utility_fct_name] - random_score_stds[utility_fct_name],
            random_score_means[utility_fct_name] + random_score_stds[utility_fct_name],
            color="gray",
            alpha=0.2,
        )

        # upper case utility_fct_name
        utility_fct_name_upper = utility_fct_name.upper()

        plt.title(utility_fct_name_upper + " vs. Percentage of Dataset Preserved")
        #subtilte
        plt.suptitle(complete_plot_name)
        plt.xlabel("Percentage of Dataset Preserved")
        plt.ylabel(utility_fct_name_upper)
        plt.gca().invert_xaxis()  # Invert X axis to show 100% at the left
        plt.legend()
        plt.grid(True)

        # Save the figure to bh_results
        plt.savefig(
            f"{dataset_folder_name}/{utility_fct_name}_vs_percentage_of_dataset_preserved_{short_plot_name}.png"
        )

        if show_plot:
            plt.show()

        # If you don't want the figure to be displayed after saving, you can close it using plt.close()
        plt.close()


def percentage_data_minimization_by_indices(percentages, ordered_indices, training_features, training_labels, validation_features, validation_labels, global_seed, utility_fct, only_training_indices=False, hyperparams=None, return_indices=False):
    """
    Evaluate model performance with decreasing percentages of data selected by ordered indices.
    
    Parameters:
    -----------
    percentages : list
        Percentages of data to preserve
    ordered_indices : array-like or dict
        Ordered indices to select data by importance, or dictionary mapping percentages to indices
    training_features : array-like
        Feature matrix for training data
    training_labels : array-like
        Labels for training data
    validation_features : array-like
        Feature matrix for validation data
    validation_labels : array-like
        Labels for validation data
    global_seed : int
        Random seed for reproducibility
    utility_fct : callable
        Function that evaluates model performance on a subset of data
    only_training_indices : bool, default=False
        If True, only pass training indices to utility function
    hyperparams : dict, default=None
        Hyperparameters for utility function
    return_indices : bool, default=False
        If True, return selected indices with scores
        
    Returns:
    --------
    list:
        If return_indices=False: List of score dictionaries for each percentage
        If return_indices=True: List of (score_dict, indices) tuples for each percentage
    """
    scores = []
    for percent in percentages:

        #if ordered_indices is a dictionary, use the indices at percent key
        if isinstance(ordered_indices, dict):
            selected_indices_value_based = ordered_indices[percent]
        else:
            # Value-based removal - Suposed to go from 100% to 0%
            cutoff = int(len(ordered_indices) * (percent / 100.0))
            selected_indices_value_based = ordered_indices[:cutoff]

        # Select the subset of the data (value-based)
        reduced_training_features_value_based = training_features[
            selected_indices_value_based
        ]
        reduced_training_labels_value_based = training_labels[
            selected_indices_value_based
        ]


        if only_training_indices:
            utility_result = utility_fct(
            reduced_training_indices=selected_indices_value_based,
            validation_labels = validation_labels,
            hyperparams=hyperparams,
            )
        else:
            # Fit and evaluate the model (value-based)
            utility_result = utility_fct(
                reduced_training_features_value_based,
                reduced_training_labels_value_based,
                validation_features,
                validation_labels,
                seed=global_seed,
                hyperparams=hyperparams,
        )
            
        if return_indices:
            scores.append((utility_result.score, selected_indices_value_based))
        else:
            scores.append(utility_result.score)
    return scores

def percentage_data_acquisition_by_indices(percentages, ordered_indices, features, labels, test_features, 
                                         test_labels, seed, utility_fct, hyperparams):
    """
    Evaluate model performance with increasing percentages of data selected by ordered indices.
    
    Parameters:
    -----------
    percentages : list
        Percentages of data to acquire
    ordered_indices : array-like
        Ordered indices to select data by importance
    features : array-like
        Feature matrix for all available training data
    labels : array-like
        Labels for all available training data
    test_features : array-like
        Feature matrix for test data
    test_labels : array-like
        Labels for test data
    seed : int
        Random seed for reproducibility
    utility_fct : callable
        Function that evaluates model performance on a subset of data
    hyperparams : dict
        Hyperparameters for utility function
        
    Returns:
    --------
    list:
        List of score dictionaries for each percentage
    """
    scores = []
    for percent in percentages:
        num_points = int(len(features) * percent / 100)
        selected_indices = ordered_indices[:num_points]
        selected_features = features[selected_indices]
        selected_labels = labels[selected_indices]
        
        # Fit and evaluate the model
        utility_result = utility_fct(
            selected_features,
            selected_labels,
            test_features,
            test_labels,
            seed,
            hyperparams
        )
        scores.append(utility_result.score)
    return scores


def value_based_data_minimization(
    files_dict,
    percentages,
    training_features,
    training_labels,
    validation_features,
    validation_labels,
    global_seed,
    utility_fct,
    indices,
    utility_fct_names,
    num_random_partitions=50,
    dataset_folder_name="results",
    show_plot=True,
    ray_parallelization=None,
):
    """
    Perform data minimization experiments using value-based methods and compare to random baseline.
    
    Parameters:
    -----------
    files_dict : dict
        Dictionary mapping value set names to file paths containing data value scores
    percentages : list
        Percentages of data to preserve
    training_features : array-like
        Feature matrix for training data
    training_labels : array-like
        Labels for training data
    validation_features : array-like
        Feature matrix for validation data
    validation_labels : array-like
        Labels for validation data
    global_seed : int
        Random seed for reproducibility
    utility_fct : callable
        Function that evaluates model performance on a subset of data
    indices : array-like
        Indices to use for random data selection
    utility_fct_names : list
        Names of utility functions to evaluate
    num_random_partitions : int, default=50
        Number of random partitions to generate for each percentage
    dataset_folder_name : str, default="results"
        Folder to save plots
    show_plot : bool, default=True
        Whether to display plots
    ray_parallelization : dict, default=None
        Configuration for Ray parallelization
    """
    print(files_dict)

    # fix seed
    np.random.seed(global_seed)
    results = {}
    # Loop over the files
    for value_set_name, file_path in files_dict.items():
        df = pd.read_csv(file_path)
        ordered_indices = df.sort_values("data_value", ascending=False)["indices"]

        scores = percentage_data_minimization_by_indices(
            percentages,
            ordered_indices,
            training_features,
            training_labels,
            validation_features,
            validation_labels,
            global_seed,
            utility_fct,
        )

        # Store the value-based results
        results[value_set_name] = scores

    # Define how many random partitions you want to generate
    random_score_means, random_score_stds = random_data_minimization(
        utility_fct,
        training_features,
        training_labels,
        validation_features,
        validation_labels,
        global_seed,
        utility_fct_names,
        percentages,
        indices,
        num_random_partitions,
        ray_parallelization=ray_parallelization,
    )

    # Plot the results
    plot_value_based_minimization(
        results,
        utility_fct_names,
        percentages,
        random_score_means,
        random_score_stds,
        dataset_folder_name,
        show_plot,
    )



def plot_value_based_minimization(
    results,
    utility_fct_names,
    percentages,
    random_score_means,
    random_score_stds,
    dataset_folder_name="results",
    show_plot=True,
    complete_plot_name='',
    show_grid=False,
    title=None,
    small_plot=False,
    plot_path=None,
    result_colors=None,
    acquisition_mode=False,
    marker_size=6,
    transpose=False,
    direct_labels=False
):
    """
    Plot comparison of value-based data minimization methods against random baseline.
    
    Parameters:
    -----------
    results : dict
        Dictionary mapping value set names to lists of scores
    utility_fct_names : list
        Names of utility functions to plot
    percentages : list
        Percentages of data used in the experiments
    random_score_means : dict
        Mean scores for random baseline at each percentage
    random_score_stds : dict
        Standard deviations of scores for random baseline
    dataset_folder_name : str, default="results"
        Folder to save plots
    show_plot : bool, default=True
        Whether to display plots
    complete_plot_name : str, default=''
        Full title for the plot
    show_grid : bool, default=False
        Whether to show grid lines on plot
    title : str, default=None
        Custom title for plot
    small_plot : bool, default=False
        Whether to create a smaller plot size
    plot_path : str, default=None
        Custom path to save the plot
    result_colors : dict, default=None
        Dictionary mapping value set names to colors
    acquisition_mode : bool, default=False
        If True, plots are for data acquisition; if False, for data removal
    marker_size : int, default=6
        Size of markers in the plot
    transpose : bool, default=False
        If True, transpose the data structure for plotting
    direct_labels : bool, default=False
        If True, add labels directly to lines instead of using a legend
    """
    
    for utility_fct_name in utility_fct_names:
        if small_plot:
            plt.figure(figsize=(6, 6))
        else:
            plt.figure(figsize=(10, 6))

        plt.rcParams.update({'font.size': 16})

        linestyle_utility = None
        marker_utility = 'o'

        # Plot each method's line
        for value_set_name, scores in results.items():
            color = result_colors.get(value_set_name) if result_colors else None
            marker = marker_utility if marker_size is not None else None
            markersize = marker_size if marker_size is not None else None
            
            # Get y values based on transpose mode
            if transpose:
                y_values = scores[utility_fct_name]
            else:
                y_values = [s[utility_fct_name] for s in scores]
            
            # Plot the line
            line = plt.plot(
                percentages,
                y_values,
                marker=marker,
                markersize=markersize,
                label=None if direct_labels else f"{value_set_name}",  # Only add label if not using direct labels
                color=color,
                linestyle=linestyle_utility
            )[0]  # Get the Line2D object
            
            if direct_labels:
                        # Get the last point coordinates
                        x_pos = percentages[-1]
                        y_pos = y_values[-1]
                        
                        # Move labels more inside the figure (e.g., at 80% of x-axis)
                        x_pos = percentages[int(len(percentages) * 0.8)]
                        # Find y-value at this x-position
                        idx = list(percentages).index(x_pos)
                        y_pos = y_values[idx]
                        
                        # Add larger offset to better space out from curves
                        x_offset = (max(percentages) - min(percentages)) * 0.05  # increased from 0.02
                        y_offset = (max(y_values) - min(y_values)) * 0.05  # increased from 0.02
                        
                        # Place text with matching color and better spacing
                        plt.annotate(
                            value_set_name,
                            xy=(x_pos, y_pos),
                            xytext=(x_pos + x_offset, y_pos + y_offset),
                            color=color,
                            fontsize=10,
                            va='center',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)  # Add white background
                        )

        # Plot random baseline
        marker = "x" if marker_size is not None else None
        markersize = marker_size if marker_size is not None else None
        
        plt.plot(
            percentages,
            random_score_means[utility_fct_name],
            marker=marker,
            markersize=markersize,
            label="Random" if not direct_labels else None,
            color="black",
            linestyle=linestyle_utility
        )

        if direct_labels:
            # Add "Random" label with similar positioning
            x_pos = percentages[int(len(percentages) * 0.8)]
            idx = list(percentages).index(x_pos)
            y_pos = random_score_means[utility_fct_name][idx]
            
            plt.annotate(
                "Random",
                xy=(x_pos, y_pos),
                xytext=(x_pos + x_offset, y_pos + y_offset),
                color="black",
                fontsize=10,
                va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
            )

        # Fill between for random std
        plt.fill_between(
            percentages,
            random_score_means[utility_fct_name] - random_score_stds[utility_fct_name],
            random_score_means[utility_fct_name] + random_score_stds[utility_fct_name],
            color="gray",
            alpha=0.2,
        )

        utility_fct_name_upper = utility_fct_name.upper().replace('_', ' ')

        if title:
            plt.title(title)
            plt.suptitle(complete_plot_name)
            
        if acquisition_mode:
            plt.xlabel("Percentage of Dataset Acquired")
        else:
            plt.xlabel("Percentage of Dataset Preserved")
            
        plt.ylabel(utility_fct_name_upper)
        
        if not acquisition_mode:
            plt.gca().invert_xaxis()
        
        # Only show legend if not using direct labels
        if not direct_labels:
            if small_plot:
                plt.legend(prop={'size': 10}, bbox_to_anchor=(0.5, 1.15), 
                          loc='upper center', ncol=2)
            else:
                plt.legend()

        plt.grid(show_grid)

        # Save plot
        plot_suffix = "_small" if small_plot else ""
        mode_str = "acquisition" if acquisition_mode else "removal"

        if plot_path:
            plt.savefig(plot_path, bbox_inches='tight')
        else:
            plt.savefig(
                f"{dataset_folder_name}/{utility_fct_name}_vs_percentage_of_dataset_{mode_str}_{complete_plot_name}{plot_suffix}.png",
                bbox_inches='tight'
            )

        if show_plot:
            plt.show()

        plt.close() 


def plot_aggregated_mean_std_curves_separate(utility_fct_names, percentages, random_score_values_all_details, results_folder, show_plot=True):
    """
    Plot aggregated mean and standard deviation curves for each utility function separately.
    
    Parameters:
    -----------
    utility_fct_names : list
        Names of utility functions to plot
    percentages : list
        Percentages of data used in the experiments
    random_score_values_all_details : dict
        Nested dictionary with detailed scores for each configuration and utility function
    results_folder : str
        Folder to save plots
    show_plot : bool, default=True
        Whether to display plots
    """
    for utility_name in utility_fct_names:
        # Initialize arrays to hold all scores for each percentage across configurations
        all_scores_at_percentages = np.zeros((len(percentages), 0))
        
        for config in random_score_values_all_details:
            # Extract the detailed scores for the current utility function from each configuration
            detailed_scores = random_score_values_all_details[config][utility_name]

            all_scores_at_percentages = np.hstack((all_scores_at_percentages, detailed_scores))


        
        # Now, compute the mean and standard deviation across all configurations for each percentage
        aggregated_means = np.mean(all_scores_at_percentages, axis=1)
        aggregated_stds = np.std(all_scores_at_percentages, axis=1)

        
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(percentages, aggregated_means, label='Aggregated Mean', marker='o')
        plt.fill_between(percentages, aggregated_means - aggregated_stds, aggregated_means + aggregated_stds, alpha=0.2, label='Aggregated Std Dev')
        
        plt.title(f'Aggregated Mean and Std Dev for {utility_name}')
        plt.xlabel('Percentage of Data Used')
        plt.ylabel(f'{utility_name} Score')
        plt.legend()
        plt.grid(True)

        # Save the figure
        plt.savefig(f'{results_folder}/aggregated_{utility_name}_mean_std.png')


        if show_plot:
            plt.show()

        plt.close()


def plot_mean_curves(utility_fct_names, percentages, aggregated_score_means, dataset_folder_name='tmp', show_plot=True):
    """
    Plot mean performance curves across different configurations for each utility function.
    
    Parameters:
    -----------
    utility_fct_names : list
        Names of utility functions to plot
    percentages : list
        Percentages of data used in the experiments
    aggregated_score_means : dict
        Nested dictionary mapping configuration keys to utility function means
    dataset_folder_name : str, default='tmp'
        Folder to save plots
    show_plot : bool, default=True
        Whether to display plots
    """
    plt.figure(figsize=(10, 6))

    # Loop over each utility function
    for utility_name in utility_fct_names:
        plt.figure(figsize=(10, 6))  # Create a new figure for each utility function
        for config_key, metrics in aggregated_score_means.items():
            if utility_name in metrics:  # Check if this utility function's data is available


                means = metrics[utility_name]  # Directly access the means array

                print(config_key)
                print(means)
                print('-------------------')

                plt.plot(percentages, means, label=f'{config_key}', marker='o')
        
        plt.title(f'Mean Performance Across Configurations for {utility_name.upper()}')
        plt.xlabel('Percentage of Data Used')
        plt.ylabel('Mean ' + utility_name.upper())
        # plt.legend()
        plt.grid(True)

        # Save the figure 
        plt.savefig(f'{dataset_folder_name}/mean_{utility_name}_vs_percentage_of_dataset_preserved.png')

        if show_plot:
            plt.show()

        plt.close()

