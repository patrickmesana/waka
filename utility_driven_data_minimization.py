import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from math import comb  # Import the combinatorial function
import knn_valuation
import utils
import data_values_analysis
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.special import comb


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from waka import (
    compute_unormalized_average_waka_values,
    compute_self_waka_value_with_optional_target,
    compute_self_waka_value_with_optional_target_with_knn,
)

from lira_attack import *
from tqdm import tqdm

from membership_inference_security_game import *

import os
import pickle

from utils import load_or_compute_and_save, time_it

import datetime

from sklearn.metrics import matthews_corrcoef

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from utils import plot_distribution, plot_distribution_with_labels, resample_to_percentages, label_ratio_by_indices



def iterative_removal(
    training_features,
    training_labels,
    percentages,
    score_fct,
    seed,
    config,
    ray_parallelization=None,
    return_iterative_scors=False,
):
    """
    Iteratively remove data points based on a scoring function.

    This function implements an iterative data removal process where at each step, points are scored
    and removed based on those scores until reaching the target percentages of data remaining.

    Parameters
    ----------
    training_features : numpy.ndarray
        Features of the training data.
    training_labels : numpy.ndarray 
        Labels of the training data.
    percentages : numpy.ndarray
        Array of percentages of data to retain, in descending order.
    score_fct : callable
        Function that computes scores for each data point to determine removal order.
        Should take (features, labels) as input and return scores.
    seed : int
        Random seed for reproducibility.
    config : dict
        Configuration dictionary containing parameters like K for KNN.
    ray_parallelization : dict, optional
        Dictionary with parallelization settings for Ray.
    return_iterative_scors : bool, optional
        If True, return scores at each iteration along with indices.

    Returns
    -------
    dict
        Maps percentages to indices of retained data points.
        If return_iterative_scors is True, also includes scores at each iteration.
    """

    percent_indices = {}
    percentages = np.insert(percentages, 0, 100, axis=0)

    current_training_features = training_features
    current_training_labels = training_labels

    K = config["K"]

    # compute the remaining number of points at each percentage
    nbr_of_points_per_percent = np.array(
        [len(training_features) * percent / 100 for percent in percentages]
    )

    if return_iterative_scors:
        original_indices = np.arange(0, len(training_features))
        iterative_scores = {}

    for p_i, percent in enumerate(percentages):

        # print(f"-----------Running for {percent}% -----------")

        if p_i == len(percentages) - 1:
            break

        unique_labels = np.unique(current_training_labels)
        if len(unique_labels) < 2:
            target_model_losses = 1 - np.sum(
                unique_labels[0] == current_training_labels[:, np.newaxis], axis=1
            )
        else:
            target_model = KNeighborsClassifier(n_neighbors=K)
            target_model.fit(current_training_features, current_training_labels)

            # Note: this is kind of useless for K=1 because we already know that the indices are in order
            _, in_indices = target_model.kneighbors(
                current_training_features, n_neighbors=K
            )
            in_neighbor_labels = current_training_labels[in_indices]

            target_model_losses = (
                1
                - np.sum(
                    in_neighbor_labels == current_training_labels[:, np.newaxis], axis=1
                )
                / K
            )

        attack_scores = score_fct(
            current_training_features,
            current_training_labels,
            target_model_losses,
            K,
            seed,
        )

        # Determine points to remove to reach the target percentage knowing that we have removed 1 - percent of the data so far
        # Thus everytime we need to remove a bigger portion of the current dataset
        num_points_to_remove = int(
            nbr_of_points_per_percent[p_i] - nbr_of_points_per_percent[p_i + 1]
        )

        # this is equivalent to cutting off the tail of the sorted scores, here the highest LiRA scores
        sorted_scores_indices = np.argsort(attack_scores)
        indices_to_remove = sorted_scores_indices[-num_points_to_remove:]
        indices_to_keep = sorted_scores_indices[:-num_points_to_remove]

        if return_iterative_scors:
            iterative_scores[percentages[p_i + 1]] = [
                attack_scores,
                original_indices[indices_to_keep],
                original_indices[indices_to_remove],
            ]

        current_training_features = np.delete(
            current_training_features, indices_to_remove, axis=0
        )
        current_training_labels = np.delete(current_training_labels, indices_to_remove)

        percent_indices[percentages[p_i + 1]] = indices_to_keep

    if return_iterative_scors:
        return iterative_scores
    else:
        return percent_indices


def iterative_vanilla_lira_removal(
    training_features,
    training_labels,
    percentages,
    seed,
    config,
    ray_parallelization=None,
    N_shadow=16,
):
    """
    Iteratively remove data points based on vanilla LiRA scores.

    Parameters:
    -----------
    training_features : array-like
        Training data features.
    training_labels : array-like
        Training data labels.
    percentages : array-like
        Array of percentages of data to keep at each iteration.
    seed : int
        Random seed for reproducibility.
    config : dict
        Configuration parameters for the model.
    ray_parallelization : dict, optional
        Configuration for Ray parallelization. Default is None.
    N_shadow : int, optional
        Number of shadow models to train. Default is 16.

    Returns:
    --------
    dict
        Dictionary mapping percentages to kept indices.
    """
    score_fct = lambda training_features, training_labels, target_model_losses, K, seed: lira_vanilla_scores(
        training_features, training_labels, target_model_losses, K, N_shadow, seed
    )

    return iterative_removal(
        training_features,
        training_labels,
        percentages,
        score_fct,
        seed,
        config,
        ray_parallelization=ray_parallelization,
    )


def iterative_waka_removal(
    training_features,
    training_labels,
    training_indices,
    test_features,
    test_labels,
    percentages,
    seed,
    config,
    ray_parallelization=None,
    self_waka=False,
):
    """
    Iteratively remove data points based on WaKA scores.

    Parameters:
    -----------
    training_features : array-like
        Training data features.
    training_labels : array-like
        Training data labels.
    training_indices : array-like
        Indices of training data points.
    test_features : array-like
        Test data features.
    test_labels : array-like
        Test data labels.
    percentages : array-like
        Array of percentages of data to keep at each iteration.
    seed : int
        Random seed for reproducibility.
    config : dict
        Configuration parameters for the model.
    ray_parallelization : dict, optional
        Configuration for Ray parallelization. Default is None.
    self_waka : bool, optional
        Whether to use self WaKA scores. Default is False.

    Returns:
    --------
    dict
        Dictionary mapping percentages to kept indices.
    """
    if self_waka:
        score_fct = lambda training_features, training_labels, target_model_losses, K, seed: compute_self_unormalized_average_waka_values(
            training_features,
            training_labels,
            training_indices,
            K,
            nbr_of_considered_points=200,
            ray_parallelization=ray_parallelization,
        )
    else:
        score_fct = lambda training_features, training_labels, target_model_losses, K, seed: np.sum(
            compute_unormalized_average_waka_values(
                training_features,
                training_labels,
                test_features,
                test_labels,
                K,
                nbr_of_considered_points=200,
                ray_parallelization=ray_parallelization,
            ),
            axis=0,
        )

    return iterative_removal(
        training_features,
        training_labels,
        percentages,
        score_fct,
        seed,
        config,
        ray_parallelization=ray_parallelization,
        return_iterative_scors=False,
    )


def iterative_privacy_score_removal(
    training_features,
    training_labels,
    percentages,
    attack_function,
    config,
    ray_parallelization=None,
):
    """
    Iteratively remove data points based on privacy scores.

    Parameters:
    -----------
    training_features : array-like
        Training data features.
    training_labels : array-like
        Training data labels.
    percentages : array-like
        Array of percentages of data to keep at each iteration.
    attack_function : callable
        Function that computes privacy scores for each data point.
    config : dict
        Configuration parameters for the model.
    ray_parallelization : dict, optional
        Configuration for Ray parallelization. Default is None.

    Returns:
    --------
    dict
        Dictionary mapping percentages to kept indices.
    """
    if ray_parallelization is not None:
        import ray
        from rayPlus import parallel_loop_lazy_with_progress

        ray.init()

    percent_indices = {}
    percentages = np.insert(percentages, 0, 100, axis=0)

    current_training_features = training_features
    current_training_labels = training_labels

    K = config["K"]

    # compute the remaining number of points at each percentage
    nbr_of_points_per_percent = np.array(
        [len(training_features) * percent / 100 for percent in percentages]
    )

    for p_i, percent in enumerate(percentages):
        print(f"-----------Running for {percent}% -----------")

        if p_i == len(percentages) - 1:
            break

        target_model = KNeighborsClassifier(n_neighbors=K)
        target_model.fit(current_training_features, current_training_labels)

        # @time_it
        def attribution_attack_percent(i):
            challenge_data_point = current_training_features[i]
            challenge_data_point_label = current_training_labels[i]

            _, challenge_points_nn_indices = target_model.kneighbors(
                np.array([challenge_data_point]), n_neighbors=K
            )
            neighbor_labels = current_training_labels[challenge_points_nn_indices]
            target_model_loss = (
                np.sum(neighbor_labels == challenge_data_point_label) / K
            )
            attack_score = attack_function(
                challenge_data_point, challenge_data_point_label, target_model_loss, K
            )

            return attack_score

        if ray_parallelization is None:
            attack_scores = [
                attribution_attack_percent(i)
                for i in range(len(current_training_features))
            ]
        else:
            attack_scores = parallel_loop_lazy_with_progress(
                list(range(len(current_training_features))),
                attribution_attack_percent,
                return_results=True,
                n_tasks=ray_parallelization["n_tasks"],
                init_and_shutdown_ray=False,
                progress_update_interval=progress_update_interval,
            )

        num_points_to_remove = int(
            nbr_of_points_per_percent[p_i] - nbr_of_points_per_percent[p_i + 1]
        )

        sorted_scores_indices = np.argsort(attack_scores)
        indices_to_remove = sorted_scores_indices[-num_points_to_remove:]
        indices_to_keep = sorted_scores_indices[:-num_points_to_remove]

        current_training_features = np.delete(
            current_training_features, indices_to_remove, axis=0
        )
        current_training_labels = np.delete(current_training_labels, indices_to_remove)

        percent_indices[percentages[p_i + 1]] = indices_to_keep

    if ray_parallelization is not None:
        ray.shutdown()

    return percent_indices


def evaluate_privacy_by_indices(
    selected_attack,
    attribution_results_with_indices,
    df_training_data,
    df_distribution_data,
    config,
    feature_names,
    label_name,
    first_seed,
    ray_parallelization=None,
    percent_nbr_of_games=10,
    batch=False,
):
    """
    Evaluate privacy of a dataset using membership inference attacks.

    Parameters:
    -----------
    selected_attack : callable
        The attack function to use for evaluation.
    attribution_results_with_indices : list
        List of tuples containing (results, indices) for each percentage.
    df_training_data : DataFrame
        Training data.
    df_distribution_data : DataFrame
        Distribution data for attack evaluation.
    config : dict
        Configuration parameters for the model.
    feature_names : list
        Names of feature columns.
    label_name : str
        Name of label column.
    first_seed : int
        Initial random seed for reproducibility.
    ray_parallelization : dict, optional
        Configuration for Ray parallelization. Default is None.
    percent_nbr_of_games : int, optional
        Number of games to run for each percentage. Default is 10.
    batch : bool, optional
        Whether to run attacks in batch mode. Default is False.

    Returns:
    --------
    list
        List of attack results for each percentage.
    """

    def attribution_attack_percent(result, indices):
        df_training_reduced = df_training_data.iloc[indices]

        target_model = KNeighborsClassifier(n_neighbors=config["K"])
        target_model.fit(
            df_training_reduced[feature_names].values,
            df_training_reduced[label_name].values,
        )

        seed_list = np.arange(first_seed, first_seed + percent_nbr_of_games)
        game_results = []

        for seed in seed_list:
            fpr, tpr, roc_auc = run_game_for_auc(
                seed,
                lambda seed: challenge_target_model(
                    target_model,
                    df_training_reduced,
                    df_distribution_data,
                    config["K"],
                    selected_attack,
                    number_of_points_per_game,
                    seed,
                    feature_names=feature_names,
                    label_name=label_name,
                    batch=batch,
                ),
            )
            game_results.append((fpr, tpr, roc_auc))

        fpr = [fpr for fpr, _, _ in game_results]
        tpr = [tpr for _, tpr, _ in game_results]
        roc_auc = np.mean([roc_auc for _, _, roc_auc in game_results])

        return fpr, tpr, roc_auc

    if ray_parallelization is None:
        attributions_percents_attack_results = [
            attribution_attack_percent(result_with_indices[0], result_with_indices[1])
            for result_with_indices in attribution_results_with_indices[1:]
        ]
    else:
        from rayPlus import parallel_loop

        attributions_percents_attack_results = parallel_loop(
            attribution_results_with_indices[1:],
            attribution_attack_percent,
            return_results=True,
            n_tasks=ray_parallelization["n_tasks"],
        )

    return attributions_percents_attack_results


def knn_utility_fct_with_distances(
    distances, training_labels, reduced_training_indices, validation_labels, hyperparams
):
    """
    Compute KNN utility function using pre-computed distances.

    Parameters:
    -----------
    distances : array-like
        Pre-computed distances between validation and training points.
    training_labels : array-like
        Labels of training data.
    reduced_training_indices : array-like
        Indices of selected training points.
    validation_labels : array-like
        Labels of validation data.
    hyperparams : dict
        Hyperparameters for KNN model.

    Returns:
    --------
    UtilityResult
        Object containing utility metrics.
    """
    K = hyperparams["K"]
    reduced_distances = distances[:, reduced_training_indices]
    reduced_training_labels = training_labels[reduced_training_indices]

    nearest_neighbors_indices = np.argsort(reduced_distances, axis=1)[:, :K]

    losses = []
    for t_i, t_nearest_neighbors_indices in enumerate(nearest_neighbors_indices):
        labels = reduced_training_labels[t_nearest_neighbors_indices]
        loss = 1 - np.sum(labels == validation_labels[t_i]) / K
        losses.append(loss)

    utility_score = np.mean(losses)

    return utility.UtilityResult({"knn_loss": utility_score})


def knn_utility_fct_old_(
    training_features,
    training_labels,
    validation_features,
    validation_labels,
    seed,
    hyperparams,
):
    """
    Compute KNN utility function using training and validation data.

    This function calculates various utility metrics for a K-nearest neighbors classifier
    by comparing predictions on validation data against the true validation labels.

    Parameters
    ----------
    training_features : numpy.ndarray
        Features of the training data.
    training_labels : numpy.ndarray
        Labels of the training data.
    validation_features : numpy.ndarray
        Features of the validation data.
    validation_labels : numpy.ndarray
        Labels of the validation data.
    seed : int
        Random seed for reproducibility.
    hyperparams : dict
        Dictionary containing hyperparameters, must include 'K' for number of neighbors.

    Returns
    -------
    utility.UtilityResult
        Object containing utility metrics including:
        - knn_loss: Mean loss based on K-nearest neighbor predictions
        - mcc: Matthews correlation coefficient
        - macro_f1: Macro-averaged F1 score
        - accuracy: Classification accuracy
        - minority_f1: F1 score for the minority class
    """
    
    K = hyperparams["K"]
    # Ensure reduced_distances only considers validation set distances
    # Assuming `distances` is a matrix of distances between all training and prototype points

    # Compute the loss for each validation point as the mean of votes that match the true label
    losses = []

    preds = []

    # Process each validation feature one at a time
    for v_feature, v_label in zip(validation_features, validation_labels):
        # Compute distances from the current validation point to all training points
        distances = cdist([v_feature], training_features)[
            0
        ]  # Get the 0th because cdist returns a 2D array even for one sample

        # Get indices of the K nearest neighbors
        nearest_neighbors_indices = np.argsort(distances)[:K]

        # Retrieve the labels of the nearest neighbors
        labels = training_labels[nearest_neighbors_indices]

        # Compute the loss for the current validation point
        loss = 1 - np.sum(labels == v_label) / K

        pred = np.argmax(np.bincount(labels))

        preds.append(pred)

        losses.append(loss)

    # The overall utility is the negative mean loss (since lower loss is better, but we might want higher scores for better)
    utility_score = np.mean(losses)

    mcc = matthews_corrcoef(validation_labels, preds)

    macro_f1 = f1_score(validation_labels, preds, average="macro")

    accuracy_score_ = accuracy_score(validation_labels, preds)

    # Find the minority class in the validation set
    unique_classes, class_counts = np.unique(validation_labels, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    minority_f1 = f1_score(validation_labels, preds, pos_label=minority_class)

    return utility.UtilityResult(
        {
            "knn_loss": utility_score,
            "mcc": mcc,
            "macro_f1": macro_f1,
            "accuracy": accuracy_score_,
            "minority_f1": minority_f1,
        }
    )


def knn_utility_fct(
    training_features,
    training_labels,
    validation_features,
    validation_labels,
    seed,
    hyperparams,
):
    """
    Compute KNN utility function.

    Parameters:
    -----------
    training_features : array-like
        Features of training data.
    training_labels : array-like
        Labels of training data.
    validation_features : array-like
        Features of validation data.
    validation_labels : array-like
        Labels of validation data.
    seed : int
        Random seed for reproducibility.
    hyperparams : dict
        Hyperparameters for KNN model.

    Returns:
    --------
    UtilityResult
        Object containing utility metrics including knn_loss, mcc, macro_f1, accuracy, and minority_f1.
    """
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import matthews_corrcoef

    K = hyperparams["K"]

    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(training_features, training_labels)

    preds = knn.predict(validation_features)

    distances, indices = knn.kneighbors(validation_features)

    neighbor_labels = training_labels[indices]

    matches = neighbor_labels == validation_labels[:, np.newaxis]
    fraction_matches = np.sum(matches, axis=1) / K
    losses = 1 - fraction_matches

    utility_score = np.mean(losses)

    mcc = matthews_corrcoef(validation_labels, preds)

    macro_f1 = f1_score(validation_labels, preds, average="macro")

    accuracy_score_ = accuracy_score(validation_labels, preds)

    unique_classes, class_counts = np.unique(validation_labels, return_counts=True)
    if len(unique_classes) == 2:
        minority_class = unique_classes[np.argmin(class_counts)]
        minority_f1 = f1_score(validation_labels, preds, pos_label=minority_class)
    else:
        minority_f1 = 0.0

    return utility.UtilityResult(
        {
            "knn_loss": utility_score,
            "mcc": mcc,
            "macro_f1": macro_f1,
            "accuracy": accuracy_score_,
            "minority_f1": minority_f1,
        }
    )


def random_data_minimization_multiple_Ks(
    hyperparams_configs,
    utility_fct_names,
    training_features,
    training_labels,
    prototypes_features,
    prototypes_labels,
    percentages,
    global_seed=global_seed,
    num_random_partitions=2,
    only_training_indices=False,
    ray_parallelization=None,
    utility_fct=knn_utility_fct,
    acquisition_mode=False,
):
    """
    Perform random data minimization for multiple hyperparameter configurations and aggregate results.

    Parameters:
    -----------
    hyperparams_configs : list
        List of hyperparameter configurations (dicts).
    utility_fct_names : list
        List of utility function names to evaluate.
    training_features : array-like
        Training data features.
    training_labels : array-like
        Training data labels.
    prototypes_features : array-like
        Prototype/test data features.
    prototypes_labels : array-like
        Prototype/test data labels.
    percentages : array-like
        Percentages of data to keep at each iteration.
    global_seed : int, optional
        Global random seed. Default is global_seed.
    num_random_partitions : int, optional
        Number of random partitions per percentage. Default is 2.
    only_training_indices : bool, optional
        Whether to use only training indices. Default is False.
    ray_parallelization : dict, optional
        Ray parallelization config. Default is None.
    utility_fct : callable, optional
        Utility function to use. Default is knn_utility_fct.
    acquisition_mode : bool, optional
        Whether to run in acquisition mode. Default is False.

    Returns:
    --------
    tuple
        (aggregated_score_means, aggregated_score_stds, aggregated_score_values_all_details)
    """
    # Initialize variables to store aggregated results
    aggregated_score_means = {}
    aggregated_score_stds = {}
    aggregated_score_values_all_details = (
        {}
    )  # Store detailed scores for each utility function

    # Iterate over each hyperparameter configuration
    for config in tqdm(hyperparams_configs):

        # Perform random data minimization for the current hyperparameter configuration
        random_score_means, random_score_stds, random_score_values_all_details = (
            data_values_analysis.random_data_minimization(
                utility_fct,
                training_features,
                training_labels,
                prototypes_features,
                prototypes_labels,
                global_seed,
                utility_fct_names,
                percentages,
                num_random_partitions=num_random_partitions,
                only_training_indices=only_training_indices,
                hyperparams=config,
                ray_parallelization=ray_parallelization,
                acquisition_mode=acquisition_mode,
            )
        )

        # Aggregate the results
        config_key = str(config)  # Convert configuration to string to use as a key
        aggregated_score_means[config_key] = random_score_means
        aggregated_score_stds[config_key] = random_score_stds
        aggregated_score_values_all_details[config_key] = (
            random_score_values_all_details
        )

    return (
        aggregated_score_means,
        aggregated_score_stds,
        aggregated_score_values_all_details,
    )


def run_only_random_minimization():
    """
    Run random data minimization experiments across multiple datasets.

    This function iterates through each dataset specified in dataset_names, performs random data minimization,
    and saves the results. For each dataset, it:
    1. Creates output directories if needed
    2. Loads training and test data
    3. Extracts features and labels
    4. Runs random minimization experiments
    5. Saves results and plots

    The minimization process gradually reduces the training set size while measuring model performance
    to analyze how the model degrades with less data.

    Global variables used:
    - dataset_names: List of dataset folder names to process
    - load_addition_data: Boolean flag for loading additional validation data
    - global_seed: Random seed for reproducibility
    - hyperparams_configs: List of hyperparameter configurations to test
    """
    for data_folder in dataset_names:
        print(f"Running for {data_folder}")

        figures_folder_name = f"{data_folder}data_minimization/"

        # test if figures_folder_name exists, if not create it

        if not os.path.exists(figures_folder_name):
            os.makedirs(figures_folder_name)

        if load_addition_data:
            df_training, test_df, df_dist_game_data_, feature_names, label_name = (
                data_loader.load(data_folder, load_addition_data=load_addition_data)
            )
            len_df_training = len(df_training)
            all_data_df = pd.concat([df_training, df_dist_game_data_])
            all_data_df = all_data_df.reset_index(drop=True)
            df_dist_game_data = all_data_df.iloc[len_df_training:]
        else:
            # Load the all_points.csv file
            all_data_df, test_df, feature_names, label_name = data_loader.load(
                data_folder
            )
            df_training, df_dist_game_data = train_test_split(
                all_data_df, test_size=0.1, random_state=global_seed
            )

        # Extracting training data and labels
        training_features = df_training[feature_names].values
        training_labels = df_training[label_name].values

        # Extracting data for plotting
        test_features = test_df[feature_names].values
        test_labels = test_df[label_name].values

        indices = np.arange(0, len(training_features))  # Indices of the training data

        # distances = cdist(training_features, test_features).T

        # utility_fct = lambda reduced_training_indices, validation_labels, hyperparams: knn_utility_fct(distances, training_labels, reduced_training_indices, validation_labels, hyperparams)

        utility_fct = knn_utility_fct

        # Verified that it goes from 100% to 0%
        random_score_means, random_score_stds, _ = load_or_compute_and_save(
            "Random Data Minimization",
            f"{figures_folder_name}/random_data_minimization_multiple_Ks.pkl",
            lambda: random_data_minimization_multiple_Ks(
                hyperparams_configs=hyperparams_configs,
                utility_fct_names=[
                    "knn_loss",
                    "mcc",
                    "macro_f1",
                    "accuracy",
                    "minority_f1",
                ],
                training_features=training_features,
                training_labels=training_labels,
                prototypes_features=test_features,
                prototypes_labels=test_labels,
                indices=indices,
                percentages=percentages,
                num_random_partitions=num_random_partitions_per_percent,
                global_seed=global_seed,
                only_training_indices=False,
                ray_parallelization={
                    "n_tasks": global_n_tasks,
                    "use_object_store": True,
                },
            ),
            overwrite=global_overwrite,
        )


def load_and_plot(figures_folder_name, hp, data_folder, load_addition_data):
    """
    Load data, run minimization and privacy evaluation, and plot results for a given configuration.

    Parameters:
    -----------
    figures_folder_name : str
        Folder to save figures and results.
    hp : str
        Hyperparameter configuration as a string.
    data_folder : str
        Name of the dataset folder.
    load_addition_data : bool
        Whether to load additional data.
    """
    config = eval(hp)

    if load_addition_data:
        df_training, test_df, df_dist_game_data_, feature_names, label_name = (
            data_loader.load(data_folder, load_addition_data=load_addition_data)
        )
        len_df_training = len(df_training)
        all_data_df = pd.concat([df_training, df_dist_game_data_])
        all_data_df = all_data_df.reset_index(drop=True)
        df_dist_game_data = all_data_df.iloc[len_df_training:]
    else:
        # Load the all_points.csv file
        all_data_df, test_df, feature_names, label_name = data_loader.load(data_folder)
        df_training, df_dist_game_data = train_test_split(
            all_data_df, test_size=0.1, random_state=global_seed
        )

    # Extracting training data and labels
    training_features = df_training[feature_names].values
    training_labels = df_training[label_name].values

    # Extracting data for plotting
    test_features = test_df[feature_names].values
    test_labels = test_df[label_name].values

    utility_fct = knn_utility_fct

    batch_attacks = True
    selected_attack = build_attacker_predict_lira_all_scores(
        all_data_df[feature_names].values,
        all_data_df[label_name].values,
        N_shadow=N_shadow,
    )

    indices = np.arange(0, len(training_features))  # Indices of the training data

    seed_list = np.arange(global_seed, global_seed + number_of_games)

    # Verified that it goes from 100% to 0%
    random_score_means, random_score_stds, _ = load_or_compute_and_save(
        "Random Data Minimization",
        f"{figures_folder_name}/random_data_minimization_multiple_{hp}.pkl",
        lambda: random_data_minimization_multiple_Ks(
            hyperparams_configs=[config],
            utility_fct_names=[
                "knn_loss",
                "mcc",
                "macro_f1",
                "accuracy",
                "minority_f1",
            ],
            training_features=training_features,
            training_labels=training_labels,
            prototypes_features=test_features,
            prototypes_labels=test_labels,
            indices=indices,
            percentages=percentages,
            num_random_partitions=num_random_partitions_per_percent,
            global_seed=global_seed,
            only_training_indices=False,
            ray_parallelization={"n_tasks": global_n_tasks, "use_object_store": True},
            utility_fct=utility_fct,
        ),
        overwrite=False,
    )

    privacy_removal_indices = load_or_compute_and_save(
        "Iterative LiRA",
        f"{figures_folder_name}/iterative_lira_results_{hp}.pkl",
        lambda: iterative_vanilla_lira_removal(
            training_features,
            training_labels,
            percentages,
            global_seed,
            config,
            ray_parallelization={"n_tasks": global_n_tasks},
        ),
        overwrite=False,
    )

    privacy_removal_indices_results_with_indices = (
        data_values_analysis.percentage_data_minimization_by_indices(
            percentages,
            privacy_removal_indices,
            training_features,
            training_labels,
            test_features,
            test_labels,
            global_seed,
            utility_fct,
            only_training_indices=False,
            hyperparams=config,
            return_indices=True,
        )
    )

    privacy_removal_percents_attack_results = load_or_compute_and_save(
        "Evaluating privacy with privacy removal",
        f"{figures_folder_name}/privacy_removal_attack_results_{hp}.pkl",
        lambda: evaluate_privacy_by_indices(
            selected_attack,
            privacy_removal_indices_results_with_indices,
            df_training,
            df_dist_game_data,
            config,
            feature_names,
            label_name,
            global_seed,
            ray_parallelization={"n_tasks": global_n_tasks},
            batch=batch_attacks,
        ),
        overwrite=False,
    )

    average_shapley_values = load_or_compute_and_save(
        "Average Shapley Values",
        f"{figures_folder_name}/average_shapley_values_{hp}.pkl",
        lambda: knn_valuation.compute_self_training_average_shapley_values(
            training_features, training_labels, config["K"]
        ),
        overwrite=False,
    )

    average_shapley_values = (average_shapley_values - average_shapley_values.min()) / (
        average_shapley_values.max() - average_shapley_values.min()
    )

    shapley_descend_value_indices = np.argsort(average_shapley_values)[::-1]

    # removal - Suposed to go from 100% to 0%
    shapley_descend_results_with_indices = (
        data_values_analysis.percentage_data_minimization_by_indices(
            percentages,
            shapley_descend_value_indices,
            training_features,
            training_labels,
            test_features,
            test_labels,
            global_seed,
            utility_fct,
            only_training_indices=False,
            hyperparams=config,
            return_indices=True,
        )
    )

    shapley_descend_results = [s for s, _ in shapley_descend_results_with_indices]

    average_waka_values = load_or_compute_and_save(
        "Average Waka Values",
        f"{figures_folder_name}/average_waka_values_{hp}.pkl",
        lambda: compute_self_unormalized_average_waka_values(
            training_features,
            training_labels,
            config["K"],
            approx="quantile",
            ray_parallelization=None,  # {"n_tasks":global_n_tasks},
            max_quantile=waka_max_quantile,
            max_contributors=waka_max_contributors,
        ),
        overwrite=False,
    )

    # perform min-max scaling on average_waka_values
    average_waka_values = (average_waka_values - average_waka_values.min()) / (
        average_waka_values.max() - average_waka_values.min()
    )

    # 0 to 1 (cutoff is from the tail)
    waka_value_ascend_indices = np.argsort(average_waka_values)

    waka_ascend_results_with_indices = (
        data_values_analysis.percentage_data_minimization_by_indices(
            percentages,
            waka_value_ascend_indices,
            training_features,
            training_labels,
            test_features,
            test_labels,
            global_seed,
            utility_fct,
            only_training_indices=False,
            hyperparams=config,
            return_indices=True,
        )
    )

    waka_ascend_percents_attack_results = load_or_compute_and_save(
        "Evaluating privacy with waka",
        f"{figures_folder_name}/waka_percents_attack_results_{hp}.pkl",
        lambda: evaluate_privacy_by_indices(
            selected_attack,
            waka_ascend_results_with_indices,
            df_training,
            df_dist_game_data,
            config,
            feature_names,
            label_name,
            global_seed,
            ray_parallelization={"n_tasks": global_n_tasks},
            batch=batch_attacks,
        ),
        overwrite=False,
    )

    waka_ascend_results = [s for s, _ in waka_ascend_results_with_indices]

    data_values_analysis.plot_value_based_minimization(
        {
            "DSV Descending": shapley_descend_results,
            "WaKA Ascending": waka_ascend_results,
        },
        ["knn_loss", "mcc", "macro_f1", "accuracy", "minority_f1"],
        percentages,
        random_score_means[hp],
        random_score_stds[hp],
        dataset_folder_name=figures_folder_name,
        show_plot=False,
        complete_plot_name=f"Random Data Minimization for KNN {hp} (tail cutoff)",
    )

    # The membership ratio suposed to be from 100% to 0%
    percents_random_removal_attack_scores = load_or_compute_and_save(
        "Random Removal",
        f"{figures_folder_name}/percents_random_removal_attack_results_{hp}.pkl",
        lambda: [
            run_multiple_games(
                df_training,
                config["K"],
                selected_attack,
                number_of_points_per_game,
                seed_list,
                percent / 100,
                feature_names=feature_names,
                label_name=label_name,
                ray_parallelization={"n_tasks": global_n_tasks},
                batch=batch_attacks,
            )
            for percent in percentages[1:]
        ],
        overwrite=False,
    )

    percents_random_removal_attack_results = [
        compute_auc_for_scores_with_ci(seed_list, percent_scores)
        for percent_scores in percents_random_removal_attack_scores
    ]

    return (
        percents_random_removal_attack_results,
        waka_ascend_percents_attack_results,
        privacy_removal_percents_attack_results,
        shapley_descend_results,
        waka_ascend_results,
        random_score_means[hp],
        random_score_stds[hp],
        percentages,
    )


def run_all(utility_driven=False, privacy_driven=True):
    """
    Run all utility-driven and privacy-driven minimization and evaluation experiments for all datasets.

    Parameters:
    -----------
    utility_driven : bool, optional
        Whether to run utility-driven experiments. Default is False.
    privacy_driven : bool, optional
        Whether to run privacy-driven experiments. Default is True.
    """
    for dataset_name in dataset_names:
        print(f"Running for {dataset_name}")

        current_datetime = datetime.datetime.now()
        date_string = current_datetime.strftime("%Y-%m-%d %H:%M")

        figures_folder_name = f"{dataset_name}/data_minimization"

        if add_folder_timestamp:
            figures_folder_name = f"{figures_folder_name}_{date_string}/"
        else:
            figures_folder_name = f"{figures_folder_name}/"

        # test if figures_folder_name exists, if not create it

        if not os.path.exists(figures_folder_name):
            os.makedirs(figures_folder_name)

        # Add logic for ag_news dataset
        if dataset_name == "ag_news":
            load_addition_data = True
        else:
            load_addition_data = False

        all_data_df, test_df, feature_names, label_name = data_loader.load(dataset_name)

        df_training, df_dist_game_data = train_test_split(
            all_data_df, test_size=0.1, random_state=global_seed
        )

        # Extracting training data and labels
        training_features = df_training[feature_names].values
        training_labels = df_training[label_name].values

        # Extracting data for plotting
        test_features = test_df[feature_names].values
        test_labels = test_df[label_name].values

        utility_fct = knn_utility_fct

        batch_attacks = True
        lira_attack_fct = build_attacker_predict_lira_all_scores(
            all_data_df[feature_names].values,
            all_data_df[label_name].values,
            N_shadow=N_shadow,
        )

        seed_list = np.arange(global_seed, global_seed + number_of_games)

        for config in hyperparams_configs:

            hp_str = str(config)

            print(f"Running for {hp_str}")

            # Verified that it goes from 100% to 0%
            utility_random_score_means, utility_random_score_stds, _ = (
                load_or_compute_and_save(
                    "Random Data Minimization",
                    f"{figures_folder_name}/random_data_minimization_multiple_{hp_str}.pkl",
                    lambda: random_data_minimization_multiple_Ks(
                        hyperparams_configs=[config],
                        utility_fct_names=[
                            "knn_loss",
                            "mcc",
                            "macro_f1",
                            "accuracy",
                            "minority_f1",
                        ],
                        training_features=training_features,
                        training_labels=training_labels,
                        prototypes_features=test_features,
                        prototypes_labels=test_labels,
                        percentages=utility_percentages,
                        num_random_partitions=num_random_partitions_per_percent,
                        global_seed=global_seed,
                        only_training_indices=False,
                        ray_parallelization={
                            "n_tasks": global_n_tasks,
                            "use_object_store": True,
                        },
                        utility_fct=utility_fct,
                    ),
                    overwrite=global_overwrite and common_prep_overwrite,
                )
            )

            load_values_from_csv = True
            if not load_values_from_csv:
                test_shapley_values = load_or_compute_and_save(
                    "Average Shapley Values",
                    f"{figures_folder_name}/average_shapley_values_{hp_str}.pkl",
                    lambda: knn_valuation.compute_training_average_shapley_values(
                        training_features,
                        training_labels,
                        val_features,
                        val_labels,
                        config["K"],
                        only_average=True,
                    ),
                    overwrite=global_overwrite,
                )

                test_waka_values = load_or_compute_and_save(
                    "Average Waka Values",
                    f"{figures_folder_name}/average_waka_values_{hp_str}.pkl",
                    lambda: compute_unormalized_average_waka_values(
                        training_features,
                        training_labels,
                        val_features,
                        val_labels,
                        config["K"],
                        approx="quantile",
                        ray_parallelization={"n_tasks": global_n_tasks},
                    ),
                    overwrite=global_overwrite,
                )

                test_waka_values = np.array(test_waka_values)

                binary_labels_comparison = np.array(
                    [
                        [
                            1 if train_label == test_label else 0
                            for test_label in val_labels
                        ]
                        for train_label in training_labels
                    ]
                )

                # Transpose the binary_labels_comparison array to align the dimensions correctly
                binary_labels_comparison_transposed = binary_labels_comparison.T

                # Create a weight matrix where matching labels (1) are multiplied by -1 and non-matching labels (0) are multiplied by 1
                weight_matrix = np.where(
                    binary_labels_comparison_transposed == 1, 1, -1
                )

                # Perform the element-wise multiplication with the weight matrix and then sum along the appropriate axis
                test_waka_values = np.sum(test_waka_values * weight_matrix, axis=0)

                self_shapley_values = load_or_compute_and_save(
                    "Average Shapley Values",
                    f"{figures_folder_name}/privacy_driven_average_shapley_values_{hp_str}.pkl",
                    lambda: knn_valuation.compute_self_training_average_shapley_values(
                        training_features, training_labels, config["K"]
                    ),
                    overwrite=global_overwrite,
                )

                self_waka_values = load_or_compute_and_save(
                    "Average Waka Values",
                    f"{figures_folder_name}/average_self_waka_values_{hp_str}.pkl",
                    lambda: compute_self_unormalized_average_waka_values(
                        training_features,
                        training_labels,
                        config["K"],
                        approx="quantile",
                        ray_parallelization={"n_tasks": global_n_tasks},
                    ),
                    overwrite=global_overwrite,
                )

                self_waka_values = np.array(self_waka_values)

            else:

                file_path = f'/Users/patrickmesana/Dev/waka-experiments/{dataset_name}/multiple_attacks/K{config["K"]}/evaluation_k_{config["K"]}.pkl'
                data = pd.read_pickle(file_path)
                # sort data by the indices in the data

                self_shapley_values = data["self_shapley_values"]
                test_shapley_values = data["av_test_shapley_values_avg"]
                self_waka_values = data["self_waka_values"]
                test_waka_values = data["av_test_waka_values_avg"]

            test_shapley_values = (test_shapley_values - test_shapley_values.min()) / (
                test_shapley_values.max() - test_shapley_values.min()
            )
            test_waka_values = (test_waka_values - test_waka_values.min()) / (
                test_waka_values.max() - test_waka_values.max()
            )
            self_waka_values = (self_waka_values - self_waka_values.min()) / (
                self_waka_values.max() - self_waka_values.min()
            )

            # indices of ascending order of average shapley values. 1 to 0 (cutoff is from the tail)

            # Utility driven attribution
            test_shapley_descend_value_indices = np.argsort(test_shapley_values)[::-1]
            test_waka_value_descend_indices = np.argsort(test_waka_values)[::-1]

            # Privacy driven attribution
            self_shapley_value_ascend_indices = np.argsort(self_shapley_values)
            self_waka_value_ascend_indices = np.argsort(self_waka_values)

            # removal - Suposed to go from 100% to 0%
            test_shapley_descend_results_with_indices = (
                data_values_analysis.percentage_data_minimization_by_indices(
                    utility_percentages,
                    test_shapley_descend_value_indices,
                    training_features,
                    training_labels,
                    test_features,
                    test_labels,
                    global_seed,
                    utility_fct,
                    only_training_indices=False,
                    hyperparams=config,
                    return_indices=True,
                )
            )

            test_shapley_descend_results = [
                s for s, _ in test_shapley_descend_results_with_indices
            ]

            test_waka_descend_results_with_indices = (
                data_values_analysis.percentage_data_minimization_by_indices(
                    utility_percentages,
                    test_waka_value_descend_indices,
                    training_features,
                    training_labels,
                    test_features,
                    test_labels,
                    global_seed,
                    utility_fct,
                    only_training_indices=False,
                    hyperparams=config,
                    return_indices=True,
                )
            )
            test_waka_descend_results = [
                s for s, _ in test_waka_descend_results_with_indices
            ]

            # removal - Suposed to go from 100% to 0%
            self_shapley_results_with_indices = (
                data_values_analysis.percentage_data_minimization_by_indices(
                    utility_percentages,
                    self_shapley_value_ascend_indices,
                    training_features,
                    training_labels,
                    test_features,
                    test_labels,
                    global_seed,
                    utility_fct,
                    only_training_indices=False,
                    hyperparams=config,
                    return_indices=True,
                )
            )

            self_shapley_ascend_results = [
                s for s, _ in self_shapley_results_with_indices
            ]

            self_waka_ascend_results_with_indices = (
                data_values_analysis.percentage_data_minimization_by_indices(
                    utility_percentages,
                    self_waka_value_ascend_indices,
                    training_features,
                    training_labels,
                    test_features,
                    test_labels,
                    global_seed,
                    utility_fct,
                    only_training_indices=False,
                    hyperparams=config,
                    return_indices=True,
                )
            )

            self_waka_ascend_results = [
                s for s, _ in self_waka_ascend_results_with_indices
            ]

            data_values_analysis.plot_value_based_minimization(
                {
                    "DSV Descending": test_shapley_descend_results,
                    "WaKA Descending": test_waka_descend_results,
                    "Self DSV Ascending": self_shapley_ascend_results,
                    "Self WaKA Ascending": self_waka_ascend_results,
                },
                ["knn_loss", "mcc", "macro_f1", "accuracy", "minority_f1"],
                utility_percentages,
                utility_random_score_means[hp_str],
                utility_random_score_stds[hp_str],
                dataset_folder_name=figures_folder_name,
                show_plot=False,
                complete_plot_name=f"Random Data Minimization for KNN {hp_str} (tail cutoff)",
            )

            ray.init()
            # The membership ratio suposed to be from 100% to 0%
            random_removal_attack_results = load_or_compute_and_save(
                "Random Removal",
                f"{figures_folder_name}/percents_random_removal_attack_results_{hp_str}.pkl",
                lambda: [
                    run_multiple_games(
                        df_training,
                        config["K"],
                        lira_attack_fct,
                        number_of_points_per_game,
                        seed_list,
                        percent / 100,
                        feature_names=feature_names,
                        label_name=label_name,
                        ray_parallelization={"n_tasks": global_n_tasks},
                        batch=batch_attacks,
                        ray_already_init=True,
                    )
                    for percent in privacy_percentages[1:]
                ],
                overwrite=global_overwrite and common_prep_overwrite,
            )
            ray.shutdown()

            percents_random_removal_attack_results = [
                compute_auc_for_scores_with_ci(seed_list, percent_scores)
                for percent_scores in random_removal_attack_results
            ]

            # verified that the indices are still in the right order - 100% to 0%
            test_shapley_descend_percents_attack_results = load_or_compute_and_save(
                "Evaluating privacy with negative shapley values",
                f"{figures_folder_name}/test_shapley_percents_attack_results_{hp_str}.pkl",
                lambda: evaluate_privacy_by_indices(
                    lira_attack_fct,
                    test_shapley_descend_results_with_indices,
                    df_training,
                    df_dist_game_data,
                    config,
                    feature_names,
                    label_name,
                    global_seed,
                    ray_parallelization={"n_tasks": global_n_tasks},
                    batch=batch_attacks,
                ),
                overwrite=global_overwrite,
            )

            test_waka_descend_percents_attack_results = load_or_compute_and_save(
                "Evaluating privacy with waka",
                f"{figures_folder_name}/test_waka_percents_attack_results_{hp_str}.pkl",
                lambda: evaluate_privacy_by_indices(
                    lira_attack_fct,
                    test_waka_descend_results_with_indices,
                    df_training,
                    df_dist_game_data,
                    config,
                    feature_names,
                    label_name,
                    global_seed,
                    ray_parallelization={"n_tasks": global_n_tasks},
                    batch=batch_attacks,
                ),
                overwrite=global_overwrite,
            )

            # verified that the indices are still in the right order - 100% to 0%
            self_shapley_ascend_percents_attack_results = load_or_compute_and_save(
                "Evaluating privacy with shapley values",
                f"{figures_folder_name}/self_shapley_percents_attack_results_{hp_str}.pkl",
                lambda: evaluate_privacy_by_indices(
                    lira_attack_fct,
                    self_shapley_results_with_indices,
                    df_training,
                    df_dist_game_data,
                    config,
                    feature_names,
                    label_name,
                    global_seed,
                    ray_parallelization={"n_tasks": global_n_tasks},
                    batch=batch_attacks,
                ),
                overwrite=global_overwrite,
            )

            self_waka_ascend_percents_attack_results = load_or_compute_and_save(
                "Evaluating privacy with waka",
                f"{figures_folder_name}/self_waka_percents_attack_results_{hp_str}.pkl",
                lambda: evaluate_privacy_by_indices(
                    lira_attack_fct,
                    self_waka_ascend_results_with_indices,
                    df_training,
                    df_dist_game_data,
                    config,
                    feature_names,
                    label_name,
                    global_seed,
                    ray_parallelization={"n_tasks": global_n_tasks},
                    batch=batch_attacks,
                ),
                overwrite=global_overwrite,
            )

            # plot the mean_auc and std_auc  of percents_random_removal_attack_results as we remove more and more data, remember that we start from percentages[1]. Also i need to plot percents_attack_results
            plt.figure()
            plt.plot(
                privacy_percentages[1:],
                [
                    mean_auc
                    for _, _, _, _, mean_auc, _, _ in percents_random_removal_attack_results
                ],
                label="Random Removal",
            )
            plt.fill_between(
                privacy_percentages[1:],
                [
                    ci_lower
                    for _, _, _, _, mean_auc, ci_lower, ci_upper in percents_random_removal_attack_results
                ],
                [
                    ci_upper
                    for _, _, _, _, mean_auc, ci_lower, ci_upper in percents_random_removal_attack_results
                ],
                alpha=0.2,
            )

            plt.plot(
                privacy_percentages[1:],
                [
                    mean_auc
                    for _, _, mean_auc in test_shapley_descend_percents_attack_results
                ],
                label="DSV Desending",
            )

            plt.plot(
                privacy_percentages[1:],
                [
                    mean_auc
                    for _, _, mean_auc in test_waka_descend_percents_attack_results
                ],
                label="WaKA Descending",
            )

            plt.plot(
                privacy_percentages[1:],
                [
                    mean_auc
                    for _, _, mean_auc in self_shapley_ascend_percents_attack_results
                ],
                label="Self DSV",
            )

            plt.plot(
                privacy_percentages[1:],
                [
                    mean_auc
                    for _, _, mean_auc in self_waka_ascend_percents_attack_results
                ],
                label="Self WaKA",
            )

            plt.xlabel("Percentage of Dataset Preserved (Tail Cutoff)")

            plt.gca().invert_xaxis()  # Invert X axis to show 100% at the left

            plt.ylabel("AUC")
            plt.legend()
            plt.title(f" Attacks AUC for KNN {hp_str}")
            plt.savefig(f"{figures_folder_name}/Attacks AUC for KNN {hp_str}.png")
            plt.close()

            lira_iterative_removal_indices = load_or_compute_and_save(
                "Iterative LiRA",
                f"{figures_folder_name}/iterative_lira_results_{hp_str}.pkl",
                lambda: iterative_vanilla_lira_removal(
                    training_features,
                    training_labels,
                    percentages,
                    global_seed,
                    config,
                    ray_parallelization={"n_tasks": global_n_tasks},
                ),
                overwrite=global_overwrite and common_prep_overwrite,
            )

            lira_iterative_emoval_indices_results_with_indices = (
                data_values_analysis.percentage_data_minimization_by_indices(
                    percentages,
                    lira_iterative_removal_indices,
                    training_features,
                    training_labels,
                    test_features,
                    test_labels,
                    global_seed,
                    utility_fct,
                    only_training_indices=False,
                    hyperparams=config,
                    return_indices=True,
                )
            )

            lira_iterative_removal_percents_attack_results = load_or_compute_and_save(
                "Evaluating privacy with privacy removal",
                f"{figures_folder_name}/privacy_removal_attack_results_{hp_str}.pkl",
                lambda: evaluate_privacy_by_indices(
                    lira_attack_fct,
                    lira_iterative_emoval_indices_results_with_indices,
                    df_training,
                    df_dist_game_data,
                    config,
                    feature_names,
                    label_name,
                    global_seed,
                    ray_parallelization={"n_tasks": global_n_tasks},
                    batch=batch_attacks,
                ),
                overwrite=global_overwrite and common_prep_overwrite,
            )

            waka_iterative_removal_indices = load_or_compute_and_save(
                "Iterative WaKA",
                f"{figures_folder_name}/iterative_waka_results_{hp_str}.pkl",
                lambda: iterative_waka_removal(
                    training_features,
                    training_labels,
                    indices,
                    test_features,
                    test_labels,
                    percentages,
                    global_seed,
                    config,
                    ray_parallelization={"n_tasks": global_n_tasks},
                ),
                overwrite=global_overwrite,
            )

            waka_iterative_emoval_indices_results_with_indices = (
                data_values_analysis.percentage_data_minimization_by_indices(
                    percentages,
                    waka_iterative_removal_indices,
                    training_features,
                    training_labels,
                    test_features,
                    test_labels,
                    global_seed,
                    utility_fct,
                    only_training_indices=False,
                    hyperparams=config,
                    return_indices=True,
                )
            )

            waka_iterative_removal_percents_attack_results = load_or_compute_and_save(
                "Evaluating privacy with privacy removal",
                f"{figures_folder_name}/privacy_removal_attack_results_{hp_str}.pkl",
                lambda: evaluate_privacy_by_indices(
                    lira_attack_fct,
                    waka_iterative_emoval_indices_results_with_indices,
                    df_training,
                    df_dist_game_data,
                    config,
                    feature_names,
                    label_name,
                    global_seed,
                    ray_parallelization={"n_tasks": global_n_tasks},
                    batch=batch_attacks,
                ),
                overwrite=global_overwrite,
            )

            lira_iterative_removal_indices = load_or_compute_and_save(
                "Iterative LiRA",
                f"{figures_folder_name}/iterative_lira_results_{hp_str}.pkl",
                lambda: iterative_vanilla_lira_removal(
                    training_features,
                    training_labels,
                    percentages,
                    global_seed,
                    config,
                    ray_parallelization={"n_tasks": global_n_tasks},
                ),
                overwrite=global_overwrite and common_prep_overwrite,
            )

            lira_iterative_emoval_indices_results_with_indices = (
                data_values_analysis.percentage_data_minimization_by_indices(
                    percentages,
                    lira_iterative_removal_indices,
                    training_features,
                    training_labels,
                    test_features,
                    test_labels,
                    global_seed,
                    utility_fct,
                    only_training_indices=False,
                    hyperparams=config,
                    return_indices=True,
                )
            )

            lira_iterative_removal_percents_attack_results = load_or_compute_and_save(
                "Evaluating privacy with privacy removal",
                f"{figures_folder_name}/privacy_removal_attack_results_{hp_str}.pkl",
                lambda: evaluate_privacy_by_indices(
                    lira_attack_fct,
                    lira_iterative_emoval_indices_results_with_indices,
                    df_training,
                    df_dist_game_data,
                    config,
                    feature_names,
                    label_name,
                    global_seed,
                    ray_parallelization={"n_tasks": global_n_tasks},
                    batch=batch_attacks,
                ),
                overwrite=global_overwrite and common_prep_overwrite,
            )

            waka_iterative_removal_indices = load_or_compute_and_save(
                "Iterative LiRA",
                f"{figures_folder_name}/iterative_lira_results_{hp_str}.pkl",
                lambda: iterative_waka_removal(
                    training_features,
                    training_labels,
                    indices,
                    percentages,
                    global_seed,
                    config,
                    ray_parallelization={"n_tasks": global_n_tasks},
                ),
                overwrite=global_overwrite and common_prep_overwrite,
            )

            waka_removal_indices_results_with_indices = (
                data_values_analysis.percentage_data_minimization_by_indices(
                    percentages,
                    waka_iterative_removal_indices,
                    training_features,
                    training_labels,
                    test_features,
                    test_labels,
                    global_seed,
                    utility_fct,
                    only_training_indices=False,
                    hyperparams=config,
                    return_indices=True,
                )
            )

            waka_iterative_removal_percents_attack_results = load_or_compute_and_save(
                "Evaluating privacy with privacy removal",
                f"{figures_folder_name}/waka_iterative_removal_attack_results_{hp_str}.pkl",
                lambda: evaluate_privacy_by_indices(
                    lira_attack_fct,
                    waka_removal_indices_results_with_indices,
                    df_training,
                    df_dist_game_data,
                    config,
                    feature_names,
                    label_name,
                    global_seed,
                    ray_parallelization={"n_tasks": global_n_tasks},
                    batch=batch_attacks,
                ),
                overwrite=global_overwrite and common_prep_overwrite,
            )


def values_for_dataset_and_param(
    training_features,
    training_labels,
    val_features,
    val_labels,
    test_features,
    test_labels,
    utility_fct,
    config,
    hp_str,
    figures_folder_name,
    overwrite_utility_minimization,
    overwrite_values,
    waka_strat="strat1",
    waka_overwrite=False,
    global_n_tasks=4,
    Tau=None,
):
    """
    Compute and plot value functions (Shapley, LOO, WaKA) for a dataset and configuration.

    Parameters:
    -----------
    training_features : array-like
        Training data features.
    training_labels : array-like
        Training data labels.
    val_features : array-like
        Validation data features.
    val_labels : array-like
        Validation data labels.
    test_features : array-like
        Test data features.
    test_labels : array-like
        Test data labels.
    utility_fct : callable
        Utility function to use.
    config : dict
        Hyperparameter configuration.
    hp_str : str
        String representation of hyperparameters.
    figures_folder_name : str
        Folder to save figures and results.
    overwrite_utility_minimization : bool
        Whether to overwrite utility minimization results.
    overwrite_values : bool
        Whether to overwrite value computation results.
    waka_strat : str, optional
        WaKA strategy. Default is "strat1".
    waka_overwrite : bool, optional
        Whether to overwrite WaKA results. Default is False.
    global_n_tasks : int, optional
        Number of parallel tasks. Default is 4.
    Tau : float, optional
        Tau parameter for WaKA. Default is None.

    Returns:
    --------
    tuple
        (test_shapley_values, test_waka_values, test_loo_values, self_shapley_values, self_waka_values, self_loo_values)
    """
    test_shapley_values = load_or_compute_and_save(
        "Test Average Shapley Values",
        f"{figures_folder_name}/test_average_shapley_values_{hp_str}.pkl",
        lambda: knn_valuation.compute_training_average_shapley_values(
            training_features,
            training_labels,
            val_features,
            val_labels,
            config["K"],
            only_average=True,
        ),
        overwrite=overwrite_values,
    )

    test_loo_values = load_or_compute_and_save(
        "Test Average LOO Values",
        f"{figures_folder_name}/test_average_loo_values_{hp_str}.pkl",
        lambda: knn_valuation.compute_training_average_leave_one_out(
            training_features,
            training_labels,
            val_features,
            val_labels,
            config["K"],
            only_average=True,
        ),
        overwrite=overwrite_values,
    )

    test_waka_values = load_or_compute_and_save(
        "Test Average Waka Values",
        f"{figures_folder_name}/test_average_waka_values_{hp_str}_tau_{Tau}.pkl",
        lambda: compute_unormalized_average_waka_values(
            training_features,
            training_labels,
            val_features,
            val_labels,
            config["K"],
            approx="quantile",
            strat=waka_strat,
            Tau=Tau,
            ray_parallelization=None,  # {"n_tasks":global_n_tasks}, #TODO : REMOVE TO PARALLELIZE
        ),
        overwrite=overwrite_values or waka_overwrite,
    )

    test_waka_values = np.array(test_waka_values)

    binary_labels_comparison = np.array(
        [
            [1 if train_label == val_label else 0 for val_label in val_labels]
            for train_label in training_labels
        ]
    )

    # Transpose the binary_labels_comparison array to align the dimensions correctly
    binary_labels_comparison_transposed = binary_labels_comparison.T

    # Create a weight matrix where matching labels (1) are multiplied by -1 and non-matching labels (0) are multiplied by 1
    weight_matrix = np.where(binary_labels_comparison_transposed == 1, 1, -1)

    # Perform the element-wise multiplication with the weight matrix and then sum along the appropriate axis
    test_waka_values = np.sum(test_waka_values * weight_matrix, axis=0)

    # plot distribution of test_waka_values
    plot_distribution(
        test_waka_values,
        f"{figures_folder_name}/test_waka_values_distribution_{hp_str}.png",
        "Test WaKA Values",
        "Density",
    )
    plot_distribution(
        test_shapley_values,
        f"{figures_folder_name}/test_shapley_values_distribution_{hp_str}.png",
        "Test Shapley Values",
        "Density",
    )

    # Plot for WaKA values
    plot_distribution_with_labels(
        test_waka_values,
        training_labels,
        f"{figures_folder_name}/test_waka_values_distribution_conditioned_{hp_str}.png",
        "Test WaKA Values (Conditioned on Labels)",
        "WaKA Values",
    )

    # Plot for Shapley values
    plot_distribution_with_labels(
        test_shapley_values,
        training_labels,
        f"{figures_folder_name}/test_shapley_values_distribution_conditioned_{hp_str}.png",
        "Test Shapley Values (Conditioned on Labels)",
        "Shapley Values",
    )

    self_shapley_values = load_or_compute_and_save(
        "Average Shapley Values",
        f"{figures_folder_name}/self_shapley_values_{hp_str}.pkl",
        lambda: knn_valuation.compute_self_training_average_shapley_values(
            training_features, training_labels, config["K"]
        ),
        overwrite=overwrite_values,
    )

    self_loo_values = load_or_compute_and_save(
        "Average LOO Values",
        f"{figures_folder_name}/self_loo_values_{hp_str}.pkl",
        lambda: knn_valuation.compute_self_training_average_leave_one_out(
            training_features, training_labels, config["K"]
        ),
        overwrite=overwrite_values,
    )

    self_waka_values_with_influences = load_or_compute_and_save(
        "Self Average Waka Values",
        f"{figures_folder_name}/self_waka_values_{hp_str}.pkl",
        lambda: compute_self_unormalized_average_waka_values_recomputable(
            training_features,
            training_labels,
            config["K"],
            ray_parallelization={"n_tasks": global_n_tasks},
        ),
        overwrite=overwrite_values,
    )

    self_waka_values = np.array([swv[0] for swv in self_waka_values_with_influences])

    self_waka_values = np.array(self_waka_values)

    test_shapley_values = (test_shapley_values - test_shapley_values.min()) / (
        test_shapley_values.max() - test_shapley_values.min()
    )
    test_waka_values = (test_waka_values - test_waka_values.min()) / (
        test_waka_values.max() - test_waka_values.min()
    )
    self_waka_values = (self_waka_values - self_waka_values.min()) / (
        self_waka_values.max() - self_waka_values.min()
    )

    return (
        test_shapley_values,
        test_waka_values,
        test_loo_values,
        self_shapley_values,
        self_waka_values,
        self_loo_values,
    )


def run_only_utility_data_acquisition_tau_exp(
    dataset_names,
    hyperparams_configs,
    overwrite_values=True,
    return_results=False,
    utility_percentages=np.arange(0.1, 20, 0.1),
    seeds=[42],
    overwrite_utility_minimization=False,
    add_folder_timestamp=False,
    waka_overwrite=False,
    global_n_tasks=4,
):
    """
    Run data acquisition experiments using utility-based value functions for tau experiment.
    Evaluates the impact of different tau values on data acquisition performance.

    This function performs data acquisition experiments specifically designed to evaluate the impact
    of different tau values on the WaKA method's performance. It tests multiple tau values derived
    from the K parameter (tau = a/K for a in range(K+1)) to understand how this parameter affects
    data acquisition effectiveness.

    Parameters:
    -----------
    dataset_names : list
        List of dataset names to run experiments on.
    hyperparams_configs : list
        List of hyperparameter configurations to test.
    overwrite_values : bool, optional
        Whether to overwrite existing value computations. Default is True.
    return_results : bool, optional
        Whether to return the results. Default is False.
    utility_percentages : array-like, optional
        Percentages of data to acquire. Default is np.arange(0.1, 20, 0.1).
    seeds : list, optional
        List of random seeds for reproducibility. Default is [42].
    overwrite_utility_minimization : bool, optional
        Whether to overwrite existing minimization results. Default is False.
    add_folder_timestamp : bool, optional
        Whether to add timestamp to output folder names. Default is False.
    waka_overwrite : bool, optional
        Whether to overwrite existing WaKA results. Default is False.
    global_n_tasks : int, optional
        Number of parallel tasks for computation. Default is 4.

    Returns:
    --------
    dict, optional
        If return_results is True, returns a dictionary containing:
        - utility: Aggregated utility results for each method and metric
        - random_means: Mean random baseline results
        - random_stds: Standard deviation of random baseline results
        - label_ratios: Label ratio evolution for each method

    Notes:
    ------
    - The function focuses on evaluating the WaKA method with different tau values
    - Tau values are computed as a/K where a ranges from 0 to K
    - Results are visualized with different opacities for each tau value
    - Performance is measured using multiple metrics: MCC, accuracy, macro F1, minority F1, and KNN loss
    - Random baseline results are computed for comparison
    - Results are saved as plots in dataset-specific folders with seed-specific subfolders
    """
    num_random_partitions_per_percent = 5
    metrics = ["mcc", "accuracy", "macro_f1", "minority_f1", "knn_loss"]

    if return_results:
        results = {}

    for dataset_name in dataset_names:
        print(f"Running for {dataset_name}")

        if return_results:
            results[dataset_name] = {}

        figures_folder_name = f"{dataset_name}/data_minimization/utility"
        if add_folder_timestamp:
            figures_folder_name = f'{figures_folder_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}/'
        else:
            figures_folder_name = f"{figures_folder_name}/"

        if not os.path.exists(figures_folder_name):
            os.makedirs(figures_folder_name)

        df_training, test_df_, feature_names, label_name = data_loader.load(
            dataset_name
        )

        for config in hyperparams_configs:
            K = config["K"]
            tau_values = [a / K for a in range(K + 1)]
            hp_str = str(config)
            print(f"Running for {hp_str}")

            # Generate colors with different opacities
            result_colors = {
                f"WaKA-{tau:.1f}": (
                    0,
                    0,
                    0.5,
                    (i + 1) / len(tau_values),
                )  # Dark blue with increasing opacity
                for i, tau in enumerate(tau_values)
            }

            if return_results:
                results[dataset_name][hp_str] = {}

            # Store results for each seed
            seed_results = {f"test_waka_tau_{tau:.1f}": [] for tau in tau_values}
            seed_results["label_ratios"] = {f"waka_{tau:.1f}": [] for tau in tau_values}

            seed_random_means = []
            seed_random_stds = []

            for seed in seeds:
                seed_folder = f"{figures_folder_name}/seed_{seed}/"
                if not os.path.exists(seed_folder):
                    os.makedirs(seed_folder)

                # Split test data for each seed
                val_df, test_df_split = train_test_split(
                    test_df_, test_size=0.5, random_state=seed
                )
                val_df = val_df.reset_index(drop=True)
                test_df_split = test_df_split.reset_index(drop=True)

                # Extract features and labels
                training_features = df_training[feature_names].values
                training_labels = df_training[label_name].values
                test_features = test_df_split[feature_names].values
                test_labels = test_df_split[label_name].values
                val_features = val_df[feature_names].values
                val_labels = val_df[label_name].values

                utility_fct = knn_utility_fct

                # Compute WaKA values for each tau value
                for tau in tau_values:
                    waka_values = load_or_compute_and_save(
                        f"Test Average Waka Values tau={tau:.1f}",
                        f"{seed_folder}/test_average_waka_values_{hp_str}_tau_{tau:.1f}.pkl",
                        lambda: compute_unormalized_average_waka_values(
                            training_features,
                            training_labels,
                            val_features,
                            val_labels,
                            config["K"],
                            approx="quantile",
                            strat="strat-removal-with-penalty",
                            Tau=tau,
                            ray_parallelization={"n_tasks": global_n_tasks},
                        ),
                        overwrite=overwrite_values or waka_overwrite,
                    )
                    waka_values = np.array(waka_values)
                    waka_values = np.sum(waka_values, axis=0)
                    waka_indices = np.argsort(waka_values)[::-1]

                    # Compute results for each tau value
                    waka_results = load_or_compute_and_save(
                        f"Test WaKA Acquisition tau={tau:.1f}",
                        f"{seed_folder}/test_waka_acquisition_indices_{hp_str}_tau_{tau:.1f}.pkl",
                        lambda: data_values_analysis.percentage_data_acquisition_by_indices(
                            utility_percentages,
                            waka_indices,
                            training_features,
                            training_labels,
                            test_features,
                            test_labels,
                            seed,
                            utility_fct,
                            config,
                        ),
                        overwrite=overwrite_utility_minimization or waka_overwrite,
                    )
                    seed_results[f"test_waka_tau_{tau:.1f}"].append(waka_results)

                    # Compute label ratios
                    label_ratios = load_or_compute_and_save(
                        f"WaKA Label Ratios tau={tau:.1f}",
                        f"{seed_folder}/label_ratios_waka_{hp_str}_tau_{tau:.1f}.pkl",
                        lambda: label_ratio_by_indices(
                            utility_percentages,
                            waka_indices,
                            training_labels,
                            is_removal=False,
                        ),
                        overwrite=overwrite_utility_minimization,
                    )
                    seed_results["label_ratios"][f"waka_{tau:.1f}"].append(label_ratios)

                # Random acquisition baseline
                random_means, random_stds, _ = load_or_compute_and_save(
                    "Random Data Acquisition",
                    f"{seed_folder}/random_data_acquisition_multiple_{hp_str}.pkl",
                    lambda: random_data_minimization_multiple_Ks(
                        hyperparams_configs=[config],
                        utility_fct_names=metrics,
                        training_features=training_features,
                        training_labels=training_labels,
                        prototypes_features=test_features,
                        prototypes_labels=test_labels,
                        percentages=utility_percentages,
                        num_random_partitions=num_random_partitions_per_percent,
                        global_seed=seed,
                        only_training_indices=False,
                        ray_parallelization={
                            "n_tasks": global_n_tasks,
                            "use_object_store": True,
                        },
                        utility_fct=utility_fct,
                        acquisition_mode=True,
                    ),
                    overwrite=False,
                )
                seed_random_means.append(random_means[hp_str])
                seed_random_stds.append(random_stds[hp_str])

                # Plot results for this seed
                results_dict = {
                    f"WaKA-{tau:.1f}": seed_results[f"test_waka_tau_{tau:.1f}"][-1]
                    for tau in tau_values
                }

                data_values_analysis.plot_value_based_minimization(
                    results_dict,
                    metrics,
                    utility_percentages,
                    random_means[hp_str],
                    random_stds[hp_str],
                    dataset_folder_name=seed_folder,
                    show_plot=False,
                    complete_plot_name=f"Data Acquisition {hp_str} Tau Experiment Seed {seed}",
                    acquisition_mode=True,
                    marker_size=None,
                    result_colors=result_colors,
                    small_plot=True,
                    show_grid=True,
                    direct_labels=True,
                )

            # Aggregate results across seeds
            aggregated_results_by_metric = {metric: {} for metric in metrics}

            for tau in tau_values:
                method = f"WaKA-{tau:.1f}"
                key = f"test_waka_tau_{tau:.1f}"
                for metric in metrics:
                    metric_values = [
                        [results[metric] for results in seed_results[key][i]]
                        for i in range(len(seed_results[key]))
                    ]
                    mean_values = np.mean(metric_values, axis=0)
                    std_values = np.std(metric_values, axis=0) / np.sqrt(len(seeds))
                    aggregated_results_by_metric[metric][method] = (
                        mean_values,
                        std_values,
                    )

            aggregated_results = {
                f"WaKA-{tau:.1f}": {
                    metric: aggregated_results_by_metric[metric][f"WaKA-{tau:.1f}"][0]
                    for metric in metrics
                }
                for tau in tau_values
            }

            # Aggregate random results
            random_means = {
                metric: np.mean([r[metric] for r in seed_random_means], axis=0)
                for metric in metrics
            }
            random_stds = {
                metric: np.mean([r[metric] for r in seed_random_stds], axis=0)
                for metric in metrics
            }

            # Aggregate label ratios
            label_ratios_dict = {
                f"WaKA-{tau:.1f}": (
                    np.mean(seed_results["label_ratios"][f"waka_{tau:.1f}"], axis=0),
                    np.std(seed_results["label_ratios"][f"waka_{tau:.1f}"], axis=0)
                    / np.sqrt(len(seeds)),
                )
                for tau in tau_values
            }

            # Plot aggregated results
            data_values_analysis.plot_value_based_minimization(
                aggregated_results,
                metrics,
                utility_percentages,
                random_means,
                random_stds,
                dataset_folder_name=figures_folder_name,
                show_plot=False,
                complete_plot_name=f"Data Acquisition {hp_str} Tau Experiment",
                acquisition_mode=True,
                marker_size=None,
                transpose=True,
                result_colors=result_colors,
                small_plot=True,
                show_grid=True,
                direct_labels=False,
            )

            if return_results:
                results[dataset_name][hp_str] = {
                    "utility": aggregated_results,
                    "random_means": random_means,
                    "random_stds": random_stds,
                    "label_ratios": label_ratios_dict,
                }

    if return_results:
        return results


def run_only_utility_data_acquisition(
    dataset_names,
    hyperparams_configs,
    overwrite_values=True,
    return_results=False,
    utility_percentages=np.arange(0.1, 20, 0.1),
    seeds=[42],
    overwrite_utility_minimization=False,
    add_folder_timestamp=False,
    waka_strat=None,
    waka_overwrite=False,
    global_n_tasks=4,
    Tau=None,
):
    """
    Run data acquisition experiments using utility-based value functions.
    Evaluates different methods (DSV, WaKA, LOO) for data acquisition and their impact on model performance.

    This function performs data acquisition experiments across multiple datasets, hyperparameter configurations,
    and random seeds. It evaluates the performance of different value-based methods (DSV, WaKA, LOO) for
    selecting data points to acquire, measuring their impact on various performance metrics.

    Parameters:
    -----------
    dataset_names : list
        List of dataset names to run experiments on.
    hyperparams_configs : list
        List of hyperparameter configurations to test.
    overwrite_values : bool, optional
        Whether to overwrite existing value computations. Default is True.
    return_results : bool, optional
        Whether to return the results. Default is False.
    utility_percentages : array-like, optional
        Percentages of data to acquire. Default is np.arange(0.1, 20, 0.1).
    seeds : list, optional
        List of random seeds for reproducibility. Default is [42].
    overwrite_utility_minimization : bool, optional
        Whether to overwrite existing minimization results. Default is False.
    add_folder_timestamp : bool, optional
        Whether to add timestamp to output folder names. Default is False.
    waka_strat : str, optional
        Strategy to use for WaKA computation. Default is None.
    waka_overwrite : bool, optional
        Whether to overwrite existing WaKA results. Default is False.
    global_n_tasks : int, optional
        Number of parallel tasks for computation. Default is 4.
    Tau : float, optional
        Tau parameter for WaKA computation. Default is None.

    Returns:
    --------
    dict, optional
        If return_results is True, returns a dictionary containing:
        - utility: Aggregated utility results for each method and metric
        - random_means: Mean random baseline results
        - random_stds: Standard deviation of random baseline results

    Notes:
    ------
    - The function evaluates three main methods: DSV (Data Shapley Value), WaKA, and LOO (Leave-One-Out)
    - Results are saved as plots in dataset-specific folders
    - Each experiment is run multiple times with different random seeds for robustness
    - Performance is measured using multiple metrics: MCC, accuracy, macro F1, minority F1, and KNN loss
    - Random baseline results are computed for comparison
    """
    # Plot results
    result_colors = {"DSV": "green", "WaKA": "blue", "LOO": "orange"}

    num_random_partitions_per_percent = 5
    metrics = ["mcc", "accuracy", "macro_f1", "minority_f1", "knn_loss"]

    if return_results:
        results = {}

    for dataset_name in dataset_names:
        print(f"Running for {dataset_name}")

        if return_results:
            results[dataset_name] = {}

        figures_folder_name = f"{dataset_name}/data_minimization/utility"
        if add_folder_timestamp:
            figures_folder_name = f'{figures_folder_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}/'
        else:
            figures_folder_name = f"{figures_folder_name}/"

        if not os.path.exists(figures_folder_name):
            os.makedirs(figures_folder_name)

        # Load initial data
        df_training, test_df_, feature_names, label_name = data_loader.load(
            dataset_name
        )

        for config in hyperparams_configs:
            hp_str = str(config)
            print(f"Running for {hp_str}")

            if return_results:
                results[dataset_name][hp_str] = {}

            # Store results for each seed
            seed_results = {
                "test_shapley_descend": [],
                "test_waka_descend": [],
                "test_loo_descend": [],
                "self_shapley_descend": [],
                "self_waka_descend": [],
                "self_loo_descend": [],
            }

            seed_random_means = []
            seed_random_stds = []

            for seed in seeds:
                seed_folder = f"{figures_folder_name}/seed_{seed}/"
                if not os.path.exists(seed_folder):
                    os.makedirs(seed_folder)

                # Split test data for each seed
                val_df, test_df_split = train_test_split(
                    test_df_, test_size=0.5, random_state=seed
                )
                val_df = val_df.reset_index(drop=True)
                test_df_split = test_df_split.reset_index(drop=True)

                # Extract features and labels
                training_features = df_training[feature_names].values
                training_labels = df_training[label_name].values
                test_features = test_df_split[feature_names].values
                test_labels = test_df_split[label_name].values
                val_features = val_df[feature_names].values
                val_labels = val_df[label_name].values

                utility_fct = knn_utility_fct

                # Get value scores
                (
                    test_shapley_values,
                    test_waka_values,
                    test_loo_values,
                    self_shapley_values,
                    self_waka_values,
                    self_loo_values,
                ) = values_for_dataset_and_param(
                    training_features,
                    training_labels,
                    val_features,
                    val_labels,
                    test_features,
                    test_labels,
                    utility_fct,
                    config,
                    hp_str,
                    seed_folder,
                    overwrite_utility_minimization,
                    overwrite_values,
                    waka_strat=waka_strat,
                    waka_overwrite=waka_overwrite,
                    global_n_tasks=global_n_tasks,
                    Tau=Tau,
                )

                # Get indices for acquisition (descending order)
                test_shapley_descend_value_indices = np.argsort(test_shapley_values)[
                    ::-1
                ]
                test_waka_value_descend_indices = np.argsort(test_waka_values)[::-1]
                test_loo_values_descend_indices = np.argsort(test_loo_values)[::-1]
                self_shapley_value_descend_indices = np.argsort(self_shapley_values)[
                    ::-1
                ]
                self_waka_value_descend_indices = np.argsort(self_waka_values)[::-1]
                self_loo_values_descend_indices = np.argsort(self_loo_values)[::-1]

                # Compute acquisition results for each method
                test_shapley_indices_results = load_or_compute_and_save(
                    "Test Shapley Acquisition",
                    f"{seed_folder}/test_shapley_acquisition_indices_{hp_str}.pkl",
                    lambda: data_values_analysis.percentage_data_acquisition_by_indices(
                        utility_percentages,
                        test_shapley_descend_value_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        config,
                    ),
                    overwrite=overwrite_utility_minimization,
                )
                seed_results["test_shapley_descend"].append(
                    test_shapley_indices_results
                )

                test_waka_indices_results = load_or_compute_and_save(
                    "Test WaKA Acquisition",
                    f"{seed_folder}/test_waka_acquisition_indices_{hp_str}_tau_{Tau}.pkl",
                    lambda: data_values_analysis.percentage_data_acquisition_by_indices(
                        utility_percentages,
                        test_waka_value_descend_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        config,
                    ),
                    overwrite=overwrite_utility_minimization or waka_overwrite,
                )
                seed_results["test_waka_descend"].append(test_waka_indices_results)

                test_loo_indices_results = load_or_compute_and_save(
                    "Test LOO Acquisition",
                    f"{seed_folder}/test_loo_acquisition_indices_{hp_str}.pkl",
                    lambda: data_values_analysis.percentage_data_acquisition_by_indices(
                        utility_percentages,
                        test_loo_values_descend_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        config,
                    ),
                    overwrite=overwrite_utility_minimization,
                )
                seed_results["test_loo_descend"].append(test_loo_indices_results)

                self_shapley_indices_results = load_or_compute_and_save(
                    "Self Shapley Acquisition",
                    f"{seed_folder}/self_shapley_acquisition_indices_{hp_str}.pkl",
                    lambda: data_values_analysis.percentage_data_acquisition_by_indices(
                        utility_percentages,
                        self_shapley_value_descend_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        config,
                    ),
                    overwrite=False,  # overwrite_utility_minimization
                )
                seed_results["self_shapley_descend"].append(
                    self_shapley_indices_results
                )

                self_waka_indices_results = load_or_compute_and_save(
                    "Self WaKA Acquisition",
                    f"{seed_folder}/self_waka_acquisition_indices_{hp_str}.pkl",
                    lambda: data_values_analysis.percentage_data_acquisition_by_indices(
                        utility_percentages,
                        self_waka_value_descend_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        config,
                    ),
                    overwrite=False,  # overwrite_utility_minimization
                )
                seed_results["self_waka_descend"].append(self_waka_indices_results)

                self_loo_indices_results = load_or_compute_and_save(
                    "Self LOO Acquisition",
                    f"{seed_folder}/self_loo_acquisition_indices_{hp_str}.pkl",
                    lambda: data_values_analysis.percentage_data_acquisition_by_indices(
                        utility_percentages,
                        self_loo_values_descend_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        config,
                    ),
                    overwrite=False,  # overwrite_utility_minimization
                )
                seed_results["self_loo_descend"].append(self_loo_indices_results)

                # Random acquisition baseline
                random_means, random_stds, _ = load_or_compute_and_save(
                    "Random Data Acquisition",
                    f"{seed_folder}/random_data_acquisition_multiple_{hp_str}.pkl",
                    lambda: random_data_minimization_multiple_Ks(
                        hyperparams_configs=[config],
                        utility_fct_names=[
                            "knn_loss",
                            "mcc",
                            "macro_f1",
                            "accuracy",
                            "minority_f1",
                        ],
                        training_features=training_features,
                        training_labels=training_labels,
                        prototypes_features=test_features,
                        prototypes_labels=test_labels,
                        percentages=utility_percentages,
                        num_random_partitions=num_random_partitions_per_percent,
                        global_seed=seed,
                        only_training_indices=False,
                        ray_parallelization={
                            "n_tasks": global_n_tasks,
                            "use_object_store": True,
                        },
                        utility_fct=utility_fct,
                        acquisition_mode=True,
                    ),
                    overwrite=False,  # overwrite_utility_minimization
                )
                seed_random_means.append(random_means[hp_str])
                seed_random_stds.append(random_stds[hp_str])

                # Plot individual seed results
                seed_results_dict = {
                    "DSV": test_shapley_indices_results,
                    "WaKA": test_waka_indices_results,
                    "LOO": test_loo_indices_results,
                }

                data_values_analysis.plot_value_based_minimization(
                    seed_results_dict,
                    metrics,
                    utility_percentages,
                    random_means[hp_str],
                    random_stds[hp_str],
                    dataset_folder_name=seed_folder,
                    show_plot=False,
                    complete_plot_name=f"Data Acquisition {hp_str} ({waka_strat} Tau={Tau}) Seed {seed}",
                    acquisition_mode=True,
                    marker_size=None,
                    result_colors=result_colors,
                    small_plot=True,
                    show_grid=True,
                )

            # Aggregate results across seeds

            aggregated_results_by_metric = {metric: {} for metric in metrics}

            # Aggregate each method's results per metric
            for method, key in [
                ("DSV", "test_shapley_descend"),
                ("WaKA", "test_waka_descend"),
                ("LOO", "test_loo_descend"),
            ]:
                for metric in metrics:
                    metric_values = [
                        [results[metric] for results in seed_results[key][i]]
                        for i in range(len(seed_results[key]))
                    ]
                    mean_values = np.mean(metric_values, axis=0)
                    std_values = np.std(metric_values, axis=0) / np.sqrt(len(seeds))
                    aggregated_results_by_metric[metric][method] = (
                        mean_values,
                        std_values,
                    )

            # Rebuild aggregated_results for plotting
            aggregated_results = {
                method: {
                    metric: aggregated_results_by_metric[metric][method][0]
                    for metric in metrics
                }
                for method in ["DSV", "WaKA", "LOO"]
            }

            # Aggregate random results
            random_means = {
                metric: np.mean([r[metric] for r in seed_random_means], axis=0)
                for metric in metrics
            }
            random_stds = {
                metric: np.mean([r[metric] for r in seed_random_stds], axis=0)
                for metric in metrics
            }

            data_values_analysis.plot_value_based_minimization(
                aggregated_results,
                metrics,
                utility_percentages,
                random_means,
                random_stds,
                dataset_folder_name=figures_folder_name,
                show_plot=False,
                complete_plot_name=f"Data Acquisition {hp_str} ({waka_strat} Tau={Tau})",
                acquisition_mode=True,
                marker_size=None,
                transpose=True,
                result_colors=result_colors,
                small_plot=True,
                show_grid=True,
            )

            if return_results:
                results[dataset_name][hp_str] = {
                    "utility": aggregated_results,
                    "random_means": random_means,
                    "random_stds": random_stds,
                }

    if return_results:
        return results






def run_only_utility_data_removal_tau_exp(
    dataset_names,
    hyperparams_configs,
    overwrite_values=True,
    return_results=False,
    utility_percentages=np.arange(100, 80, -0.1),
    seeds=[42],
    overwrite_utility_minimization=False,
    waka_overwrite=False,
    global_n_tasks=4,
):
    """
    Run data removal experiments using utility-based value functions for tau experiment.
    Evaluates the impact of different tau values on data removal performance.

    This function performs data removal experiments specifically designed to evaluate the impact
    of different tau values on the WaKA method's performance. It tests multiple tau values derived
    from the K parameter (tau = a/K for a in range(K+1)) to understand how this parameter affects
    data removal effectiveness.

    Parameters:
    -----------
    dataset_names : list
        List of dataset names to run experiments on.
    hyperparams_configs : list
        List of hyperparameter configurations to test.
    overwrite_values : bool, optional
        Whether to overwrite existing value computations. Default is True.
    return_results : bool, optional
        Whether to return the results. Default is False.
    utility_percentages : array-like, optional
        Percentages of data to remove. Default is np.arange(100, 80, -0.1).
    seeds : list, optional
        List of random seeds for reproducibility. Default is [42].
    overwrite_utility_minimization : bool, optional
        Whether to overwrite existing minimization results. Default is False.
    waka_overwrite : bool, optional
        Whether to overwrite existing WaKA results. Default is False.
    global_n_tasks : int, optional
        Number of parallel tasks for computation. Default is 4.

    Returns:
    --------
    dict, optional
        If return_results is True, returns a dictionary containing:
        - utility: Aggregated utility results for each method and metric
        - random_means: Mean random baseline results
        - random_stds: Standard deviation of random baseline results
        - label_ratios: Label ratio evolution for each method

    Notes:
    ------
    - The function focuses on evaluating the WaKA method with different tau values
    - Tau values are computed as a/K where a ranges from 0 to K
    - Results are visualized with different opacities for each tau value
    - Performance is measured using multiple metrics: MCC, accuracy, macro F1, minority F1, and KNN loss
    - Random baseline results are computed for comparison
    - Results are saved as plots in dataset-specific folders with seed-specific subfolders
    """
    global_n_tasks = 4  # 40#number_of_games
    num_random_partitions_per_percent = 5  # 50
    metrics = ["mcc", "accuracy", "macro_f1", "minority_f1", "knn_loss"]

    if return_results:
        results = {}

    for dataset_name in dataset_names:
        print(f"Running for {dataset_name}")

        if return_results:
            results[dataset_name] = {}

        current_datetime = datetime.datetime.now()
        date_string = current_datetime.strftime("%Y-%m-%d %H:%M")

        figures_folder_name = f"{dataset_name}/data_minimization/utility"

        if add_folder_timestamp:
            figures_folder_name = f"{figures_folder_name}_{date_string}/"
        else:
            figures_folder_name = f"{figures_folder_name}/"

        if not os.path.exists(figures_folder_name):
            os.makedirs(figures_folder_name)

        df_training, test_df_, feature_names, label_name = data_loader.load(
            dataset_name
        )

        for config in hyperparams_configs:
            K = config["K"]
            tau_values = [a / K for a in range(K + 1)]
            hp_str = str(config)
            print(f"Running for {hp_str}")

            # Generate colors with different opacities
            result_colors = {
                f"WaKA-{tau:.1f}": (
                    0,
                    0,
                    0.5,
                    (i + 1) / len(tau_values),
                )  # Dark blue with increasing opacity
                for i, tau in enumerate(tau_values)
            }

            if return_results:
                results[dataset_name][hp_str] = {}

            # Store results for each seed
            seed_results = {f"test_waka_tau_{tau:.1f}": [] for tau in tau_values}
            seed_results["label_ratios"] = {f"waka_{tau:.1f}": [] for tau in tau_values}

            seed_random_means = []
            seed_random_stds = []

            for seed in seeds:
                seed_folder = f"{figures_folder_name}/seed_{seed}/"
                if not os.path.exists(seed_folder):
                    os.makedirs(seed_folder)

                val_df, test_df = train_test_split(
                    test_df_, test_size=0.5, random_state=seed
                )
                val_df = val_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)

                training_features = df_training[feature_names].values
                training_labels = df_training[label_name].values
                test_features = test_df[feature_names].values
                test_labels = test_df[label_name].values
                val_features = val_df[feature_names].values
                val_labels = val_df[label_name].values

                utility_fct = knn_utility_fct

                # Compute WaKA values for each tau value
                for tau in tau_values:
                    waka_values = load_or_compute_and_save(
                        f"Test Average Waka Values tau={tau:.1f}",
                        f"{seed_folder}/test_average_waka_values_{hp_str}_tau_{tau:.1f}.pkl",
                        lambda: compute_unormalized_average_waka_values(
                            training_features,
                            training_labels,
                            val_features,
                            val_labels,
                            config["K"],
                            approx="quantile",
                            strat="strat-removal-with-penalty",
                            Tau=tau,
                            ray_parallelization={"n_tasks": global_n_tasks},
                        ),
                        overwrite=overwrite_values or waka_overwrite,
                    )
                    waka_values = np.array(waka_values)
                    waka_values = np.sum(waka_values, axis=0)
                    waka_indices = np.argsort(waka_values)[::-1]

                    # Compute results for this tau value
                    waka_results = load_or_compute_and_save(
                        f"Test WaKA Descend Indices tau={tau:.1f}",
                        f"{seed_folder}/test_waka_descend_indices_{hp_str}_tau_{tau:.1f}.pkl",
                        lambda: data_values_analysis.percentage_data_minimization_by_indices(
                            utility_percentages,
                            waka_indices,
                            training_features,
                            training_labels,
                            test_features,
                            test_labels,
                            seed,
                            utility_fct,
                            only_training_indices=False,
                            hyperparams=config,
                            return_indices=True,
                        ),
                        overwrite=overwrite_utility_minimization or waka_overwrite,
                    )
                    seed_results[f"test_waka_tau_{tau:.1f}"].append(
                        [s for s, _ in waka_results]
                    )

                    # Compute label ratios
                    label_ratios = load_or_compute_and_save(
                        f"WaKA Label Ratios tau={tau:.1f}",
                        f"{seed_folder}/label_ratios_waka_{hp_str}_tau_{tau:.1f}.pkl",
                        lambda: label_ratio_by_indices(
                            utility_percentages,
                            waka_indices,
                            training_labels,
                            is_removal=True,
                        ),
                        overwrite=True,  # overwrite_utility_minimization
                    )
                    seed_results["label_ratios"][f"waka_{tau:.1f}"].append(label_ratios)

                # Random minimization
                random_means, random_stds, _ = load_or_compute_and_save(
                    "Random Data Minimization",
                    f"{seed_folder}/random_data_minimization_multiple_{hp_str}.pkl",
                    lambda: random_data_minimization_multiple_Ks(
                        hyperparams_configs=[config],
                        utility_fct_names=metrics,
                        training_features=training_features,
                        training_labels=training_labels,
                        prototypes_features=test_features,
                        prototypes_labels=test_labels,
                        percentages=utility_percentages,
                        num_random_partitions=num_random_partitions_per_percent,
                        global_seed=seed,
                        only_training_indices=False,
                        ray_parallelization={
                            "n_tasks": global_n_tasks,
                            "use_object_store": True,
                        },
                        utility_fct=utility_fct,
                    ),
                    overwrite=False,
                )
                seed_random_means.append(random_means[hp_str])
                seed_random_stds.append(random_stds[hp_str])

                # Plot results for this seed
                waka_results_dict = {
                    f"WaKA-{tau:.1f}": seed_results[f"test_waka_tau_{tau:.1f}"][-1]
                    for tau in tau_values
                }
                data_values_analysis.plot_value_based_minimization(
                    waka_results_dict,
                    metrics,
                    utility_percentages,
                    random_means[hp_str],
                    random_stds[hp_str],
                    dataset_folder_name=seed_folder,
                    show_plot=False,
                    complete_plot_name=f"Data Removal {hp_str} Tau Experiment Seed {seed}",
                    marker_size=None,
                    transpose=False,
                    result_colors=result_colors,
                    small_plot=True,
                    show_grid=True,
                    direct_labels=True,
                )

                # Plot label ratio evolution for this seed
                label_ratios_dict = {
                    f"WaKA-{tau:.1f}": seed_results["label_ratios"][f"waka_{tau:.1f}"][
                        -1
                    ]
                    for tau in tau_values
                }
                plot_label_ratio_evolution(
                    utility_percentages,
                    label_ratios_dict,
                    label_ratio_by_indices(
                        [100],
                        np.arange(len(training_labels)),
                        training_labels,
                        is_removal=True,
                    )[0],
                    f"{seed_folder}/label_ratio_evolution_{hp_str}_tau_exp.png",
                    title=f"Label Ratio Evolution for KNN {hp_str} Seed {seed}",
                    result_colors=result_colors,
                )

            # Aggregate results across seeds
            aggregated_results_by_metric = {metric: {} for metric in metrics}

            for tau in tau_values:
                method = f"WaKA-{tau:.1f}"
                key = f"test_waka_tau_{tau:.1f}"
                for metric in metrics:
                    metric_values = [
                        [results[metric] for results in seed_results[key][i]]
                        for i in range(len(seed_results[key]))
                    ]
                    mean_values = np.mean(metric_values, axis=0)
                    std_values = np.std(metric_values, axis=0) / np.sqrt(len(seeds))
                    aggregated_results_by_metric[metric][method] = (
                        mean_values,
                        std_values,
                    )

            aggregated_results = {
                f"WaKA-{tau:.1f}": {
                    metric: aggregated_results_by_metric[metric][f"WaKA-{tau:.1f}"][0]
                    for metric in metrics
                }
                for tau in tau_values
            }

            # Aggregate random results
            random_means = {
                metric: np.mean([r[metric] for r in seed_random_means], axis=0)
                for metric in metrics
            }
            random_stds = {
                metric: np.mean([r[metric] for r in seed_random_stds], axis=0)
                for metric in metrics
            }

            # Aggregate label ratios
            label_ratios_dict = {
                f"WaKA-{tau:.1f}": (
                    np.mean(seed_results["label_ratios"][f"waka_{tau:.1f}"], axis=0),
                    np.std(seed_results["label_ratios"][f"waka_{tau:.1f}"], axis=0)
                    / np.sqrt(len(seeds)),
                )
                for tau in tau_values
            }

            initial_label_ratio = np.mean(
                [
                    label_ratio_by_indices(
                        [100],
                        np.arange(len(training_labels)),
                        training_labels,
                        is_removal=True,
                    )[0]
                    for _ in seeds
                ]
            )

            # Plot aggregated results
            data_values_analysis.plot_value_based_minimization(
                aggregated_results,
                metrics,
                utility_percentages,
                random_means,
                random_stds,
                dataset_folder_name=figures_folder_name,
                show_plot=False,  # TODO : False
                complete_plot_name=f"Data Removal {hp_str} Tau Experiment",
                marker_size=None,
                transpose=True,
                result_colors=result_colors,
                small_plot=True,
                show_grid=True,
                direct_labels=False,
            )

            plot_label_ratio_evolution(
                utility_percentages,
                {k: v[0] for k, v in label_ratios_dict.items()},
                initial_label_ratio,
                f"{figures_folder_name}/label_ratio_evolution_{hp_str}_tau_exp.png",
                title=f"Label Ratio Evolution for KNN {hp_str}",
                result_colors=result_colors,
            )

            if return_results:
                results[dataset_name][hp_str] = {
                    "utility": aggregated_results,
                    "random_means": random_means,
                    "random_stds": random_stds,
                    "label_ratios": label_ratios_dict,
                }

    if return_results:
        return results


def run_only_utility_data_removal(
    dataset_names,
    hyperparams_configs,
    overwrite_values=True,
    return_results=False,
    utility_percentages=np.arange(100, 80, -0.1),
    seeds=[42],
    overwrite_utility_minimization=False,
    waka_strat=None,
    waka_overwrite=False,
    global_n_tasks=4,
    Tau=None,
):
    """
    Run data removal experiments using utility-based value functions.
    Evaluates different methods (DSV, WaKA, LOO) for data removal and their impact on model performance.

    This function performs data removal experiments across multiple datasets, hyperparameter configurations,
    and random seeds. It evaluates the performance of different value-based methods (DSV, WaKA, LOO) for
    selecting data points to remove, measuring their impact on various performance metrics.

    Parameters:
    -----------
    dataset_names : list
        List of dataset names to run experiments on.
    hyperparams_configs : list
        List of hyperparameter configurations to test.
    overwrite_values : bool, optional
        Whether to overwrite existing value computations. Default is True.
    return_results : bool, optional
        Whether to return the results. Default is False.
    utility_percentages : array-like, optional
        Percentages of data to remove. Default is np.arange(100, 80, -0.1).
    seeds : list, optional
        List of random seeds for reproducibility. Default is [42].
    overwrite_utility_minimization : bool, optional
        Whether to overwrite existing minimization results. Default is False.
    waka_strat : str, optional
        Strategy to use for WaKA computation. Default is None.
    waka_overwrite : bool, optional
        Whether to overwrite existing WaKA results. Default is False.
    global_n_tasks : int, optional
        Number of parallel tasks for computation. Default is 4.
    Tau : float, optional
        Tau parameter for WaKA computation. Default is None.

    Returns:
    --------
    dict, optional
        If return_results is True, returns a dictionary containing:
        - utility: Aggregated utility results for each method and metric
        - random_means: Mean random baseline results
        - random_stds: Standard deviation of random baseline results
        - label_ratios: Label ratio evolution for each method

    Notes:
    ------
    - The function evaluates three main methods: DSV (Data Shapley Value), WaKA, and LOO (Leave-One-Out)
    - Results are saved as plots in dataset-specific folders
    - Each experiment is run multiple times with different random seeds for robustness
    - Performance is measured using multiple metrics: MCC, accuracy, macro F1, minority F1, and KNN loss
    - Random baseline results are computed for comparison
    - The function supports different WaKA strategies and tau values for fine-tuning the removal process
    """
    global_n_tasks = 4  # 40#number_of_games
    num_random_partitions_per_percent = 5  # 50
    metrics = ["mcc", "accuracy", "macro_f1", "minority_f1", "knn_loss"]

    result_colors = {"DSV": "green", "WaKA": "blue", "LOO": "orange"}

    if return_results:
        results = {}

    for dataset_name in dataset_names:
        print(f"Running for {dataset_name}")

        if return_results:
            results[dataset_name] = {}

        current_datetime = datetime.datetime.now()
        date_string = current_datetime.strftime("%Y-%m-%d %H:%M")

        figures_folder_name = f"{dataset_name}/data_minimization/utility"

        if add_folder_timestamp:
            figures_folder_name = f"{figures_folder_name}_{date_string}/"
        else:
            figures_folder_name = f"{figures_folder_name}/"

        if not os.path.exists(figures_folder_name):
            os.makedirs(figures_folder_name)

        df_training, test_df_, feature_names, label_name = data_loader.load(
            dataset_name
        )

        for config in hyperparams_configs:
            hp_str = str(config)
            print(f"Running for {hp_str}")

            if return_results:
                results[dataset_name][hp_str] = {}

            # Store results for each seed
            seed_results = {
                "test_shapley_descend": [],
                "test_waka_descend": [],
                "test_loo_descend": [],
                "self_shapley_ascend": [],
                "self_waka_ascend": [],
                "self_loo_ascend": [],
                "label_ratios": {"dsv": [], "waka": [], "loo": []},
            }

            seed_random_means = []
            seed_random_stds = []

            for seed in seeds:
                seed_folder = f"{figures_folder_name}/seed_{seed}/"
                if not os.path.exists(seed_folder):
                    os.makedirs(seed_folder)

                val_df, test_df = train_test_split(
                    test_df_, test_size=0.5, random_state=seed
                )
                val_df = val_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)

                training_features = df_training[feature_names].values
                training_labels = df_training[label_name].values
                test_features = test_df[feature_names].values
                test_labels = test_df[label_name].values
                val_features = val_df[feature_names].values
                val_labels = val_df[label_name].values

                utility_fct = knn_utility_fct

                (
                    test_shapley_values,
                    test_waka_values,
                    test_loo_values,
                    self_shapley_values,
                    self_waka_values,
                    self_loo_values,
                ) = values_for_dataset_and_param(
                    training_features,
                    training_labels,
                    val_features,
                    val_labels,
                    test_features,
                    test_labels,
                    utility_fct,
                    config,
                    hp_str,
                    seed_folder,
                    overwrite_utility_minimization,
                    overwrite_values,
                    waka_strat=waka_strat,
                    Tau=Tau,
                    waka_overwrite=waka_overwrite,
                    global_n_tasks=global_n_tasks,
                )

                # Get indices in descending/ascending order
                test_shapley_descend_value_indices = np.argsort(test_shapley_values)[
                    ::-1
                ]
                test_waka_value_descend_indices = np.argsort(test_waka_values)[::-1]
                test_loo_values_descend_indices = np.argsort(test_loo_values)[::-1]
                self_shapley_value_ascend_indices = np.argsort(self_shapley_values)
                self_waka_value_ascend_indices = np.argsort(self_waka_values)
                self_loo_values_ascend_indices = np.argsort(self_loo_values)

                # Compute minimization results for each method
                test_shapley_descend_indices = load_or_compute_and_save(
                    "Test Shapley Descend Indices",
                    f"{seed_folder}/test_shapley_descend_indices_{hp_str}.pkl",
                    lambda: data_values_analysis.percentage_data_minimization_by_indices(
                        utility_percentages,
                        test_shapley_descend_value_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        only_training_indices=False,
                        hyperparams=config,
                        return_indices=True,
                    ),
                    overwrite=overwrite_utility_minimization,
                )
                seed_results["test_shapley_descend"].append(
                    [s for s, _ in test_shapley_descend_indices]
                )

                test_waka_descend_indices = load_or_compute_and_save(
                    "Test WaKA Descend Indices",
                    f"{seed_folder}/test_waka_descend_indices_{hp_str}_tau_{tau}.pkl",
                    lambda: data_values_analysis.percentage_data_minimization_by_indices(
                        utility_percentages,
                        test_waka_value_descend_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        only_training_indices=False,
                        hyperparams=config,
                        return_indices=True,
                    ),
                    overwrite=overwrite_utility_minimization or waka_overwrite,
                )
                seed_results["test_waka_descend"].append(
                    [s for s, _ in test_waka_descend_indices]
                )

                test_loo_descend_indices = load_or_compute_and_save(
                    "Test LOO Descend Indices",
                    f"{seed_folder}/test_loo_descend_indices_{hp_str}.pkl",
                    lambda: data_values_analysis.percentage_data_minimization_by_indices(
                        utility_percentages,
                        test_loo_values_descend_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        only_training_indices=False,
                        hyperparams=config,
                        return_indices=True,
                    ),
                    overwrite=overwrite_utility_minimization,
                )
                seed_results["test_loo_descend"].append(
                    [s for s, _ in test_loo_descend_indices]
                )

                self_shapley_ascend_indices = load_or_compute_and_save(
                    "Self Shapley Ascend Indices",
                    f"{seed_folder}/self_shapley_ascend_indices_{hp_str}.pkl",
                    lambda: data_values_analysis.percentage_data_minimization_by_indices(
                        utility_percentages,
                        self_shapley_value_ascend_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        only_training_indices=False,
                        hyperparams=config,
                        return_indices=True,
                    ),
                    overwrite=False,  # overwrite_utility_minimization
                )
                seed_results["self_shapley_ascend"].append(
                    [s for s, _ in self_shapley_ascend_indices]
                )

                self_waka_ascend_indices = load_or_compute_and_save(
                    "Self WaKA Ascend Indices",
                    f"{seed_folder}/self_waka_ascend_indices_{hp_str}.pkl",
                    lambda: data_values_analysis.percentage_data_minimization_by_indices(
                        utility_percentages,
                        self_waka_value_ascend_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        only_training_indices=False,
                        hyperparams=config,
                        return_indices=True,
                    ),
                    overwrite=False,  # overwrite_utility_minimization
                )
                seed_results["self_waka_ascend"].append(
                    [s for s, _ in self_waka_ascend_indices]
                )

                self_loo_ascend_indices = load_or_compute_and_save(
                    "Self LOO Ascend Indices",
                    f"{seed_folder}/self_loo_ascend_indices_{hp_str}.pkl",
                    lambda: data_values_analysis.percentage_data_minimization_by_indices(
                        utility_percentages,
                        self_loo_values_ascend_indices,
                        training_features,
                        training_labels,
                        test_features,
                        test_labels,
                        seed,
                        utility_fct,
                        only_training_indices=False,
                        hyperparams=config,
                        return_indices=True,
                    ),
                    overwrite=False,  # overwrite_utility_minimization
                )
                seed_results["self_loo_ascend"].append(
                    [s for s, _ in self_loo_ascend_indices]
                )

                # Random minimization
                random_means, random_stds, _ = load_or_compute_and_save(
                    "Random Data Minimization",
                    f"{seed_folder}/random_data_minimization_multiple_{hp_str}.pkl",
                    lambda: random_data_minimization_multiple_Ks(
                        hyperparams_configs=[config],
                        utility_fct_names=[
                            "knn_loss",
                            "mcc",
                            "macro_f1",
                            "accuracy",
                            "minority_f1",
                        ],
                        training_features=training_features,
                        training_labels=training_labels,
                        prototypes_features=test_features,
                        prototypes_labels=test_labels,
                        percentages=utility_percentages,
                        num_random_partitions=num_random_partitions_per_percent,
                        global_seed=seed,
                        only_training_indices=False,
                        ray_parallelization={
                            "n_tasks": global_n_tasks,
                            "use_object_store": True,
                        },
                        utility_fct=utility_fct,
                    ),
                    overwrite=False,  # overwrite_utility_minimization
                )
                seed_random_means.append(random_means[hp_str])
                seed_random_stds.append(random_stds[hp_str])

                # Compute label ratios for each method
                label_ratios_removal = load_or_compute_and_save(
                    "DSV Label Ratios",
                    f"{seed_folder}/label_ratios_dsv_{hp_str}.pkl",
                    lambda: label_ratio_by_indices(
                        utility_percentages,
                        test_shapley_descend_value_indices,
                        training_labels,
                        is_removal=True,
                    ),
                    overwrite=overwrite_utility_minimization,
                )
                seed_results["label_ratios"]["dsv"].append(label_ratios_removal)

                label_ratios_removal_waka = load_or_compute_and_save(
                    "WaKA Label Ratios",
                    f"{seed_folder}/label_ratios_waka_{hp_str}_{waka_strat}.pkl",
                    lambda: label_ratio_by_indices(
                        utility_percentages,
                        test_waka_value_descend_indices,
                        training_labels,
                        is_removal=True,
                    ),
                    overwrite=overwrite_utility_minimization,
                )
                seed_results["label_ratios"]["waka"].append(label_ratios_removal_waka)

                label_ratios_removal_loo = load_or_compute_and_save(
                    "LOO Label Ratios",
                    f"{seed_folder}/label_ratios_loo_{hp_str}.pkl",
                    lambda: label_ratio_by_indices(
                        utility_percentages,
                        test_loo_values_descend_indices,
                        training_labels,
                        is_removal=True,
                    ),
                    overwrite=overwrite_utility_minimization,
                )
                seed_results["label_ratios"]["loo"].append(label_ratios_removal_loo)

                data_values_analysis.plot_value_based_minimization(
                    {
                        "DSV": seed_results["test_shapley_descend"][-1],
                        "WaKA": seed_results["test_waka_descend"][-1],
                        "LOO": seed_results["test_loo_descend"][-1],
                    },
                    metrics,
                    utility_percentages,
                    random_means[hp_str],
                    random_stds[hp_str],
                    dataset_folder_name=seed_folder,
                    show_plot=False,
                    complete_plot_name=f"Data Removal {hp_str} ({waka_strat} Tau={Tau}) Seed {seed}",
                    marker_size=None,
                    transpose=False,
                    result_colors=result_colors,
                    small_plot=True,
                    show_grid=True,
                )

                # Plot label ratio evolution for this seed
                plot_label_ratio_evolution(
                    utility_percentages,
                    {
                        "DSV": seed_results["label_ratios"]["dsv"][-1],
                        "WaKA": seed_results["label_ratios"]["waka"][-1],
                        "LOO": seed_results["label_ratios"]["loo"][-1],
                    },
                    label_ratio_by_indices(
                        [100],
                        np.arange(len(training_labels)),
                        training_labels,
                        is_removal=True,
                    )[0],
                    f"{seed_folder}/label_ratio_evolution_{hp_str}_tau_{tau}.png",
                    title=f"Label Ratio Evolution for KNN {hp_str} Seed {seed}",
                    result_colors=result_colors,
                )

            # Aggregate results across seeds
            # Initialize separate aggregated results per metric

            aggregated_results_by_metric = {metric: {} for metric in metrics}

            # Aggregate each method's results per metric
            for method, key in [
                ("DSV", "test_shapley_descend"),
                ("WaKA", "test_waka_descend"),
                ("LOO", "test_loo_descend"),
            ]:
                for metric in metrics:
                    metric_values = [
                        [results[metric] for results in seed_results[key][i]]
                        for i in range(len(seed_results[key]))
                    ]
                    mean_values = np.mean(metric_values, axis=0)
                    std_values = np.std(metric_values, axis=0) / np.sqrt(len(seeds))
                    aggregated_results_by_metric[metric][method] = (
                        mean_values,
                        std_values,
                    )

            # Rebuild aggregated_results for plotting
            aggregated_results = {
                method: {
                    metric: aggregated_results_by_metric[metric][method][0]
                    for metric in metrics
                }
                for method in ["DSV", "WaKA", "LOO"]
            }

            # Aggregate random results across seeds
            random_means = {
                metric: np.mean([r[metric] for r in seed_random_means], axis=0)
                for metric in metrics
            }
            random_stds = {
                metric: np.mean([r[metric] for r in seed_random_stds], axis=0)
                for metric in metrics
            }

            # Aggregate label ratios
            label_ratios_dict = {
                "DSV": (
                    np.mean(seed_results["label_ratios"]["dsv"], axis=0),
                    np.std(seed_results["label_ratios"]["dsv"], axis=0)
                    / np.sqrt(len(seeds)),
                ),
                "WaKA": (
                    np.mean(seed_results["label_ratios"]["waka"], axis=0),
                    np.std(seed_results["label_ratios"]["waka"], axis=0)
                    / np.sqrt(len(seeds)),
                ),
                "LOO": (
                    np.mean(seed_results["label_ratios"]["loo"], axis=0),
                    np.std(seed_results["label_ratios"]["loo"], axis=0)
                    / np.sqrt(len(seeds)),
                ),
            }

            # Compute initial label ratio (averaged across seeds)
            initial_label_ratios = [
                label_ratio_by_indices(
                    [100],
                    np.arange(len(training_labels)),
                    training_labels,
                    is_removal=True,
                )[0]
                for _ in seeds
            ]
            initial_label_ratio = np.mean(initial_label_ratios)

            # Plot aggregated results
            data_values_analysis.plot_value_based_minimization(
                aggregated_results,
                metrics,
                utility_percentages,
                random_means,
                random_stds,
                dataset_folder_name=figures_folder_name,
                show_plot=False,
                complete_plot_name=f"Data Removal {hp_str} ({waka_strat} Tau={Tau})",
                marker_size=None,
                transpose=True,
                result_colors=result_colors,
                small_plot=True,
                show_grid=True,
            )

            # Plot label ratio evolution with error bars
            plot_label_ratio_evolution(
                utility_percentages,
                {k: v[0] for k, v in label_ratios_dict.items()},  # means
                initial_label_ratio,
                f"{figures_folder_name}/label_ratio_evolution_{hp_str}_tau_{tau}.png",
                title=f"Label Ratio Evolution for KNN {hp_str}",
                result_colors=result_colors,
            )

            if return_results:
                results[dataset_name][hp_str] = {
                    "utility": aggregated_results,
                    "random_means": random_means,
                    "random_stds": random_stds,
                    "label_ratios": label_ratios_dict,
                }

    if return_results:
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global_n_tasks",
        type=int,
        default=30,
        help="Number of global tasks for parallelization",
    )
    parser.add_argument(
        "--experiment_name", type=str, default="tau_exp", help="Experiment name"
    )
    parser.add_argument("--dataset_name", type=str, default="imdb", help="Dataset name")
    parser.add_argument(
        "--overwrite_all", type=bool, default=False, help="Overwrite all results"
    )
    parser.add_argument(
        "--waka_default_strat", type=int, default=-1, help="Waka default strategy"
    )
    parser.add_argument("--tau", type=float, default=0.4, help="Tau value")

    args = parser.parse_args()

    number_of_points_per_game = 50
    number_of_games = 50
    global_n_tasks = 4
    num_random_partitions_per_percent = 5  # 50

    utility_percentages = np.arange(100, 38, -2)

    global_seed = 42  # Global seed for reproducibility
    hyperparams_configs = [{"K": 5}]
    global_overwrite = True
    common_prep_overwrite = True
    add_folder_timestamp = False

    progress_update_interval = 10
    N_shadow = 16

    waka_max_quantile = 900
    waka_max_contributors = 2000

    dataset_names = []

    print(f"Global n tasks: {args.global_n_tasks}")
    print(f"Experiment name: {args.experiment_name}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Overwrite all: {args.overwrite_all}")
    print(f"Waka default strategy: {args.waka_default_strat}")

    hyperparams_configs = [{"K": 5}]

    if args.overwrite_all:
        overwrite_all = True
        waka_overwrite = True
    else:
        overwrite_all = False
        waka_overwrite = False

    val_test_seeds = [42, 43, 44, 45, 46]

    experiment_name = args.experiment_name

    for tau in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        if experiment_name == "data_removal" or experiment_name == "all":

            run_only_utility_data_removal(
                dataset_names,
                hyperparams_configs,
                overwrite_values=overwrite_all,
                waka_overwrite=waka_overwrite,
                overwrite_utility_minimization=overwrite_all,
                seeds=val_test_seeds,
                waka_strat=(
                    f"strat-removal-{args.waka_default_strat}"
                    if args.waka_default_strat >= 0
                    else "strat-removal-with-penalty"
                ),
                Tau=tau,
                global_n_tasks=args.global_n_tasks,
            )

        if experiment_name == "data_acquisition" or experiment_name == "all":

            run_only_utility_data_acquisition(
                dataset_names,
                hyperparams_configs,
                overwrite_values=overwrite_all,
                waka_overwrite=waka_overwrite,
                overwrite_utility_minimization=overwrite_all,
                seeds=val_test_seeds,
                waka_strat=(
                    f"strat-acquisition-{args.waka_default_strat}"
                    if args.waka_default_strat >= 0
                    else "strat-acquisition-with-penalty"
                ),
                Tau=tau,
                global_n_tasks=args.global_n_tasks,
            )

    if experiment_name == "tau_exp":

        run_only_utility_data_removal_tau_exp(
            dataset_names,
            hyperparams_configs,
            overwrite_values=overwrite_all,
            waka_overwrite=waka_overwrite,
            overwrite_utility_minimization=overwrite_all,
            seeds=val_test_seeds,
            global_n_tasks=args.global_n_tasks,
        )

        run_only_utility_data_acquisition_tau_exp(
            dataset_names,
            hyperparams_configs,
            overwrite_values=overwrite_all,
            waka_overwrite=waka_overwrite,
            overwrite_utility_minimization=overwrite_all,
            seeds=val_test_seeds,
            global_n_tasks=args.global_n_tasks,
        )
