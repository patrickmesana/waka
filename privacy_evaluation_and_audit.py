from waka import (
    compute_unormalized_average_waka_values,
    compute_self_waka_value_with_optional_target,
    compute_self_waka_value_with_optional_target_with_knn,
    compute_self_unormalized_average_waka_values,
    compute_self_unormalized_average_waka_values_recomputable,
)
import knn_valuation
import utils
from lira_attack import *

import os
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from utils import load_or_compute_and_save
import scipy.stats as stats
import time
from scipy.stats import sem
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import data_loader

from scipy.spatial.distance import cdist


import sys
from sklearn.model_selection import StratifiedKFold


default_distance_metric = "euclidean"


def membership_inference_security_game(
    df_data,
    K,
    attacker_predict,
    number_of_iterations,
    game_seed,
    membership_ratio=0.5,
    feature_names=["x", "y"],
    label_name="label",
    batch=False,
    distribution_data=None,
    target_model_fct=None,
    loss_fct=None,
):
    """
    Implements a membership inference security game to evaluate model privacy.

    Parameters:
    -----------
    df_data : pandas.DataFrame
        The dataset containing features and labels
    K : int
        Number of nearest neighbors for KNN model
    attacker_predict : callable
        Function that implements the attacker's prediction strategy
    number_of_iterations : int
        Number of iterations to run the security game
    game_seed : int
        Random seed for reproducibility
    membership_ratio : float, default=0.5
        Ratio of data to use for training vs distribution
    feature_names : list, default=["x", "y"]
        Names of feature columns in df_data
    label_name : str, default="label"
        Name of label column in df_data
    batch : bool, default=False
        Whether to run attacks in batch mode
    distribution_data : pandas.DataFrame, optional
        Pre-defined distribution data to use
    target_model_fct : callable, optional
        Function to create target model (defaults to KNN)
    loss_fct : callable, optional
        Custom loss function for model evaluation

    Returns:
    --------
    tuple
        Results from challenge_target_model containing scores, labels, indices, and losses
    """
    if distribution_data is None:
        df_training_data, df_distribution_data = train_test_split(
            df_data, test_size=1 - membership_ratio, random_state=game_seed
        )
    else:
        df_distribution_data = distribution_data
        df_training_data = df_data.sample(
            n=len(df_distribution_data), random_state=game_seed
        )

    if target_model_fct is not None:
        target_model = target_model_fct()
        target_model.fit(
            df_training_data[feature_names].values, df_training_data[label_name].values
        )
    else:
        target_model = KNeighborsClassifier(n_neighbors=K)
        target_model.fit(
            df_training_data[feature_names].values, df_training_data[label_name].values
        )

    return challenge_target_model(
        target_model,
        df_training_data,
        df_distribution_data,
        K,
        attacker_predict,
        number_of_iterations,
        game_seed,
        feature_names=feature_names,
        label_name=label_name,
        batch=batch,
        loss_fct=loss_fct,
    )


def select_random_index(df_training_data, df_distribution_data):

    if df_distribution_data.empty:
        idx = random.choice(df_training_data.index.tolist())
        b = 1
    else:
        b = random.choice([0, 1])
        if b == 0:
            idx = random.choice(df_distribution_data.index.tolist())
        else:
            idx = random.choice(df_training_data.index.tolist())
    return idx, b


def challenge_target_model(
    target_model,
    df_training_data,
    df_distribution_data,
    K,
    attacker_predict,
    number_of_iterations,
    game_seed,
    feature_names=["x", "y"],
    label_name="label",
    batch=False,
    loss_fct=None,
):
    """
    Challenges a target model with membership inference attacks.

    Parameters:
    -----------
    target_model : sklearn.base.BaseEstimator
        The trained target model to challenge
    df_training_data : pandas.DataFrame
        Training data used to train the target model
    df_distribution_data : pandas.DataFrame
        Distribution data not used in training
    K : int
        Number of nearest neighbors for KNN-based attacks
    attacker_predict : callable
        Function implementing the attacker's prediction strategy
    number_of_iterations : int
        Number of challenge points to generate
    game_seed : int
        Random seed for reproducibility
    feature_names : list, default=["x", "y"]
        Names of feature columns
    label_name : str, default="label"
        Name of label column
    batch : bool, default=False
        Whether to run attacks in batch mode
    loss_fct : callable, optional
        Custom loss function for model evaluation

    Returns:
    --------
    tuple
        (scores, labels, indices, target_losses) where:
        - scores: Attack scores for each challenge point
        - labels: True membership labels (1 for training, 0 for distribution)
        - indices: Indices of challenge points
        - target_losses: Loss values for each challenge point
    """
    scores, labels, indices, target_losses = [], [], [], []

    random.seed(game_seed)

    for _ in range(number_of_iterations):
        idx, b = select_random_index(df_training_data, df_distribution_data)

        if b == 0:
            df_data = df_distribution_data
        else:
            df_data = df_training_data

        # Here this is loc and not iloc because 'select_random_index' returns the index from the dataframe
        challenge_point = df_data.loc[idx, feature_names].values.astype(float)
        challenge_label = df_data.loc[idx, label_name]

        if loss_fct is None:
            _, challenge_points_nn_indices = target_model.kneighbors(
                np.array([challenge_point]), n_neighbors=K
            )
            neighbor_labels = df_training_data[label_name].values[
                challenge_points_nn_indices
            ]
            target_model_loss = 1 - np.sum(neighbor_labels == challenge_label) / K
        else:
            target_model_loss = loss_fct(target_model, challenge_point, challenge_label)

        if not batch:
            attack_score = attacker_predict(
                challenge_point,
                challenge_label,
                target_model_loss,
                K,
                challenge_point_idx=idx,
            )
            scores.append(attack_score)

        labels.append(b)
        indices.append(idx)
        target_losses.append(target_model_loss)

    if batch:
        scores = attacker_predict(indices, target_losses, K)

    return scores, labels, indices, target_losses


def run_game_for_auc(seed, game_fct):
    scores, labels, _, _ = game_fct(seed)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def run_multiple_games(
    df_data,
    K,
    attacker_predict,
    number_of_games,
    seed_list,
    membership_ratio,
    feature_names=None,
    label_name=None,
    ray_parallelization=None,
    batch=False,
    distribution_data=None,
    ray_already_init=False,
    target_model_fct=None,
    loss_fct=None,
):
    """
    Runs multiple security games in parallel using Ray for distributed computing.

    Parameters:
    -----------
    df_data : pandas.DataFrame
        The dataset containing features and labels
    K : int
        Number of nearest neighbors for KNN model
    attacker_predict : callable
        Function implementing the attacker's prediction strategy
    number_of_games : int
        Number of security games to run
    seed_list : list
        List of random seeds for each game
    membership_ratio : float
        Ratio of data to use for training vs distribution
    feature_names : list, optional
        Names of feature columns
    label_name : str, optional
        Name of label column
    ray_parallelization : dict, optional
        Configuration for Ray parallelization
    batch : bool, default=False
        Whether to run attacks in batch mode
    distribution_data : pandas.DataFrame, optional
        Pre-defined distribution data
    ray_already_init : bool, default=False
        Whether Ray is already initialized
    target_model_fct : callable, optional
        Function to create target model
    loss_fct : callable, optional
        Custom loss function

    Returns:
    --------
    list
        List of results from each security game
    """

    def run_game_with_seed(seed):
        return membership_inference_security_game(
            df_data,
            K,
            attacker_predict,
            number_of_games,
            feature_names=feature_names,
            label_name=label_name,
            game_seed=seed,
            membership_ratio=membership_ratio,
            batch=batch,
            distribution_data=distribution_data,
            target_model_fct=target_model_fct,
            loss_fct=loss_fct,
        )

    if ray_parallelization is None:
        all_scores = [run_game_with_seed(seed) for seed in seed_list]
    else:
        from rayPlus import parallel_loop

        # Warning! if return_results=True, the results will be stored in memory
        # and it can be a problem if the results are too big, it will crash
        # your computer

        all_scores = parallel_loop(
            seed_list,
            run_game_with_seed,
            return_results=True,
            n_tasks=ray_parallelization["n_tasks"],
            init_and_shutdown_ray=not ray_already_init,
        )

    return all_scores


def run_multiple_games_with_target_points(
    df_data,
    K,
    attacker_predict,
    game_seed_list,
    membership_ratio,
    target_points_indices,
    feature_names=None,
    label_name=None,
    ray_parallelization=None,
    batch=False,
    distribution_data=None,
    ray_already_init=False,
    target_model_fct=None,
    loss_fct=None,
):
    # If using Ray, put df_data in object store at the start
    if ray_parallelization is not None:
        import ray

        ray.init()

        df_data_ref = ray.put(df_data)

    else:
        df_data_ref = df_data

    def run_game_with_seed(game_seed):
        # Get df_data from Ray object store if needed
        if ray_parallelization is not None:
            import ray

            df_data_local = ray.get(df_data_ref)
        else:
            df_data_local = df_data_ref

        # Fix membership ratio to 0.5 because the distribution of sets is not uniform
        local_membership_ratio = 0.5

        df_training_data, df_distribution_data = train_test_split(
            df_data_local, test_size=1 - local_membership_ratio, random_state=game_seed
        )

        if target_model_fct is not None:
            target_model = target_model_fct()
            target_model.fit(
                df_training_data[feature_names].values,
                df_training_data[label_name].values,
            )
        else:
            target_model = KNeighborsClassifier(n_neighbors=K)
            target_model.fit(
                df_training_data[feature_names].values,
                df_training_data[label_name].values,
            )

        challenge_points = df_data.loc[
            target_points_indices, feature_names
        ].values.astype(float)
        challenge_labels = df_data.loc[target_points_indices, label_name].values

        if loss_fct is None:
            _, challenge_points_nn_indices = target_model.kneighbors(
                challenge_points, n_neighbors=K
            )
            neighbor_labels = df_training_data[label_name].values[
                challenge_points_nn_indices
            ]
            matches = neighbor_labels == challenge_labels[:, np.newaxis]
            target_model_losses = 1 - np.sum(matches, axis=1) / K
        else:
            # Calculate losses using custom loss function for each point
            target_model_losses = []
            for i in range(len(challenge_points)):
                challenge_point = challenge_points[i]
                challenge_label = challenge_labels[i]
                target_model_loss = loss_fct(
                    target_model, challenge_point, challenge_label
                )
                target_model_losses.append(target_model_loss)
            target_model_losses = np.array(target_model_losses)

        attack_scores = attacker_predict(target_points_indices, target_model_losses, K)

        in_training_data = target_points_indices.isin(df_training_data.index)

        labels = in_training_data.astype(int).tolist()

        return attack_scores, labels, target_points_indices, target_model_losses

    if ray_parallelization is None:
        results = [run_game_with_seed(game_seed) for game_seed in game_seed_list]
    else:
        from rayPlus import parallel_loop

        results = parallel_loop(
            game_seed_list,
            run_game_with_seed,
            return_results=True,
            n_tasks=ray_parallelization["n_tasks"],
            init_and_shutdown_ray=False,
        )
        ray.shutdown()

    return results


def transform_scores(scores):
    # Replace 'inf' with a finite large number (max finite value * inf_replacement_factor)
    max_finite_value = max(filter(lambda x: x != float("inf"), scores))

    scores_replaced_inf = [max_finite_value if x == float("inf") else x for x in scores]

    # Apply a logarithmic transformation
    # scores_log_transformed = np.log(scores_replaced_inf)

    return scores_replaced_inf


def compute_auc_for_scores(seed_list, all_scores):
    all_fpr = np.linspace(0, 1, 100)
    tprs = []
    auc_scores = []

    for i, seed in enumerate(seed_list):
        scores, labels, _, _ = all_scores[i]
        scores = transform_scores(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        tprs.append(np.interp(all_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the curve ends at (1, 1)
    mean_auc = auc(all_fpr, mean_tpr)
    std_auc = np.std(auc_scores)
    std_tpr = np.std(tprs, axis=0)

    upper_tpr = np.minimum(mean_tpr + std_tpr, 1)
    lower_tpr = np.maximum(mean_tpr - std_tpr, 0)

    return all_fpr, mean_tpr, upper_tpr, lower_tpr, mean_auc, std_auc


def compute_auc_for_scores_with_ci(seed_list, all_scores, alpha=0.05):
    all_fpr = np.linspace(0, 1, 100)
    tprs = []
    auc_scores = []

    for i, seed in enumerate(seed_list):
        scores, labels, _, _ = all_scores[i]
        scores = transform_scores(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        tprs.append(np.interp(all_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the curve ends at (1, 1)
    mean_auc = auc(all_fpr, mean_tpr)

    # Compute the standard error of the mean AUC
    sem_auc = stats.sem(auc_scores)

    # Compute the confidence interval for the AUC scores using the normal distribution
    ci_lower = mean_auc - stats.norm.ppf(1 - alpha / 2) * sem_auc
    ci_upper = mean_auc + stats.norm.ppf(1 - alpha / 2) * sem_auc

    tprs = np.array(tprs)
    tpr_lower = np.percentile(tprs, 100 * alpha / 2, axis=0)
    tpr_upper = np.percentile(tprs, 100 * (1 - alpha / 2), axis=0)

    return all_fpr, mean_tpr, tpr_upper, tpr_lower, mean_auc, ci_lower, ci_upper


def compute_tpr_at_fpr_with_ci(seed_list, all_scores, fpr_threshold=0.05, alpha=0.05):
    all_fpr = np.linspace(0, 1, 100)
    tprs = []
    auc_scores = []

    for i, seed in enumerate(seed_list):
        scores, labels, _, _ = all_scores[i]
        scores = transform_scores(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        tprs.append(np.interp(all_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the curve ends at (1, 1)
    mean_auc = auc(all_fpr, mean_tpr)

    tprs = np.array(tprs)
    tpr_at_fpr = np.interp(fpr_threshold, all_fpr, mean_tpr)
    tpr_at_fpr_upper = np.percentile(tprs, 100 * (1 - alpha / 2), axis=0)
    tpr_at_fpr_lower = np.percentile(tprs, 100 * alpha / 2, axis=0)

    tpr_at_fpr_upper = np.interp(fpr_threshold, all_fpr, tpr_at_fpr_upper)
    tpr_at_fpr_lower = np.interp(fpr_threshold, all_fpr, tpr_at_fpr_lower)

    return tpr_at_fpr, tpr_at_fpr_lower, tpr_at_fpr_upper


def compute_auc_for_scores_4print(seed_list, all_scores, alpha=0.05):
    all_fpr = np.linspace(0, 1, 100)
    tprs = []
    auc_scores = []

    for i, seed in enumerate(seed_list):
        scores, labels, _, _ = all_scores[i]
        scores = transform_scores(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        tprs.append(np.interp(all_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the curve ends at (1, 1)
    mean_auc = auc(all_fpr, mean_tpr)
    std_auc = np.std(auc_scores)

    # Compute the standard error of the mean TPR at each FPR
    sem_tpr = stats.sem(tprs, axis=0)

    # Compute the confidence interval for the TPR at each FPR
    ci_lower_tpr = mean_tpr - stats.norm.ppf(1 - alpha / 2) * sem_tpr
    ci_upper_tpr = mean_tpr + stats.norm.ppf(1 - alpha / 2) * sem_tpr

    # Ensure the CI bounds are within valid range [0, 1]
    ci_lower_tpr = np.maximum(ci_lower_tpr, 0)
    ci_upper_tpr = np.minimum(ci_upper_tpr, 1)

    return all_fpr, mean_tpr, mean_auc, std_auc, ci_lower_tpr, ci_upper_tpr


def plot_roc_curve(
    all_fpr,
    mean_tpr,
    lower_tpr,
    upper_tpr,
    mean_auc,
    std_auc,
    attack_name,
    plot_folder,
    use_log_scale=True,
):
    plt.figure()
    plt.plot(
        all_fpr,
        mean_tpr,
        color="darkorange",
        label="Mean ROC (AUC = %0.2f $\\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
    )
    plt.fill_between(
        all_fpr, lower_tpr, upper_tpr, color="#FFDAB9", alpha=0.3, label="± std. dev."
    )
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="navy")

    if use_log_scale:
        plt.xscale("log")
        plt.yscale("log")
        # Adjust the bounds for log scale, ensuring no zero or negative values
        plt.xlim([max(min(all_fpr), 0.001), 1.0])  # Adjust if necessary
        plt.ylim([max(min(lower_tpr), 0.001), 1.05])  # Adjust if necessary
    else:
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - {attack_name}")

    plt.legend(loc="lower right")
    plt.savefig(
        f"{plot_folder}/{attack_name}_ROC{'_log_log' if use_log_scale else ''}.png"
    )
    plt.close()


def analyze_auc_convergence(seed_list, all_scores, plot_folder, attack_name):
    subset_sizes = np.linspace(1, len(seed_list), num=10, dtype=int)
    mean_aucs = []
    std_aucs = []

    for size in subset_sizes:
        _, _, _, _, mean_auc, std_auc = compute_auc_for_scores(
            seed_list[:size], all_scores[:size]
        )
        mean_aucs.append(mean_auc)
        std_aucs.append(std_auc)

    plt.figure()
    # plt.errorbar(subset_sizes, mean_aucs, yerr=std_aucs, fmt='-o')
    plt.title("AUC Convergence Analysis")
    plt.xlabel("Number of Games")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.savefig(f"{plot_folder}/{attack_name}_auc_convergence.png")
    plt.close()

    return mean_aucs, std_aucs


def run_multiple_games_and_plot(
    df_data,
    K,
    attacker_predict,
    number_of_games,
    attack_name,
    plot_folder,
    feature_names=None,
    label_name=None,
    seed_list=list(range(42, 63)),
    membership_ratio=0.5,
    ray_parallelization=None,
    batch=False,
    distribution_data=None,
    ray_already_init=False,
    target_model_fct=None,
    loss_fct=None,
):
    """
    Runs multiple security games and generates evaluation plots.

    Parameters:
    -----------
    df_data : pandas.DataFrame
        The dataset containing features and labels
    K : int
        Number of nearest neighbors for KNN model
    attacker_predict : callable
        Function implementing the attacker's prediction strategy
    number_of_games : int
        Number of security games to run
    attack_name : str
        Name of the attack for plot titles and file names
    plot_folder : str
        Directory to save plots and results
    feature_names : list, optional
        Names of feature columns
    label_name : str, optional
        Name of label column
    seed_list : list, default=range(42, 63)
        List of random seeds for each game
    membership_ratio : float, default=0.5
        Ratio of data to use for training vs distribution
    ray_parallelization : dict, optional
        Configuration for Ray parallelization
    batch : bool, default=False
        Whether to run attacks in batch mode
    distribution_data : pandas.DataFrame, optional
        Pre-defined distribution data
    ray_already_init : bool, default=False
        Whether Ray is already initialized
    target_model_fct : callable, optional
        Function to create target model
    loss_fct : callable, optional
        Custom loss function

    Returns:
    --------
    list
        Results from all security games
    """
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # list of (scores, labels, indices) for each game
    all_scores = run_multiple_games(
        df_data,
        K,
        attacker_predict,
        number_of_games,
        seed_list,
        membership_ratio,
        feature_names=feature_names,
        label_name=label_name,
        ray_parallelization=ray_parallelization,
        batch=batch,
        distribution_data=distribution_data,
        ray_already_init=ray_already_init,
        target_model_fct=target_model_fct,
        loss_fct=loss_fct,
    )

    # pickle all scores
    with open(f"{plot_folder}/{attack_name}_scores.pkl", "wb") as f:
        pickle.dump(all_scores, f)

    # TODO : technically the part below could be in a separate function

    all_fpr, mean_tpr, upper_tpr, lower_tpr, mean_auc, std_auc = compute_auc_for_scores(
        seed_list, all_scores
    )

    # TPR at a fixed low FPR (e.g., 0.001% or 0.1%). Here we pick 5%.
    tpr_at_fpr = np.interp(0.05, all_fpr, mean_tpr)

    # save statistics to csv file
    with open(f"{plot_folder}/{attack_name}_statistics.csv", "w") as f:
        f.write(f"mean_auc,std_auc,tpr_at_fpr\n")
        f.write(f"{mean_auc},{std_auc},{tpr_at_fpr}\n")

    plot_roc_curve(
        all_fpr,
        mean_tpr,
        lower_tpr,
        upper_tpr,
        mean_auc,
        std_auc,
        attack_name,
        plot_folder,
        True,
    )
    plot_roc_curve(
        all_fpr,
        mean_tpr,
        lower_tpr,
        upper_tpr,
        mean_auc,
        std_auc,
        attack_name,
        plot_folder,
        False,
    )

    # Analyze and plot AUC convergence
    # analyze_auc_convergence(seed_list, all_scores, plot_folder, attack_name)

    return all_scores


def scores_to_outlier_indicators(scores):
    mean_value = np.mean(scores)
    std_deviation = np.std(scores)
    outlier_indicators = []

    for score in scores:
        # Compute the z-score
        z_score = (score - mean_value) / std_deviation

        # Use the CDF to get a probability indicator
        probability_indicator = norm.cdf(z_score)

        # Convert this into an outlier likelihood indicator Since the CDF gives
        # us the probability of a value being less than z, we can interpret
        # extreme values (close to 0 or 1) as being outliers. However, to map
        # this directly to an indicator where higher values indicate higher
        # outlier likelihood:
        outlier_indicator = 2 * min(probability_indicator, 1 - probability_indicator)

        outlier_indicators.append(outlier_indicator)

    return outlier_indicators


def build_attacker_predict_shapley(all_data, all_labels, test_data, test_labels, K):
    average_shapley_values = knn_valuation.compute_training_average_shapley_values(
        all_data, all_labels, test_data, test_labels, K, only_average=True
    )

    def attacker_predict_shapley(
        challenge_point, challenge_label, target_model_loss, K, challenge_point_idx
    ):

        # find for the closest point in all_data to data_point
        sorted_indices = utils.order_points_by_distance(
            all_data, np.array([challenge_point])
        )

        # find the shapley value of the closest point TODO : understand why
        # if you put negative shapley works better than others in some
        # cases it probably is because the high value are a counter
        # indicator of the membership
        shapley_value = average_shapley_values[sorted_indices[0][0]]

        return shapley_value

    return attacker_predict_shapley


def build_attacker_predict_self_shapley(all_data, all_labels, K):
    average_shapley_values = knn_valuation.compute_self_training_average_shapley_values(
        all_data, all_labels, K
    )

    def attacker_predict_shapley(
        challenge_point, challenge_label, target_model_loss, K, challenge_point_idx
    ):

        # find for the closest point in all_data to data_point
        sorted_indices = utils.order_points_by_distance(
            all_data, np.array([challenge_point])
        )

        # find the shapley value of the closest point TODO : understand why
        # if you put negative shapley works better than others in some
        # cases it probably is because the high value are a counter
        # indicator of the membership
        shapley_value = average_shapley_values[sorted_indices[0][0]]

        return shapley_value

    return attacker_predict_shapley


def build_attacker_predict_self_shapley__old(all_data, all_labels):
    def attacker_predict_self_shapley(data_point, data_point_label, target_model, K):
        distances = cdist(all_data, np.array([data_point]), default_distance_metric)

        # Find the index of the closest point
        closest_point_index = np.argmin(distances)

        # Replace the closest point with the data point in a new dataset
        modified_data = np.copy(all_data)
        modified_labels = np.copy(all_labels)
        modified_data[closest_point_index] = data_point
        modified_labels[closest_point_index] = (
            data_point_label  # Ensure label consistency
        )

        # make data_point and data_point_labels arrays of size 1
        data_point_array = np.array([data_point])
        data_point_label_array = np.array([data_point_label])

        self_shapley_values = knn_valuation.compute_training_average_shapley_values(
            modified_data, modified_labels, data_point_array, data_point_label_array, K
        )

        return self_shapley_values[closest_point_index]

        # return self_shapley_values[closest_point_index]

    return attacker_predict_self_shapley


def build_attacker_predict_self_waka(all_data, all_labels):
    def attacker_predict_self_waka(data_point, data_point_label, target_model, K):
        distances = cdist(all_data, np.array([data_point]), default_distance_metric)

        # Find the index of the closest point
        closest_point_index = np.argmin(distances)

        # Replace the closest point with the data point in a new dataset
        modified_data = np.copy(all_data)
        modified_labels = np.copy(all_labels)
        modified_data[closest_point_index] = data_point
        modified_labels[closest_point_index] = (
            data_point_label  # Ensure label consistency
        )

        # make data_point and data_point_labels arrays of size 1
        data_point_array = np.array([data_point])
        data_point_label_array = np.array([data_point_label])

        self_waka_values = compute_unormalized_average_waka_values(
            modified_data,
            modified_labels,
            data_point_array,
            data_point_label_array,
            K,
            default_distance_metric=default_distance_metric,
        )

        return self_waka_values[closest_point_index]

    return attacker_predict_self_waka


def build_attacker_predict_lira_all_scores(all_data, all_labels, N_shadow=1, seed=42):
    def attacker_predict_lira_all_scores(indices, target_losses, K):
        return lira_vanilla_scores(
            all_data,
            all_labels,
            target_losses,
            K,
            N_shadow,
            seed,
            target_indices=indices,
        )

    return attacker_predict_lira_all_scores


def build_attacker_predict_lira_regression_all_scores(
    all_data, all_labels, N_shadow=1, seed=42, model_loss_fct=None
):

    model_params = {"solver": "liblinear", "max_iter": 100}
    model_cls = LogisticRegression

    def attacker_predict_lira_all_scores(indices, target_losses, K):
        return lira_model_agnostic_scores(
            all_data,
            all_labels,
            target_losses,
            model_cls=model_cls,
            model_params=model_params,
            target_indices=indices,
            N_shadow=N_shadow,
            seed=42,
            should_logit_transform=True,
            model_loss_fct=model_loss_fct,
        )

    return attacker_predict_lira_all_scores


def build_attacker_predict_waka_target(
    all_data,
    all_labels,
    normalized=False,
    k1function=False,
    approx=None,
    K=None,
    knn_model_ref=None,
    nbr_of_considered_points=100,
    knn_model_copy=None,
):

    def attacker_predict_waka_target(
        data_point, data_point_label, target_model_loss, K, challenge_point_idx
    ):

        return compute_self_waka_value_with_optional_target_with_knn(
            all_labels,
            np.array([all_data[challenge_point_idx]]),
            np.array([all_labels[challenge_point_idx]]),
            K,
            target_model_loss,
            default_distance_metric=default_distance_metric,
            knn_model_ref=knn_model_ref,
            knn_model_copy=knn_model_copy,
            nbr_of_considered_points=nbr_of_considered_points,
            target_point_idx=challenge_point_idx,
            all_data=all_data,
        )[0]

    return attacker_predict_waka_target


def build_attacker_predict_waka_without_target(
    all_data, all_labels, normalized=False, k1function=False, approx=None
):
    def attacker_predict_waka_without_target(
        data_point, data_point_label, target_model_loss, K
    ):
        distances = np.linalg.norm(all_data - data_point, axis=1)

        # Find the index of the closest point
        closest_point_index = np.argmin(distances)

        # Replace the closest point with the data point in a new dataset
        modified_data = np.copy(all_data)
        modified_labels = np.copy(all_labels)
        modified_data[closest_point_index] = data_point
        modified_labels[closest_point_index] = (
            data_point_label  # Ensure label consistency
        )

        # make data_point and data_point_labels arrays of size 1
        data_point_array = np.array([data_point])
        data_point_label_array = np.array([data_point_label])

        modified_data_minus_idx = np.delete(modified_data, closest_point_index, 0)
        modified_labels_minux_idx = np.delete(modified_labels, closest_point_index, 0)

        return compute_self_waka_value_with_optional_target(
            modified_data_minus_idx,
            modified_labels_minux_idx,
            data_point_array,
            data_point_label_array,
            K,
            default_distance_metric=default_distance_metric,
        )[0]

    return attacker_predict_waka_without_target


# %%


def build_attacker_predict_waka(all_data, all_labels):

    # from utils import order_points_by_distance
    average_waka_values = compute_unormalized_average_waka_values(
        all_data, all_labels, test_points, test_labels, K
    )

    def attacker_predict_waka(data_point, data_point_label, target_model, K):

        # find for the closest point in all_data to data_point
        sorted_indices = utils.order_points_by_distance(all_data, [data_point])

        # find the shapley value of the closest point
        waka_value = average_waka_values[sorted_indices[0][0]]

        return waka_value

    return attacker_predict_waka


# %%


def build_attacker_predict_lira(
    all_data, all_labels, N_shadow=1000, shadow_dataset_proportion=[0.2, 1]
):

    def attacker_predict_lira(
        challenge_data_point,
        challenge_data_point_label,
        target_model_loss,
        K,
        challenge_point_idx,
    ):

        # Perform LiRA attack to predict if data_point was in the training set
        score = lira_online_attack_reg(
            all_data,
            all_labels,
            target_model_loss,
            challenge_data_point,
            challenge_data_point_label,
            N_shadow,
            K,
            shadow_dataset_proportion=shadow_dataset_proportion,
            default_distance_metric=default_distance_metric,
        )
        return score

    return attacker_predict_lira


def build_attacker_predict_lira_vanilla(all_data, all_labels, K, N_shadow=10, seed=42):

    seed_list = list(range(seed, seed + N_shadow))
    shadows = []
    sample_size = int(len(all_data)) // 2

    for shadow_seed in seed_list:

        rng = np.random.RandomState(shadow_seed)

        shadow_indices = rng.choice(len(all_data), sample_size, replace=True)

        # indices that are not shadow indices
        non_shadow_indices = np.array(
            [i for i in range(len(all_data)) if i not in shadow_indices]
        )

        # train a KNN model on the shadow indices
        shadow_data = all_data[shadow_indices]
        shadow_labels = all_labels[shadow_indices]
        shadow_model1 = KNeighborsClassifier(n_neighbors=K)
        shadow_model1.fit(shadow_data, shadow_labels)

        # train a KNN model on the non shadow indices
        shadow_data2 = all_data[non_shadow_indices]
        shadow_labels2 = all_labels[non_shadow_indices]
        shadow_model2 = KNeighborsClassifier(n_neighbors=K)
        shadow_model2.fit(shadow_data2, shadow_labels2)

        shadows.append((shadow_model1, shadow_model2))

    def attacker_predict_lira_vanilla(
        challenge_data_point,
        challenge_data_point_label,
        target_model_loss,
        K,
        challenge_point_idx,
    ):

        # Perform LiRA attack to predict if data_point was in the training set
        score = lira_vanilla_attack(
            all_data,
            all_labels,
            target_model_loss,
            challenge_data_point,
            challenge_data_point_label,
            challenge_point_idx,
            shadows,
            K,
            seed_list,
            default_distance_metric=default_distance_metric,
        )
        return score

    return attacker_predict_lira_vanilla


def build_attacker_predict_lira_knn_uniform(all_data, all_labels):
    def attacker_predict_lira_knn_uniform(
        challenge_data_point, challenge_data_point_label, target_model_loss, K
    ):

        # Perform LiRA attack to predict if data_point was in the training set
        score = lira_knn_online_attack_uniform(
            all_data,
            all_labels,
            target_model_loss,
            challenge_data_point,
            challenge_data_point_label,
            K,
        )

        return score

    return attacker_predict_lira_knn_uniform


def build_attacker_predict_lira_knn_localsearch(all_data, all_labels):
    def attacker_predict_lira_knn(
        challenge_data_point, challenge_data_point_label, target_model_loss, K
    ):

        # Perform LiRA attack to predict if data_point was in the training set
        score = lira_knn_online_attack_localsearch(
            all_data,
            all_labels,
            target_model_loss,
            challenge_data_point,
            challenge_data_point_label,
            K,
        )

        return score

    return attacker_predict_lira_knn


def build_attacker_predict_knn_density(all_data, K):
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(
        n_neighbors=K + 1
    )  # K+1 because the point itself will be included
    nn.fit(all_data)
    # Compute distances for all points in the dataset to their K nearest
    # neighbors
    all_distances = []
    for point in all_data:
        dists, _ = nn.kneighbors([point])
        # Exclude the point itself and take the average of the distances
        all_distances.append(np.mean(dists[0][1:]))

    def attacker_predict_knn_distance(
        challenge_data_point, challenge_data_point_label, target_model, K
    ):

        # Perform LiRA attack to predict if data_point was in the training set
        return knn_distance_attack(challenge_data_point, nn, all_distances, K)

    return attacker_predict_knn_distance


def all_attacks_experiment(dataset_name="celeba/"):

    synthetic_data_folders = [dataset_name]

    nbr_of_tasks = 2  # 48
    ray_parallelization_config = None  # {"n_tasks": nbr_of_tasks}

    for data_folder in synthetic_data_folders:
        print(f"Running for {data_folder}")

        number_of_iteration_per_game = 50

        global_seed = 42

        number_of_games = 50

        n_tasks = 5  # number_of_games TODO : CHANGE

        nbr_of_considered_points = 100

        game_seeds = list(range(global_seed, global_seed + number_of_games))

        load_addional_data = False

        if load_addional_data:
            df_data, df_test, dist_data, feature_names, label_name = data_loader.load(
                data_folder, load_addition_data=load_addional_data
            )
            distribution_data = pd.concat([df_test, dist_data])

            # reset the index
            distribution_data = distribution_data.reset_index(drop=True)
        else:
            df_data, df_test, feature_names, label_name = data_loader.load(data_folder)

        start_time = time.time()
        for K in list(range(1, 6)):
            print(f"Running for K={K}")

            plot_folder = data_folder + f"multiple_attacks/K{K}"
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)

            import ray

            knn_model = NearestNeighbors(n_neighbors=K)
            knn_model.fit(df_data[feature_names].values)

            if ray_parallelization_config is not None:

                ray.init()
                knn_model_ref = ray.put(knn_model)
            else:
                knn_model_ref = None

            # TODO: I AM HERE
            waka_scores = run_multiple_games_and_plot(
                df_data,
                K,
                build_attacker_predict_waka_target(
                    df_data[feature_names].values,
                    df_data[label_name].values,
                    normalized=False,
                    approx="quantile",
                    K=K,
                    knn_model_ref=knn_model_ref,
                    nbr_of_considered_points=nbr_of_considered_points,
                    knn_model_copy=(
                        knn_model if ray_parallelization_config is None else None
                    ),
                ),
                number_of_iteration_per_game,
                "WaKA-Target-New",
                plot_folder,
                feature_names=feature_names,
                label_name=label_name,
                seed_list=game_seeds,
                ray_parallelization=None,  # ray_parallelization_config,
                distribution_data=None,  # distribution_data if load_addional_data else None,
                ray_already_init=True,
            )

            if ray_parallelization_config is not None:
                ray.shutdown()

            # End time
            end_time = time.time()
            # Calculate the time taken
            execution_time = end_time - start_time
            print(f"WaKA Execution time: {execution_time:.4f} seconds")

            run_multiple_games_and_plot(
                df_data,
                K,
                build_attacker_predict_shapley(
                    df_data[feature_names].values,
                    df_data[label_name].values,
                    df_test[feature_names].values,
                    df_test[label_name].values,
                    K,
                ),
                number_of_iteration_per_game,
                "Shapley",
                plot_folder,
                feature_names=feature_names,
                label_name=label_name,
                seed_list=game_seeds,
                ray_parallelization=ray_parallelization_config,
                batch=False,
                distribution_data=distribution_data if load_addional_data else None,
            )

            run_multiple_games_and_plot(
                df_data,
                K,
                build_attacker_predict_self_shapley(
                    df_data[feature_names].values, df_data[label_name].values, K
                ),
                number_of_iteration_per_game,
                "Self-Shapley-New",
                plot_folder,
                feature_names=feature_names,
                label_name=label_name,
                seed_list=game_seeds,
                ray_parallelization=ray_parallelization_config,
                batch=False,
                distribution_data=distribution_data if load_addional_data else None,
            )

            # Start time
            start_time = time.time()

            N_shadow = 16  # df_data[feature_names].values.shape[0] * 2
            print(f"Vanilla LiRA with N_shadow={N_shadow}")
            # run_multiple_games_and_plot(df_data, K, build_attacker_predict_lira_vanilla(df_data[feature_names].values, df_data[label_name].values, K, N_shadow=N_shadow), number_of_iteration_per_game, 'LiRA-Vanilla', plot_folder, feature_names=feature_names, label_name=label_name, seed_list=game_seeds
            #                             , ray_parallelization=ray_parallelization_config
            #                             )

            run_multiple_games_and_plot(
                df_data,
                K,
                build_attacker_predict_lira_all_scores(
                    df_data[feature_names].values,
                    df_data[label_name].values,
                    N_shadow=N_shadow,
                ),
                number_of_iteration_per_game,
                "LiRA-Batch-Vanilla",
                plot_folder,
                feature_names=feature_names,
                label_name=label_name,
                seed_list=game_seeds,
                ray_parallelization=None,  # ray_parallelization_config,
                batch=True,
                distribution_data=distribution_data if load_addional_data else None,
            )

            N_shadow = 16  # df_data[feature_names].values.shape[0] * 2
            print(f"Vanilla LiRA NON-BATCH with N_shadow={N_shadow}")

            all_data_size = df_data[label_name].values.shape[0]
            proportion_max = 1000 / all_data_size
            run_multiple_games_and_plot(
                df_data,
                K,
                build_attacker_predict_lira(
                    df_data[feature_names].values,
                    df_data[label_name].values,
                    N_shadow=N_shadow,
                    shadow_dataset_proportion=[proportion_max, proportion_max],
                ),
                number_of_iteration_per_game,
                "LiRA-Non-Batch",
                plot_folder,
                feature_names=feature_names,
                label_name=label_name,
                seed_list=game_seeds,
                ray_parallelization=ray_parallelization_config,
            )
            # End time
            end_time = time.time()
            # Calculate the time taken
            execution_time = end_time - start_time
            print(f"LirA Execution time: {execution_time:.4f} seconds")

            N_shadow = 16  # df_data[feature_names].values.shape[0] * 2
            print(f"LiRA on Regression with N_shadow={N_shadow}")

            model_params = {"solver": "liblinear", "max_iter": 100}
            model_cls = LogisticRegression

            def log_loss_fct(model, challenge_point, challenge_label):

                # Ensure challenge_point is in the correct format
                challenge_point = np.array(challenge_point).reshape(1, -1)

                # Check if the model has `predict_proba`
                if hasattr(model, "predict_proba"):
                    # Get predicted probabilities for the challenge point
                    probs = model.predict_proba(challenge_point)

                    # Ensure challenge_label is correctly formatted (in case of multi-class classification)
                    correct_label_prob = probs[
                        0, challenge_label
                    ]  # Probability for the true class

                    loss = correct_label_prob

                else:
                    raise ValueError("Model does not have a `predict_proba` method")

                return loss

            run_multiple_games_and_plot(
                df_data,
                1,
                build_attacker_predict_lira_regression_all_scores(
                    df_data[feature_names].values,
                    df_data[label_name].values,
                    N_shadow=N_shadow,
                    model_loss_fct=log_loss_fct,
                ),
                number_of_iteration_per_game,
                "LiRA-Regression",
                plot_folder,
                feature_names=feature_names,
                label_name=label_name,
                seed_list=game_seeds,
                ray_parallelization=None,  # ray_parallelization_config,
                batch=True,
                distribution_data=distribution_data if load_addional_data else None,
                target_model_fct=lambda: model_cls(**model_params),
                loss_fct=log_loss_fct,
            )

            def prob_loss_fct(model, challenge_point, challenge_label):

                # Ensure challenge_point is in the correct format
                challenge_point = np.array(challenge_point).reshape(1, -1)

                # Check if the model has `predict_proba`
                if hasattr(model, "predict_proba"):
                    # Get predicted probabilities for the challenge point
                    probs = model.predict_proba(challenge_point)

                    # Ensure challenge_label is correctly formatted (in case of multi-class classification)
                    correct_label_prob = probs[
                        0, challenge_label
                    ]  # Probability for the true class

                    # Compute the log-loss for the challenge point (consistent with previous implementation)
                    loss = 1 - correct_label_prob

                else:
                    raise ValueError("Model does not have a `predict_proba` method")

                return loss

            import ray

            K = 1

            knn_model = NearestNeighbors(n_neighbors=K)
            knn_model.fit(df_data[feature_names].values)

            if ray_parallelization_config is not None:

                ray.init()
                knn_model_ref = ray.put(knn_model)
            else:
                knn_model_ref = None

            run_multiple_games_and_plot(
                df_data,
                K,
                build_attacker_predict_waka_target(
                    df_data[feature_names].values,
                    df_data[label_name].values,
                    normalized=False,
                    approx="quantile",
                    K=K,
                    knn_model_ref=knn_model_ref,
                    nbr_of_considered_points=nbr_of_considered_points,
                    knn_model_copy=(
                        knn_model if ray_parallelization_config is None else None
                    ),
                ),
                number_of_iteration_per_game,
                "LiRA-Regression-with-WaKA",
                plot_folder,
                feature_names=feature_names,
                label_name=label_name,
                seed_list=game_seeds,
                ray_parallelization=ray_parallelization_config,
                batch=False,
                distribution_data=distribution_data if load_addional_data else None,
                target_model_fct=lambda: model_cls(**model_params),
                loss_fct=prob_loss_fct,
                ray_already_init=True,
            )

            if ray_parallelization_config is not None:
                ray.shutdown()


def lira_experiment():

    synthetic_data_folders = [
        "synthetic-overlapping/",
        "synthetic-closerings/",
        "synthetic-fatmoons/",
        "synthetic-overlapping-multi/",
    ]

    number_of_iteration_per_game = 50  # number of points
    global_seed = 42
    number_of_games = 48

    N_shadows = list(range(20, 100, 20))
    nbr_of_bootstrap = 5

    force_overwrite = True

    game_seeds = list(range(global_seed, global_seed + number_of_games))

    ray_parallelization = {"n_tasks": number_of_games, "init_and_shutdown_ray": False}

    if not ray_parallelization is None and ray_parallelization["init_and_shutdown_ray"]:
        import ray

        ray.init()

    for synthetic_data_folder in synthetic_data_folders:
        print(f"Running for {synthetic_data_folder}")

        file_path = synthetic_data_folder + "training.csv"
        df_data = pd.read_csv(file_path)

        plot_folder_base = synthetic_data_folder + "lira-experiment/"

        for K in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print(f"Running for K={K}")

            plot_folder = plot_folder_base + f"K{K}"
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)

            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)

            all_data = df_data[feature_names].values
            all_labels = df_data[label_name].values

            regularization_term = 0.1
            membership_ratio = 0.5

            def compute_games_likelihood_ratios():

                def compute_game_likelihood_ratios(game_seed):
                    # prepare game
                    df_training_data, df_distribution_data = train_test_split(
                        df_data, test_size=1 - membership_ratio, random_state=game_seed
                    )
                    target_model = KNeighborsClassifier(n_neighbors=K)
                    target_model.fit(
                        df_training_data[feature_names].values,
                        df_training_data[label_name].values,
                    )

                    random.seed(game_seed)
                    random_indices = [
                        select_random_index(df_training_data, df_distribution_data)
                        for _ in range(number_of_iteration_per_game)
                    ]  # list of (idx, b)

                    def bootstrap_likelihood_ratios(
                        all_data,
                        all_labels,
                        challenge_point,
                        challenge_label,
                        N_shadows,
                        K,
                        nbr_of_bootstrap,
                        target_model_loss,
                        regularization_term,
                    ):
                        bootstrap_results = {}

                        for N_shadow_per_step in N_shadows:
                            likelihood_ratios = []
                            standard_errors = []  # Store SE for each iteration

                            for seed in range(nbr_of_bootstrap):
                                # Assuming train_shadow_models_and_collect_scores and compute_likelihood_ratio_reg are defined
                                losses_in, losses_out = (
                                    train_shadow_models_and_collect_scores(
                                        all_data,
                                        all_labels,
                                        challenge_point,
                                        challenge_label,
                                        N_shadow_per_step,
                                        K,
                                        seed=seed,  # Using iteration count as seed
                                        shadow_dataset_proportion=[0.2, 1],
                                        ray_parallelization=None,
                                    )
                                )  # Adapt as necessary

                                # Compute the likelihood ratio for this bootstrap sample
                                likelihood_ratio = compute_likelihood_ratio_reg(
                                    losses_in,
                                    losses_out,
                                    target_model_loss,
                                    regularization_term,
                                )
                                likelihood_ratios.append(likelihood_ratio)

                                # Calculate standard error of the mean up to the current iteration
                                current_se = sem(
                                    likelihood_ratios
                                )  # Standard error of the mean
                                standard_errors.append(current_se)

                            # Calculate the mean and 95% confidence interval using the final list of likelihood ratios
                            mean_lr = np.mean(likelihood_ratios)
                            final_se = standard_errors[-1]  # The last SE calculated
                            ci_lr = (
                                mean_lr - 1.96 * final_se,
                                mean_lr + 1.96 * final_se,
                            )  # Approximate CI for large sample sizes

                            # Store results for this N_shadow setting
                            bootstrap_results[N_shadow_per_step] = {
                                "mean": mean_lr,
                                "SE": final_se,
                                "CI": ci_lr,
                                "all_SEs": standard_errors,  # Optional: Store all SEs for analysis
                            }

                        return bootstrap_results

                    def bootstrap_likelihood_ratios_for_point_at_idx(idx, b):

                        challenge_point = df_data.loc[idx, feature_names].values
                        challenge_label = df_data.loc[idx, label_name]

                        _, challenge_points_nn_indices = target_model.kneighbors(
                            np.array([challenge_point]), n_neighbors=K
                        )
                        neighbor_labels = df_training_data[label_name].values[
                            challenge_points_nn_indices
                        ]
                        target_model_loss = (
                            np.sum(neighbor_labels == challenge_label) / K
                        )

                        likelihood_ratios = bootstrap_likelihood_ratios(
                            all_data,
                            all_labels,
                            challenge_point,
                            challenge_label,
                            N_shadows,
                            K,
                            nbr_of_bootstrap,
                            target_model_loss,
                            regularization_term,
                        )

                        return likelihood_ratios, b

                    return [
                        bootstrap_likelihood_ratios_for_point_at_idx(idx, b)
                        for idx, b in random_indices
                    ]

                if ray_parallelization is None:
                    return [
                        compute_game_likelihood_ratios(game_seed)
                        for game_seed in game_seeds
                    ]
                else:
                    from rayPlus import parallel_loop

                    return parallel_loop(
                        game_seeds,
                        compute_game_likelihood_ratios,
                        return_results=True,
                        n_tasks=ray_parallelization["n_tasks"],
                    )

            # structure : game -> point -> (N_shadow -> stats, label)
            games_likelihood_ratios = load_or_compute_and_save(
                "LiRA points games likelihood ratios",
                f"{plot_folder}/games_likelihood_ratios.pkl",
                compute_games_likelihood_ratios,
                overwrite=force_overwrite,
            )

            # Initialize storage for aggregated results and labels
            aggregate_means = {
                n: [] for n in N_shadows
            }  # For storing all means for each N_shadow across games
            labels = []  # To store labels of all points across all games

            for game_results in games_likelihood_ratios:
                for lr, label in game_results:
                    labels.append(label)  # Append label
                    for (
                        n_shadow,
                        stats,
                    ) in (
                        lr.items()
                    ):  # Iterate through each N_shadow's stats in the tuple
                        aggregate_means[n_shadow].append(
                            stats["mean"]
                        )  # Append mean likelihood ratio

            auc_scores = {}
            tpr_at_fpr_scores = {}

            for n_shadow in N_shadows:
                scores = aggregate_means[n_shadow]
                fpr, tpr, thresholds = roc_curve(labels, scores)
                auc_scores[n_shadow] = auc(fpr, tpr)
                tpr_at_fpr_scores[n_shadow] = np.interp(
                    0.05, fpr, tpr
                )  # Interpolate TPR at 5% FPR

            plt.figure(figsize=(10, 6))
            n_shadows_ordered = sorted(N_shadows)
            means = [np.mean(aggregate_means[n]) for n in n_shadows_ordered]
            std_devs = [np.std(aggregate_means[n], ddof=1) for n in n_shadows_ordered]

            plt.errorbar(
                n_shadows_ordered,
                means,
                yerr=std_devs,
                fmt="-o",
                capsize=5,
                label="Mean LR with Std Dev",
            )
            plt.xticks(n_shadows_ordered, labels=[f"{n}" for n in n_shadows_ordered])
            plt.xlabel("Shadow Model Size (N_shadow)")
            plt.ylabel("Mean Likelihood Ratio")
            plt.title(
                "Mean Likelihood Ratios Across Games for Different N_shadow Sizes"
            )
            plt.legend()
            plt.grid(True, linestyle="--", linewidth=0.5)
            # save the plot
            plt.savefig(f"{plot_folder}/mean_likelihood_ratios.png")

            plt.figure(figsize=(10, 6))
            plt.plot(
                n_shadows_ordered,
                [auc_scores[n] for n in n_shadows_ordered],
                "-o",
                label="AUC",
                color="blue",
                markersize=8,
            )
            plt.plot(
                n_shadows_ordered,
                [tpr_at_fpr_scores[n] for n in n_shadows_ordered],
                "-o",
                label="TPR at 5% FPR",
                color="green",
                markersize=8,
            )
            plt.xlabel("Shadow Model Size (N_shadow)")
            plt.ylabel("Scores")
            plt.title("AUC and TPR at 5% FPR Across Different N_shadow Sizes")
            plt.xticks(n_shadows_ordered)
            plt.legend()
            plt.grid(True, which="major", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            # save the plot
            plt.savefig(f"{plot_folder}/auc_and_tpr_scores.png")

            # Initialize a dictionary to hold the means for each game and N_shadow
            games_means = {
                game_index: {n: [] for n in N_shadows}
                for game_index in range(len(game_seeds))
            }

            # Iterate through games_likelihood_ratios to fill the structure
            for game_index, game_results in enumerate(games_likelihood_ratios):
                for (
                    lr,
                    _,
                ) in (
                    game_results
                ):  # We ignore the label here as we're focusing on plotting means
                    for n_shadow, stats in lr.items():
                        games_means[game_index][n_shadow].append(
                            stats["mean"]
                        )  # Append mean for this game and N_shadow

            plt.figure(figsize=(12, 8))

            n_shadows_ordered = sorted(N_shadows)  # Ensure N_shadow values are in order

            # Plot a curve for each game
            for game_index, n_shadows_means in games_means.items():
                means_for_game = [
                    np.mean(n_shadows_means[n]) for n in n_shadows_ordered
                ]  # Aggregate means across all points for this game
                plt.plot(
                    n_shadows_ordered,
                    means_for_game,
                    "-o",
                    label=f"Game {game_index + 1}",
                )

            plt.xlabel("Shadow Model Size (N_shadow)")
            plt.ylabel("Mean Likelihood Ratio")
            plt.title(
                "Mean Likelihood Ratios for Each Game Across Different N_shadow Sizes"
            )
            plt.xticks(n_shadows_ordered, labels=[f"{n}" for n in n_shadows_ordered])
            plt.legend()
            plt.grid(True, which="major", linestyle="--", linewidth=0.5)
            # save the plot
            plt.savefig(f"{plot_folder}/game_mean_likelihood_ratios.png")

    if not ray_parallelization is None and ray_parallelization["init_and_shutdown_ray"]:
        ray.shutdown()

    print("Done!")


def compute_target_point_scores(all_games_scores, target_points_indices):
    """
    Computes attack scores and accuracy for target points across multiple games.

    Parameters:
    -----------
    all_games_scores : list
        Results from all security games
    target_points_indices : array-like
        Indices of target points to evaluate

    Returns:
    --------
    tuple
        (target_points_scores, target_points_accuracy) where:
        - target_points_scores: Average attack scores for each target point
        - target_points_accuracy: Average accuracy of membership predictions
    """
    # create an array of the size of target_points_indices
    target_points_scores = np.zeros(len(target_points_indices))
    target_points_accuracy = np.zeros(len(target_points_indices))

    # fpr_min = 0.05
    fpr_min = 0.5

    for (
        attack_scores,
        labels,
        target_points_indices,
        target_model_losses,
    ) in all_games_scores:
        fpr, tpr, thresholds = roc_curve(labels, attack_scores)
        # np.interp(0.05, fpr, tpr)

        # interpolate threshold at 5% FPR
        threshold = np.interp(fpr_min, fpr, thresholds)

        # use the threshold to determine if the target points are in the training set
        predictions = (attack_scores >= threshold).astype(int)

        successes = predictions == labels

        # convert successes into binary array
        successes = successes.astype(int)

        target_points_accuracy += successes

        target_points_scores += np.array(attack_scores)

    target_points_scores /= len(all_games_scores)
    target_points_accuracy /= len(all_games_scores)

    return target_points_scores, target_points_accuracy


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold


def eval_privacy_using_knn(
    df_training,
    K,
    feature_names,
    label_name,
    game_seeds,
    plot_folder,
    should_overwrite,
    ray_parallelization_config,
):
    """
    Evaluates privacy using KNN-based membership inference attacks.

    Parameters:
    -----------
    df_training : pandas.DataFrame
        Training dataset
    K : int
        Number of nearest neighbors for KNN model
    feature_names : list
        Names of feature columns
    label_name : str
        Name of label column
    game_seeds : list
        List of random seeds for games
    plot_folder : str
        Directory to save results
    should_overwrite : bool
        Whether to overwrite existing results
    ray_parallelization_config : dict
        Configuration for Ray parallelization

    Returns:
    --------
    tuple
        (target_points_scores, target_points_accuracy, all_games_scores) where:
        - target_points_scores: Attack scores for target points
        - target_points_accuracy: Accuracy of membership predictions
        - all_games_scores: Results from all security games
    """
    membership_ratio = 0.5

    target_points_indices = df_training.index
    training_features = df_training[feature_names].values
    training_labels = df_training[label_name].values

    N_shadow = 16

    all_games_scores = load_or_compute_and_save(
        "All games training evaluation",
        f"{plot_folder}/all_games_scores_for_targets.pkl",
        lambda: run_multiple_games_with_target_points(
            df_training,
            K,
            build_attacker_predict_lira_all_scores(
                training_features, training_labels, N_shadow=N_shadow
            ),
            game_seeds,
            membership_ratio,
            target_points_indices,
            feature_names=feature_names,
            label_name=label_name,
            distribution_data=None,
            ray_parallelization=ray_parallelization_config,
        ),
        overwrite=should_overwrite,
    )

    target_points_scores, target_points_accuracy = compute_target_point_scores(
        all_games_scores, target_points_indices
    )
    return target_points_scores, target_points_accuracy, all_games_scores


def eval_privacy_using_regression(
    df_training,
    feature_names,
    label_name,
    game_seeds,
    plot_folder,
    should_overwrite,
    ray_parallelization_config,
):
    """
    Evaluates privacy using regression-based membership inference attacks.

    Parameters:
    -----------
    df_training : pandas.DataFrame
        Training dataset
    feature_names : list
        Names of feature columns
    label_name : str
        Name of label column
    game_seeds : list
        List of random seeds for games
    plot_folder : str
        Directory to save results
    should_overwrite : bool
        Whether to overwrite existing results
    ray_parallelization_config : dict
        Configuration for Ray parallelization

    Returns:
    --------
    tuple
        (target_points_scores, target_points_accuracy, all_games_scores) where:
        - target_points_scores: Attack scores for target points
        - target_points_accuracy: Accuracy of membership predictions
        - all_games_scores: Results from all security games
    """
    membership_ratio = 0.5

    target_points_indices = df_training.index
    training_features = df_training[feature_names].values
    training_labels = df_training[label_name].values

    N_shadow = 16

    model_params = {"solver": "liblinear", "max_iter": 100}
    model_cls = LogisticRegression

    def log_loss_fct(model, challenge_point, challenge_label):
        # Ensure challenge_point is in the correct format
        challenge_point = np.array(challenge_point).reshape(1, -1)

        # Check if the model has `predict_proba`
        if hasattr(model, "predict_proba"):
            # Get predicted probabilities for the challenge point
            probs = model.predict_proba(challenge_point)

            # Ensure challenge_label is correctly formatted (in case of multi-class classification)
            correct_label_prob = probs[
                0, challenge_label
            ]  # Probability for the true class

            loss = correct_label_prob

        else:
            raise ValueError("Model does not have a `predict_proba` method")

        return loss

    all_games_scores = load_or_compute_and_save(
        "All games training evaluation",
        f"{plot_folder}/all_games_scores_for_targets.pkl",
        lambda: run_multiple_games_with_target_points(
            df_training,
            1,
            build_attacker_predict_lira_regression_all_scores(
                training_features,
                training_labels,
                N_shadow=N_shadow,
                model_loss_fct=log_loss_fct,
            ),
            game_seeds,
            membership_ratio,
            target_points_indices,
            feature_names=feature_names,
            label_name=label_name,
            distribution_data=None,
            ray_parallelization=ray_parallelization_config,
            target_model_fct=lambda: model_cls(**model_params),
            loss_fct=log_loss_fct,
        ),
        overwrite=should_overwrite,
    )

    target_points_scores, target_points_accuracy = compute_target_point_scores(
        all_games_scores, target_points_indices
    )
    return target_points_scores, target_points_accuracy, all_games_scores


def plot_save_target_points_auc(
    game_seeds, all_games_scores, plot_folder, attack_name="LiRA-All-points"
):
    all_fpr, mean_tpr, upper_tpr, lower_tpr, mean_auc, std_auc = compute_auc_for_scores(
        game_seeds, all_games_scores
    )

    # TPR at a fixed low FPR (e.g., 0.001% or 0.1%). Here we pick 5%.
    tpr_at_fpr = np.interp(0.05, all_fpr, mean_tpr)

    # save statistics to csv file
    with open(f"{plot_folder}/{attack_name}_statistics.csv", "w") as f:
        f.write(f"mean_auc,std_auc,tpr_at_fpr\n")
        f.write(f"{mean_auc},{std_auc},{tpr_at_fpr}\n")

    plot_roc_curve(
        all_fpr,
        mean_tpr,
        lower_tpr,
        upper_tpr,
        mean_auc,
        std_auc,
        attack_name,
        plot_folder,
        True,
    )
    plot_roc_curve(
        all_fpr,
        mean_tpr,
        lower_tpr,
        upper_tpr,
        mean_auc,
        std_auc,
        attack_name,
        plot_folder,
        False,
    )


def read_values_or_compute(
    training_features,
    training_labels,
    val_features,
    val_labels,
    data_values_folder_path,
    config,
    global_overwrite=False,
):

    hp_str = str(config)

    test_shapley_values = load_or_compute_and_save(
        "Test Average Shapley Values",
        f"{data_values_folder_path}/test_average_shapley_values_{hp_str}.pkl",
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

    test_loo_values = load_or_compute_and_save(
        "Test Average LOO Values",
        f"{data_values_folder_path}/test_average_loo_values_{hp_str}.pkl",
        lambda: knn_valuation.compute_training_average_leave_one_out(
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
        "Test Average Waka Values",
        f"{data_values_folder_path}/test_average_waka_values_{hp_str}.pkl",
        lambda: compute_unormalized_average_waka_values(
            training_features,
            training_labels,
            val_features,
            val_labels,
            config["K"],
            approx="quantile",
            ray_parallelization=None,  # {"n_tasks":global_n_tasks},
        ),
        overwrite=global_overwrite,
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

    self_shapley_values = load_or_compute_and_save(
        "Average Shapley Values",
        f"{data_values_folder_path}/self_shapley_values_{hp_str}.pkl",
        lambda: knn_valuation.compute_self_training_average_shapley_values(
            training_features, training_labels, config["K"]
        ),
        overwrite=global_overwrite,
    )

    self_loo_values = load_or_compute_and_save(
        "Average LOO Values",
        f"{data_values_folder_path}/self_loo_values_{hp_str}.pkl",
        lambda: knn_valuation.compute_self_training_average_leave_one_out(
            training_features, training_labels, config["K"]
        ),
        overwrite=global_overwrite,
    )

    self_waka_values_with_influences = load_or_compute_and_save(
        "Average Waka Values",
        f"{data_values_folder_path}/self_waka_values_{hp_str}.pkl",
        lambda: compute_self_unormalized_average_waka_values_recomputable(
            training_features,
            training_labels,
            config["K"],
            ray_parallelization=None,  # {"n_tasks":global_n_tasks},
        ),
        overwrite=global_overwrite,
    )

    self_waka_values = np.array([swv[0] for swv in self_waka_values_with_influences])
    self_waka_influences = np.array(
        [swv[1] for swv in self_waka_values_with_influences]
    )

    return {
        "test_shapley_values": test_shapley_values,
        "test_loo_values": test_loo_values,
        "test_waka_values": test_waka_values,
        "self_shapley_values": self_shapley_values,
        "self_loo_values": self_loo_values,
        "self_waka_values": self_waka_values,
        "self_waka_influences": self_waka_influences,
    }


def curate_and_evaluate_removal(
    removal_type,
    result_key,
    result,
    df_training,
    K,
    preserved_percent,
    data_folder,
    feature_names,
    label_name,
    game_seeds,
    should_overwrite,
    ray_parallelization_config,
    plot_folder,
):

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    ordered_indices = result[result_key].argsort()
    cutoff = int(len(ordered_indices) * preserved_percent)
    selected_indices = ordered_indices[:cutoff]

    df_training_curated = df_training.iloc[selected_indices]
    df_training_curated = df_training_curated.reset_index(drop=True)

    print(f"Evaluating {preserved_percent*100}% points after {removal_type} curation")

    result_data = {}
    result_data["previous_prediction_av"] = result["predictions_av"][selected_indices]
    result_data["selected_indices"] = selected_indices
    result_data["curated_indices"] = ordered_indices[cutoff:]

    target_points_scores, target_points_accuracy, all_games_scores = (
        eval_privacy_using_knn(
            df_training_curated,
            K,
            feature_names,
            label_name,
            game_seeds,
            plot_folder,
            should_overwrite,
            ray_parallelization_config,
        )
    )

    plot_save_target_points_auc(
        game_seeds, all_games_scores, plot_folder, attack_name="LiRA-kNN"
    )

    result_data["predictions_av"] = target_points_accuracy
    # result_data['lira_scores'] = target_points_scores
    result_data["all_games_scores"] = all_games_scores
    result_data["labels"] = df_training_curated[label_name].values

    # Save a pickle file with the data
    with open(f"{plot_folder}/evaluation_k_{K}.pkl", "wb") as f:
        pickle.dump(result_data, f)


def evaluate_training_data(global_seed=42, dataset_name="cifar-imgnetemb/"):

    synthetic_data_folders = [dataset_name]

    global_n_tasks = 4
    ray_parallelization_config = None  # {"n_tasks": global_n_tasks}
    should_overwrite = False
    preserved_percent = 0.90

    root_folder = ""
    # root_folder = "/Users/patrickmesana/Dev/waka_results/experiments-pre-satml/"

    for dataset_name in synthetic_data_folders:

        data_folder = root_folder + dataset_name

        data_values_folder_path = (
            data_folder + f"data_minimization/utility/seed_{global_seed}"
        )

        print(f"Running for {data_folder}")

        number_of_games = 50

        game_seeds = list(range(global_seed, global_seed + number_of_games))

        df_training, test_df, feature_names, label_name = data_loader.load(dataset_name)

        val_df, test_df = train_test_split(
            test_df, test_size=0.5, random_state=global_seed
        )

        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        K_list = [1, 5]

        for K in K_list:

            plot_folder = data_folder + f"multiple_attacks/K{K}/100percent"

            # create plot_folder if it does not exists
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)

            target_points_scores, target_points_accuracy, all_games_scores = (
                eval_privacy_using_knn(
                    df_training,
                    K,
                    feature_names,
                    label_name,
                    game_seeds,
                    plot_folder,
                    should_overwrite,
                    ray_parallelization_config,
                )
            )

            plot_save_target_points_auc(
                game_seeds, all_games_scores, plot_folder, attack_name="LiRA-kNN"
            )

            result = read_values_or_compute(
                df_training[feature_names].values,
                df_training[label_name].values,
                val_df[feature_names].values,
                val_df[label_name].values,
                data_values_folder_path,
                {"K": K},
                False,
            )

            result["all_games_scores"] = all_games_scores
            result["predictions_av"] = target_points_accuracy
            result["labels"] = df_training[label_name].values

            with open(f"{plot_folder}/evaluation_k_{K}.pkl", "wb") as f:
                pickle.dump(result, f)

            removal_types = ["self-waka"]
            removal_keys = ["self_waka_values"]

            for removal_type, removal_key in zip(removal_types, removal_keys):
                curate_and_evaluate_removal(
                    removal_type,
                    removal_key,
                    result,
                    df_training,
                    K,
                    preserved_percent,
                    data_folder,
                    feature_names,
                    label_name,
                    game_seeds,
                    should_overwrite,
                    ray_parallelization_config,
                    data_folder
                    + f"multiple_attacks/K{K}/{int(preserved_percent*100)}percent-{removal_type}-removal",
                )


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "lira":
            lira_experiment()
        elif sys.argv[1] == "attacks":
            dataset_name = sys.argv[2] if len(sys.argv) > 2 else "celeba"
            all_attacks_experiment(dataset_name=dataset_name + "/")
        elif sys.argv[1] == "evaluate":
            dataset_name = sys.argv[2] if len(sys.argv) > 2 else "celeba"
            evaluate_training_data(dataset_name=dataset_name + "/")
        else:
            print("Unknown option")
    else:
        # evaluate_training_data(dataset_name='bank/')  # Default behavior
        all_attacks_experiment(dataset_name="imdb/")
