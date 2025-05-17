import numpy as np

from scipy.special import comb


from scipy.spatial.distance import cdist

from tqdm import tqdm

import math
from math import log, log2

from utils import time_it, check_number, cumsum

from time import sleep

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

import ray

# export RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1


def cummulative_nbr_of_combinations_for_set(n, fct=None):
    """
    Calculate the cumulative number of combinations for a set of size n.

    Args:
        n (int): Size of the set
        fct (str, optional): Type of function to use for calculation.
                            If 'polynomial', returns n²+1, otherwise returns 2^n

    Returns:
        int: The cumulative number of combinations
    """
    if fct == "polynomial":
        return n**2 + 1

    return 2**n


def compute_self_waka_value_with_optional_target(
    all_data,
    all_labels,
    test_data,
    test_labels,
    K,
    target_model_loss=None,
    default_distance_metric="euclidean",
    max_quantile=1000,
    max_contributors=None,
    knn_model=None,
):
    """
    Compute unormalized self WAKA value with optional target model loss.

    Args:
        all_data (numpy.ndarray): Training data points
        all_labels (numpy.ndarray): Labels for the training data
        test_data (numpy.ndarray): Test data points to evaluate
        test_labels (numpy.ndarray): Labels for the test data
        K (int): Number of nearest neighbors to consider
        target_model_loss (float, optional): Target loss for the model
        default_distance_metric (str, optional): Distance metric to use, default is 'euclidean'
        max_quantile (int, optional): Maximum number of quantiles to consider
        max_contributors (int, optional): Maximum number of contributors to consider
        knn_model (object, optional): Pre-fitted KNN model

    Returns:
        list: Unormalized self WAKA values
    """
    N = len(all_labels)

    if knn_model is None:
        knn_model = KNeighborsClassifier(n_neighbors=K)
        knn_model.fit(all_data, all_labels)

    return compute_self_waka_value_with_optional_target_with_knn(
        all_labels,
        test_data,
        test_labels,
        K,
        target_model_loss,
        default_distance_metric,
        max_quantile,
        max_contributors,
        knn_model,
    )


def compute_self_waka_value_with_optional_target_with_knn(
    all_labels,
    test_data,
    test_labels,
    K,
    target_model_loss=None,
    default_distance_metric="euclidean",
    max_quantile=1000,
    max_contributors=None,
    knn_model_ref=None,
    nbr_of_considered_points=200,
    knn_model_copy=None,
    target_point_idx=None,
    all_data=None,
    rng=None,
):
    """
    Compute unormalized self WAKA value using a pre-fitted KNN model.

    Args:
        all_labels (numpy.ndarray): Labels for all data points
        test_data (numpy.ndarray): Test data points to evaluate
        test_labels (numpy.ndarray): Labels for the test data
        K (int): Number of nearest neighbors to consider
        target_model_loss (float, optional): Target loss for the model
        default_distance_metric (str, optional): Distance metric to use, default is 'euclidean'
        max_quantile (int, optional): Maximum number of quantiles to consider
        max_contributors (int, optional): Maximum number of contributors to consider
        knn_model_ref (object, optional): Reference to a KNN model in Ray
        nbr_of_considered_points (int, optional): Number of points to consider in the neighborhood
        knn_model_copy (object, optional): Copy of a KNN model to use
        target_point_idx (int, optional): Index of the target point
        all_data (numpy.ndarray, optional): All data points
        rng (numpy.random.Generator, optional): Random number generator

    Returns:
        list: Unormalized self WAKA values
    """
    # test_data and test_labels are arrays of shape (1, d) , thus they contain only one point

    if rng is None:
        rng = np.random.default_rng(42)

    # get the indices of the first 100 closest points to test_data
    if knn_model_ref is not None:
        knn_model = ray.get(knn_model_ref)
    else:
        knn_model = knn_model_copy

    fit_model = True

    if fit_model:
        # sorted_indices = knn_model.kneighbors(test_data, n_neighbors=nbr_of_considered_points+1, return_distance=False)[0]
        # Subsample test_data

        # remove the target point from all labels first
        all_labels_without_target = np.delete(all_labels, target_point_idx)
        all_data_without_target = np.delete(all_data, target_point_idx, axis=0)

        subsample_indices = np.random.choice(
            len(all_data_without_target), size=1000, replace=False
        )

        knn_model = NearestNeighbors(n_neighbors=K)
        knn_model.fit(all_data_without_target[subsample_indices])

    # Call kneighbors on the subsampled test_data
    subsampled_sorted_indices = knn_model.kneighbors(
        test_data, n_neighbors=nbr_of_considered_points, return_distance=False
    )

    if fit_model:
        # Map the subsampled indices back to the original test_data indices
        sorted_indices = subsample_indices[subsampled_sorted_indices[0]]

        sorted_labels = all_labels_without_target[sorted_indices]

    else:
        # Map the subsampled indices back to the original test_data indices
        sorted_indices = np.array(
            [subsample_indices[idx] for idx in subsampled_sorted_indices[0]]
        )
        # remove the target point from the sorted indices
        sorted_indices = sorted_indices[sorted_indices != target_point_idx]

        # get a random nbr_of_considered_points
        mask = np.zeros(len(sorted_indices), dtype=bool)

        mask[
            np.random.choice(
                len(sorted_indices), nbr_of_considered_points, replace=False
            )
        ] = True
        sorted_indices = sorted_indices[mask]


        sorted_labels = all_labels[sorted_indices]

    t_label = test_labels[0]


    all_losses = [i / K for i in range(0, K + 1)]

    # Initialize arrays to store the number of possible positive and negative votes per quantile
    nbr_possible_pos_votes = np.zeros(nbr_of_considered_points, dtype=int)
    nbr_possible_neg_votes = np.zeros(nbr_of_considered_points, dtype=int)

    # Iterate over each sorted index
    for j, sorted_label in enumerate(sorted_labels):
        # Update the number of possible positive and negative votes for the corresponding quantile
        nbr_possible_pos_votes[j] += sorted_label == t_label
        nbr_possible_neg_votes[j] += sorted_label != t_label

    nbr_possible_pos_votes = np.cumsum(nbr_possible_pos_votes)
    nbr_possible_neg_votes = np.cumsum(nbr_possible_neg_votes)

    aggregated_loss_contributions_with = [0 for _ in all_losses]
    aggregated_loss_contributions_without = [0 for _ in all_losses]

    possible_counts = {
        j: {i: int(comb(j, i)) for i in range(0, K)}
        for j in range(nbr_of_considered_points + 1)
    }

    for j, j_label in enumerate(sorted_labels):

        if j >= K - 1:
            count_j_at_kth = cummulative_nbr_of_combinations_for_set(
                j
            )  
            for l_i, loss in enumerate(all_losses):

                nbr_neg_votes = math.floor(loss * K)
                nbr_pos_votes = K - nbr_neg_votes

                if j_label != t_label:

                    if K > 0:
                        pos_votes_to_fill_with = nbr_pos_votes - 1
                        neg_votes_to_fill_with = nbr_neg_votes

                        pos_votes_to_fill_without = nbr_pos_votes
                        neg_votes_to_fill_without = nbr_neg_votes - 1

                        if (
                            pos_votes_to_fill_with < 0
                            or neg_votes_to_fill_with < 0
                            or nbr_possible_pos_votes[j] < pos_votes_to_fill_with
                            or nbr_possible_neg_votes[j] < neg_votes_to_fill_with
                        ):
                            nbr_of_models_with_loss_with = 0
                        else:
                            comb_pos_votes = possible_counts[nbr_possible_pos_votes[j]][
                                pos_votes_to_fill_with
                            ]
                            comb_neg_votes = possible_counts[nbr_possible_neg_votes[j]][
                                neg_votes_to_fill_with
                            ]
                            nbr_of_models_with_loss_with = (
                                comb_pos_votes * comb_neg_votes
                            )

                        if (
                            pos_votes_to_fill_without < 0
                            or neg_votes_to_fill_without < 0
                            or nbr_possible_pos_votes[j] < pos_votes_to_fill_without
                            or nbr_possible_neg_votes[j] < neg_votes_to_fill_without
                        ):
                            nbr_of_models_with_loss_without = 0
                        else:
                            comb_pos_votes = possible_counts[nbr_possible_pos_votes[j]][
                                pos_votes_to_fill_without
                            ]
                            comb_neg_votes = possible_counts[nbr_possible_neg_votes[j]][
                                neg_votes_to_fill_without
                            ]
                            nbr_of_models_with_loss_without = (
                                comb_pos_votes * comb_neg_votes
                            )

                    else:
                        if (
                            K == 1
                        ):  
                            nbr_of_models_with_loss_with = 1 if loss == 0 else 0
                            nbr_of_models_with_loss_without = 0 if loss == 0 else 1
                        else:
                            raise ValueError("K must be 1 or greater")


                    aggregated_loss_contributions_with[l_i] += (
                        nbr_of_models_with_loss_with / count_j_at_kth
                    )
                    aggregated_loss_contributions_without[l_i] += (
                        nbr_of_models_with_loss_without / count_j_at_kth
                    )

    signed_aggregated_deltas = [
        l_with - l_without
        for l_with, l_without in zip(
            aggregated_loss_contributions_with, aggregated_loss_contributions_without
        )
    ]

    reversed_signed_aggregated_deltas = signed_aggregated_deltas[::-1]

    cumm_reversed_signed_aggregated_deltas = cumsum(reversed_signed_aggregated_deltas)

    cumm_aggregated_deltas = [abs(d) for d in cumm_reversed_signed_aggregated_deltas]

    target_loss_int = math.floor(target_model_loss * K)
    idx = -target_loss_int - 1

    # Now correctly slice the reversed array according to the original logic
    target_cumm_aggregated_deltas1 = cumm_aggregated_deltas[:idx]

    target_cumm_aggregated_deltas2 = cumm_aggregated_deltas[idx:]

    wasserstein_aggregated_deltas = sum(target_cumm_aggregated_deltas1) - sum(
        target_cumm_aggregated_deltas2
    )

    target_score = wasserstein_aggregated_deltas


    return [target_score]


def compute_unormalized_average_waka_values(
    training_data,
    training_labels,
    test_data,
    test_labels,
    K,
    approx=None,
    ray_parallelization=None,
    default_distance_metric="euclidean",
    nbr_of_considered_points=100,
    knn_model_ref=None,
    strat="waka-strat-signed",
    Tau=None,
):
    """
    Compute unormalized average WAKA values for test data based on training data.

    Args:
        training_data (numpy.ndarray): Training data points
        training_labels (numpy.ndarray): Labels for the training data
        test_data (numpy.ndarray): Test data points to evaluate
        test_labels (numpy.ndarray): Labels for the test data
        K (int): Number of nearest neighbors to consider
        approx (any, optional): Approximation parameter (not used in this implementation)
        ray_parallelization (dict, optional): Parameters for ray parallelization
        default_distance_metric (str, optional): Distance metric to use, default is 'euclidean'
        nbr_of_considered_points (int, optional): Number of points to consider in the neighborhood
        knn_model_ref (object, optional): Reference to a KNN model
        strat (str, optional): Strategy for WAKA computation, default is "waka-strat-signed"
        Tau (float, optional): Parameter for penalty-based strategies

    Returns:
        list: Unormalized average WAKA values for each test point
    """

    N = len(training_data)

    training_labels = np.array(training_labels)
    test_labels = np.array(test_labels)

    # knn_model = ray.get(knn_model_ref)
    knn_model = NearestNeighbors(n_neighbors=min(K, N))
    knn_model.fit(training_data, training_labels)
    sorted_indices = knn_model.kneighbors(
        test_data, n_neighbors=min(nbr_of_considered_points, N), return_distance=False
    )

    # Get sorted labels using advanced indexing
    sorted_labels = training_labels[sorted_indices]

    all_losses = [i / K for i in range(0, K + 1)]

    test_waka_values = []

    possible_counts = {
        j: {i: int(comb(j, i)) for i in range(0, K)}
        for j in range(nbr_of_considered_points + 1)
    }

    # @time_it
    def t_iteration(t_label, t_sorted_labels, t_sorted_indices):
        """
        Calculate WAKA values for a single test point.

        Args:
            t_label (any): Label of the test point
            t_sorted_labels (numpy.ndarray): Sorted labels of the nearest neighbors
            t_sorted_indices (numpy.ndarray): Indices of the nearest neighbors

        Returns:
            numpy.ndarray: WAKA values for each training point
        """

        # Initialize arrays to store the number of possible positive and negative votes per quantile
        nbr_possible_pos_votes = np.zeros(nbr_of_considered_points, dtype=int)
        nbr_possible_neg_votes = np.zeros(nbr_of_considered_points, dtype=int)

        # Iterate over each sorted index
        for j, t_sorted_label in enumerate(t_sorted_labels):
            # Update the number of possible positive and negative votes for the corresponding quantile
            nbr_possible_pos_votes[j] += t_sorted_label == t_label
            nbr_possible_neg_votes[j] += t_sorted_label != t_label

        nbr_possible_pos_votes = np.cumsum(nbr_possible_pos_votes)
        nbr_possible_neg_votes = np.cumsum(nbr_possible_neg_votes)

        waka_values_ = []

        # here {i} can only be inside the regiion of nbr_of_considered_points
        for i, label_i in enumerate(t_sorted_labels[:-1]):

            aggregated_loss_contributions_with = [0 for _ in all_losses]
            aggregated_loss_contributions_without = [0 for _ in all_losses]

            # what is complicated to understand is that we only count contributions (swaps), this is because when you do the substraction
            # the number of models in coomon will cancel out. There are 2^n models but a majority contributes exactly the same to each value of the loss, only were {i} and {j}
            # can actually swap there can be a different contribution.

            # for j in contribution_indices:
            for j, j_label in enumerate(t_sorted_labels):

                if j >= K - 1 and j > i:

                    # count_j_at_kth = cummulative_nbr_of_combinations_for_set(nbr_of_considered_points - j)
                    count_j_at_kth = cummulative_nbr_of_combinations_for_set(j)

                    # for all loss value, we want to count models (onlye the differences with swaps) when {i} is in or out
                    for l_i, loss in enumerate(all_losses):

                        nbr_neg_votes = math.floor(loss * K)
                        nbr_pos_votes = K - nbr_neg_votes

                        if j_label != label_i:

                            pos_votes_to_fill_with = nbr_pos_votes - (
                                1 if label_i == t_label else 0
                            )
                            neg_votes_to_fill_with = nbr_neg_votes - (
                                1 if label_i != t_label else 0
                            )

                            if (
                                pos_votes_to_fill_with < 0
                                or neg_votes_to_fill_with < 0
                                or (loss == 0.0 and label_i != t_label)
                                or (loss == 1.0 and label_i == t_label)
                            ):
                                nbr_of_models_with_loss_with = 0
                            else:
                                if (
                                    nbr_possible_pos_votes[j] >= pos_votes_to_fill_with
                                    and nbr_possible_neg_votes[j]
                                    >= neg_votes_to_fill_with
                                ):

                                    comb_pos_votes = possible_counts[
                                        nbr_possible_pos_votes[j]
                                    ][pos_votes_to_fill_with]

                                    comb_neg_votes = possible_counts[
                                        nbr_possible_neg_votes[j]
                                    ][neg_votes_to_fill_with]
                                    nbr_of_models_with_loss_with = (
                                        comb_pos_votes * comb_neg_votes
                                    )
                                else:
                                    nbr_of_models_with_loss_with = 0

                            pos_votes_to_fill_without = nbr_pos_votes - (
                                1 if j_label == t_label else 0
                            )
                            neg_votes_to_fill_without = nbr_neg_votes - (
                                1 if j_label != t_label else 0
                            )

                            if (
                                pos_votes_to_fill_without < 0
                                or neg_votes_to_fill_without < 0
                                or (loss == 0.0 and j_label != t_label)
                                or (loss == 1.0 and j_label == t_label)
                            ):
                                nbr_of_models_with_loss_without = 0
                            else:
                                if (
                                    nbr_possible_pos_votes[j]
                                    >= pos_votes_to_fill_without
                                    and nbr_possible_neg_votes[j]
                                    >= neg_votes_to_fill_without
                                ):

                                    comb_pos_votes = possible_counts[
                                        nbr_possible_pos_votes[j]
                                    ][pos_votes_to_fill_without]

                                    comb_neg_votes = possible_counts[
                                        nbr_possible_neg_votes[j]
                                    ][neg_votes_to_fill_without]

                                    nbr_of_models_with_loss_without = (
                                        comb_pos_votes * comb_neg_votes
                                    )
                                else:
                                    nbr_of_models_with_loss_without = 0

                            aggregated_loss_contributions_with[l_i] += (
                                nbr_of_models_with_loss_with / count_j_at_kth
                            )
                            aggregated_loss_contributions_without[l_i] += (
                                nbr_of_models_with_loss_without / count_j_at_kth
                            )

            # FIXIT
            import warnings

            try:
                # Catch warnings as exceptions
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)

                    signed_aggregated_deltas = [
                        l_with - l_without
                        for l_with, l_without in zip(
                            aggregated_loss_contributions_with,
                            aggregated_loss_contributions_without,
                        )
                    ]

                    cumsumabs_after_split = False
                    # an experiment to look at the decision loss, default is the second one


                    pos_decision_idx = math.floor(0.5 * K)
                    if pos_decision_idx < 0.5 * K:
                        pos_decision_idx += 1

                    pos_decision_idx_adjust = (
                        1  # TODO we should test for all strat +1 and -1
                    )

                    # same label means it will be a positive vote
                    if label_i == t_label:
                        # Extract strategy type (removal or acquisition) and parameters
                        strat_type, strat_params = strat.split('-', 2)[1:]
                        
                        # Handle special case for full strategy
                        if strat_params == "full":
                            cumm_signed_aggregated_deltas = cumsum(signed_aggregated_deltas)
                            cumm_aggregated_deltas = [abs(l) for l in cumm_signed_aggregated_deltas]
                            wasserstein_aggregated = sum(cumm_aggregated_deltas)
                            
                        # Handle penalty-based strategies
                        elif strat_params == "removal-with-penalty":
                            if Tau == 1.0:
                                tau_idx = 0
                            else:
                                tau_idx = math.floor((1 - Tau) * K) + 1
                            cumm_signed_aggregated_deltas = cumsum(signed_aggregated_deltas[::-1])
                            cumm_aggregated_deltas = [abs(l) for l in cumm_signed_aggregated_deltas]
                            target_cumm_aggregated_deltas2 = cumm_aggregated_deltas[:tau_idx]
                            wasserstein_aggregated = sum(target_cumm_aggregated_deltas2)
                            
                        elif strat_params == "acquisition-with-penalty":
                            cumm_signed_aggregated_deltas = cumsum(signed_aggregated_deltas[::-1])
                            cumm_aggregated_deltas = [abs(l) for l in cumm_signed_aggregated_deltas]
                            wasserstein_aggregated = sum(cumm_aggregated_deltas)
                            
                        else:
                            raise ValueError(f"Unknown strategy number: {strat_num}")
                            

                    # different label means it will be a negative vote (multiply by -1)
                    else:
                        # Extract strategy type (removal or acquisition) and parameters
                        strat_type, strat_params = strat.split('-', 2)[1:]
                        
                        # Handle special case for full strategy
                        if strat_params == "full":
                            cumm_signed_aggregated_deltas = cumsum(signed_aggregated_deltas)
                            cumm_aggregated_deltas = [abs(l) for l in cumm_signed_aggregated_deltas]
                            wasserstein_aggregated = -sum(cumm_aggregated_deltas)
                     
                            
                        # Handle penalty-based strategies
                        elif strat_params == "removal-with-penalty":
                            cumm_signed_aggregated_deltas = cumsum(signed_aggregated_deltas)
                            cumm_aggregated_deltas = [abs(l) for l in cumm_signed_aggregated_deltas]
                            wasserstein_aggregated = -sum(cumm_aggregated_deltas)
                            
                        elif strat_params == "acquisition-with-penalty":
                            if Tau == 0.0:
                                tau_idx = 0
                            else:
                                tau_idx = math.floor(Tau * K) + 1
                            cumm_signed_aggregated_deltas = cumsum(signed_aggregated_deltas)
                            cumm_aggregated_deltas = [abs(l) for l in cumm_signed_aggregated_deltas]
                            target_cumm_aggregated_deltas2 = cumm_aggregated_deltas[:tau_idx]
                            wasserstein_aggregated = -sum(target_cumm_aggregated_deltas2)
                        else:
                            raise ValueError(f"Unknown strategy number: {strat_params}")
                
                        
         

            except RuntimeWarning as e:
                print(f"Warning caught: {e}")
                raise e
                # Handle the overflow here, e.g., by using a different approach
                # For demonstration, returning a placeholder value

            waka_values_.append(wasserstein_aggregated)

        # we put the waka values in the correct order
        waka_values = np.zeros(N)
        waka_values[t_sorted_indices[:-1]] = np.array(waka_values_)

        return waka_values

    zip_args = zip(test_labels, sorted_labels, sorted_indices)
    total_size = len(test_labels)

    if ray_parallelization is None:

        test_waka_values = [
            t_iteration(t_label, t_sorted_labels, t_sorted_indices)
            for i, (t_label, t_sorted_labels, t_sorted_indices) in enumerate(zip_args)
        ]

    else:
        from rayPlus import (
            parallel_loop,
            parallel_loop_lazy,
            parallel_loop_lazy_with_progress,
        )

        n_tasks = ray_parallelization[
            "n_tasks"
        ]  # max(total_size // 10 + 1, ray_parallelization["n_tasks"])
        # Warning! if return_results=True, the results will be stored in memory and it can be a problem if the results are too big, it will crash your computer
        test_waka_values = parallel_loop_lazy_with_progress(
            zip_args,
            total_size,
            t_iteration,
            return_results=True,
            n_tasks=n_tasks,
            object_store_memory=None,
        )  

    # waka_values_average = np.mean(test_waka_values, axis=0)
    return test_waka_values


def compute_self_unormalized_average_waka_values(
    training_data,
    training_labels,
    K,
    ray_parallelization=None,
    default_distance_metric="euclidean",
    nbr_of_considered_points=100,
    knn_model_ref=None,
):
    """
    Compute self unormalized average WAKA values for training data.

    Args:
        training_data (numpy.ndarray): Training data points
        training_labels (numpy.ndarray): Labels for the training data
        K (int): Number of nearest neighbors to consider
        ray_parallelization (dict, optional): Parameters for ray parallelization
        default_distance_metric (str, optional): Distance metric to use, default is 'euclidean'
        nbr_of_considered_points (int, optional): Number of points to consider in the neighborhood
        knn_model_ref (object, optional): Reference to a KNN model

    Returns:
        numpy.ndarray: Self unormalized average WAKA values for each training point
    """

    N = len(training_data)

    training_labels = np.array(training_labels)

    training_indices = np.arange(0, len(training_labels))

    # knn_model = ray.get(knn_model_ref)
    knn_model = NearestNeighbors(n_neighbors=K)
    knn_model.fit(training_data, training_labels)

    all_losses = [i / K for i in range(0, K + 1)]

    test_waka_values = []

    possible_counts = {
        j: {i: int(comb(j, i)) for i in range(0, K)}
        for j in range(nbr_of_considered_points + 1)
    }

    # @time_it
    def t_iteration(data_point, data_point_idx, data_point_label):
        """
        Calculate WAKA values for a single training point.

        Args:
            data_point (numpy.ndarray): A single training data point
            data_point_idx (int): Index of the data point
            data_point_label (any): Label of the data point

        Returns:
            tuple: Index and WAKA value for the data point
        """

        t_sorted_indices = knn_model.kneighbors(
            [data_point],
            n_neighbors=nbr_of_considered_points + 1,
            return_distance=False,
        )

        # remove data_point_idx from t_sorted_indices if it is in the list
        t_sorted_indices = [idx for idx in t_sorted_indices[0] if idx != data_point_idx]

        t_sorted_labels = training_labels[t_sorted_indices]

        # Initialize arrays to store the number of possible positive and negative votes per quantile
        nbr_possible_pos_votes = np.zeros(nbr_of_considered_points, dtype=int)
        nbr_possible_neg_votes = np.zeros(nbr_of_considered_points, dtype=int)

        # Iterate over each sorted index
        for j, t_sorted_label in enumerate(t_sorted_labels):
            # Update the number of possible positive and negative votes for the corresponding quantile
            nbr_possible_pos_votes[j] += t_sorted_label == data_point_label
            nbr_possible_neg_votes[j] += t_sorted_label != data_point_label

        nbr_possible_pos_votes = np.cumsum(nbr_possible_pos_votes)
        nbr_possible_neg_votes = np.cumsum(nbr_possible_neg_votes)

        label_i = data_point_label

        aggregated_loss_contributions_with = [0 for _ in all_losses]
        aggregated_loss_contributions_without = [0 for _ in all_losses]

        for j, j_label in enumerate(t_sorted_labels):

            if j >= K - 1:

                # count_j_at_kth = cummulative_nbr_of_combinations_for_set(nbr_of_considered_points - j)
                count_j_at_kth = cummulative_nbr_of_combinations_for_set(
                    j
                )  # TODO : IS IT BETTER THAN THE PREVIOUS ONE?

                # for all loss value, we want to count models (onlye the differences with swaps) when {i} is in or out
                for l_i, loss in enumerate(all_losses):

                    nbr_neg_votes = math.floor(loss * K)
                    nbr_pos_votes = K - nbr_neg_votes

                    if j_label != label_i:

                        pos_votes_to_fill_with = nbr_pos_votes - 1
                        neg_votes_to_fill_with = nbr_neg_votes

                        pos_votes_to_fill_without = nbr_pos_votes
                        neg_votes_to_fill_without = nbr_neg_votes - 1

                        if (
                            pos_votes_to_fill_with < 0
                            or neg_votes_to_fill_with < 0
                            or loss == 1.0
                        ):  # or pos_votes_to_fill_with > nbr_pos_votes or neg_votes_to_fill_with > nbr_neg_votes:
                            nbr_of_models_with_loss_with = 0
                        else:
                            if (
                                nbr_possible_pos_votes[j] >= pos_votes_to_fill_with
                                and nbr_possible_neg_votes[j] >= neg_votes_to_fill_with
                            ):

                                comb_pos_votes = possible_counts[
                                    nbr_possible_pos_votes[j]
                                ][pos_votes_to_fill_with]

                                comb_neg_votes = possible_counts[
                                    nbr_possible_neg_votes[j]
                                ][neg_votes_to_fill_with]
                                nbr_of_models_with_loss_with = (
                                    comb_pos_votes * comb_neg_votes
                                )
                            else:
                                nbr_of_models_with_loss_with = 0

                        if (
                            pos_votes_to_fill_without < 0
                            or neg_votes_to_fill_without < 0
                            or loss == 0.0
                        ):  # or pos_votes_to_fill_without > nbr_pos_votes or neg_votes_to_fill_without > nbr_neg_votes:
                            nbr_of_models_with_loss_without = 0
                        else:
                            if (
                                nbr_possible_pos_votes[j] >= pos_votes_to_fill_without
                                and nbr_possible_neg_votes[j]
                                >= neg_votes_to_fill_without
                            ):

                                comb_pos_votes = possible_counts[
                                    nbr_possible_pos_votes[j]
                                ][pos_votes_to_fill_without]

                                comb_neg_votes = possible_counts[
                                    nbr_possible_neg_votes[j]
                                ][neg_votes_to_fill_without]

                                nbr_of_models_with_loss_without = (
                                    comb_pos_votes * comb_neg_votes
                                )
                            else:
                                nbr_of_models_with_loss_without = 0

                        aggregated_loss_contributions_with[l_i] += (
                            nbr_of_models_with_loss_with / count_j_at_kth
                        )
                        aggregated_loss_contributions_without[l_i] += (
                            nbr_of_models_with_loss_without / count_j_at_kth
                        )

        import warnings

        try:
            # Catch warnings as exceptions
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)

                signed_aggregated_deltas = [
                    l_with - l_without
                    for l_with, l_without in zip(
                        aggregated_loss_contributions_with,
                        aggregated_loss_contributions_without,
                    )
                ]

                cumm_signed_aggregated_deltas = cumsum(signed_aggregated_deltas)

                cumm_aggregated_deltas = [abs(l) for l in cumm_signed_aggregated_deltas]
                # This operation might trigger a RuntimeWarning for overflow
                wasserstein_aggregated = sum(cumm_aggregated_deltas[::-1])

        except RuntimeWarning as e:
            print(f"Warning caught: {e}")
            raise e

        # we put the waka values in the correct order
        # waka_values = np.zeros(N)
        # waka_values[data_point_idx] = np.array(wasserstein_aggregated)

        return data_point_idx, wasserstein_aggregated

    zip_args = zip(training_data, training_indices, training_labels)
    total_size = len(training_indices)

    if ray_parallelization is None:

        test_waka_values = [
            t_iteration(data_point, data_point_idx, data_point_label)
            for i, (data_point, data_point_idx, data_point_label) in enumerate(zip_args)
        ]

    else:
        from rayPlus import (
            parallel_loop,
            parallel_loop_lazy,
            parallel_loop_lazy_with_progress,
        )

        n_tasks = ray_parallelization[
            "n_tasks"
        ]  # max(total_size // 10 + 1, ray_parallelization["n_tasks"])
        # Warning! if return_results=True, the results will be stored in memory and it can be a problem if the results are too big, it will crash your computer
        test_waka_values = parallel_loop_lazy_with_progress(
            zip_args,
            total_size,
            t_iteration,
            return_results=True,
            n_tasks=n_tasks,
            object_store_memory=None,
        )

    result = np.zeros(N)
    for idx, waka_value in test_waka_values:
        result[idx] = waka_value

    # waka_values_average = np.mean(test_waka_values, axis=0)
    return result


def compute_self_waka_contributions(
    t_sorted_labels,
    t_sorted_indices,
    possible_counts,
    K,
    label_i,
    nbr_possible_pos_votes,
    nbr_possible_neg_votes,
):
    """
    Compute self WAKA contributions for a set of data points.

    Args:
        t_sorted_labels (numpy.ndarray): Sorted labels of nearest neighbors
        t_sorted_indices (numpy.ndarray): Indices of nearest neighbors
        possible_counts (dict): Precomputed combinations
        K (int): Number of nearest neighbors to consider
        label_i (any): Label of the target point
        nbr_possible_pos_votes (numpy.ndarray): Cumulative count of possible positive votes
        nbr_possible_neg_votes (numpy.ndarray): Cumulative count of possible negative votes

    Returns:
        tuple: Two lists containing loss contributions with and without the target point
    """

    all_losses = [i / K for i in range(0, K + 1)]

    all_loss_contributions_with = [[] for _ in all_losses]
    all_loss_contributions_without = [[] for _ in all_losses]
    for j, j_label in enumerate(t_sorted_labels):

        if j >= K - 1:

            # count_j_at_kth = cummulative_nbr_of_combinations_for_set(nbr_of_considered_points - j)
            count_j_at_kth = (
                2**j
            )  # (j-(K-1)) # TODO : IS IT BETTER THAN THE PREVIOUS ONE?

            # for all loss value, we want to count models (onlye the differences with swaps) when {i} is in or out
            for l_i, loss in enumerate(all_losses):

                nbr_neg_votes = math.floor(loss * K)
                nbr_pos_votes = K - nbr_neg_votes

                if j_label != label_i:

                    # TODO : a better implem would be to start with fixing the point j that could be replaced by i, count it as we count K=1 for each loss but for any K. Then have a function
                    # that evaluate if there are points to fill or not.

                    if K > 1:
                        pos_votes_to_fill_with = nbr_pos_votes - 1
                        neg_votes_to_fill_with = nbr_neg_votes

                        pos_votes_to_fill_without = nbr_pos_votes
                        neg_votes_to_fill_without = nbr_neg_votes - 1

                        if (
                            pos_votes_to_fill_with < 0
                            or neg_votes_to_fill_with < 0
                            or nbr_possible_pos_votes[j] < pos_votes_to_fill_with
                            or nbr_possible_neg_votes[j] < neg_votes_to_fill_with
                            or loss == 1.0
                        ):
                            nbr_of_models_with_loss_with = 0
                        else:
                            comb_pos_votes = possible_counts[nbr_possible_pos_votes[j]][
                                pos_votes_to_fill_with
                            ]
                            comb_neg_votes = possible_counts[nbr_possible_neg_votes[j]][
                                neg_votes_to_fill_with
                            ]
                            nbr_of_models_with_loss_with = (
                                comb_pos_votes * comb_neg_votes
                            )

                        if (
                            pos_votes_to_fill_without < 0
                            or neg_votes_to_fill_without < 0
                            or nbr_possible_pos_votes[j] < pos_votes_to_fill_without
                            or nbr_possible_neg_votes[j] < neg_votes_to_fill_without
                            or loss == 0.0
                        ):
                            nbr_of_models_with_loss_without = 0
                        else:
                            comb_pos_votes = possible_counts[nbr_possible_pos_votes[j]][
                                pos_votes_to_fill_without
                            ]
                            comb_neg_votes = possible_counts[nbr_possible_neg_votes[j]][
                                neg_votes_to_fill_without
                            ]
                            nbr_of_models_with_loss_without = (
                                comb_pos_votes * comb_neg_votes
                            )

                    else:
                        if K == 1:
                            nbr_of_models_with_loss_with = 1 if loss == 0 else 0
                            nbr_of_models_with_loss_without = 0 if loss == 0 else 1
                        else:
                            raise ValueError("K must be 1 or greater")

                    all_loss_contributions_with[l_i].append(
                        (
                            t_sorted_indices[j],
                            nbr_of_models_with_loss_with / count_j_at_kth,
                        )
                    )
                    all_loss_contributions_without[l_i].append(
                        (
                            t_sorted_indices[j],
                            nbr_of_models_with_loss_without / count_j_at_kth,
                        )
                    )
                else:
                    all_loss_contributions_with[l_i].append((t_sorted_indices[j], 0))
                    all_loss_contributions_without[l_i].append((t_sorted_indices[j], 0))

    return all_loss_contributions_with, all_loss_contributions_without


def self_waka_t_iteration(
    training_labels,
    nbr_of_considered_points,
    knn_model,
    K,
    data_point,
    data_point_idx,
    data_point_label,
):
    """
    Perform a single iteration of self WAKA computation for a training point.

    Args:
        training_labels (numpy.ndarray): Labels for all training data points
        nbr_of_considered_points (int): Number of points to consider in the neighborhood
        knn_model (object): Fitted KNN model
        K (int): Number of nearest neighbors to consider
        data_point (numpy.ndarray): A single training data point
        data_point_idx (int): Index of the data point
        data_point_label (any): Label of the data point

    Returns:
        tuple: Two lists containing loss contributions with and without the target point
    """

    t_sorted_indices = knn_model.kneighbors(
        [data_point], n_neighbors=nbr_of_considered_points + 1, return_distance=False
    )

    # remove data_point_idx from t_sorted_indices if it is in the list
    t_sorted_indices = [idx for idx in t_sorted_indices[0] if idx != data_point_idx]

    t_sorted_labels = training_labels[t_sorted_indices]

    # Initialize arrays to store the number of possible positive and negative votes per quantile
    nbr_possible_pos_votes = np.zeros(nbr_of_considered_points, dtype=int)
    nbr_possible_neg_votes = np.zeros(nbr_of_considered_points, dtype=int)

    # Iterate over each sorted index
    for j, t_sorted_label in enumerate(t_sorted_labels):
        # Update the number of possible positive and negative votes for the corresponding quantile
        nbr_possible_pos_votes[j] += t_sorted_label == data_point_label
        nbr_possible_neg_votes[j] += t_sorted_label != data_point_label

    nbr_possible_pos_votes = np.cumsum(nbr_possible_pos_votes)
    nbr_possible_neg_votes = np.cumsum(nbr_possible_neg_votes)

    label_i = data_point_label

    # possible_counts = precompute_combinations(nbr_of_considered_points, K)
    possible_counts = {
        j: {i: int(comb(j, i)) for i in range(0, K)}
        for j in range(nbr_of_considered_points + 1)
    }
    return compute_self_waka_contributions(
        t_sorted_labels,
        t_sorted_indices,
        possible_counts,
        K,
        label_i,
        nbr_possible_pos_votes,
        nbr_possible_neg_votes,
    )


def self_waka_aggregate(all_loss_contributions_with, all_loss_contributions_without):
    """
    Aggregate self WAKA contributions and calculate Wasserstein influences.

    Args:
        all_loss_contributions_with (list): Loss contributions with the target point
        all_loss_contributions_without (list): Loss contributions without the target point

    Returns:
        tuple: Wasserstein aggregated value and a list of Wasserstein influences
    """
    import warnings

    try:
        # Catch warnings as exceptions
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            aggregated_loss_contributions_with = np.zeros(
                len(all_loss_contributions_with)
            )
            aggregated_loss_contributions_without = np.zeros(
                len(all_loss_contributions_without)
            )

            for l_i, _ in enumerate(all_loss_contributions_with):
                aggregated_loss_contributions_with[l_i] = sum(
                    [l[1] for l in all_loss_contributions_with[l_i]]
                )
                aggregated_loss_contributions_without[l_i] = sum(
                    [l[1] for l in all_loss_contributions_without[l_i]]
                )

            signed_aggregated_deltas = [
                l_with - l_without
                for l_with, l_without in zip(
                    aggregated_loss_contributions_with,
                    aggregated_loss_contributions_without,
                )
            ]
            cumm_signed_aggregated_deltas = np.cumsum(signed_aggregated_deltas)
            cumm_aggregated_deltas = [abs(l) for l in cumm_signed_aggregated_deltas]
            # This operation might trigger a RuntimeWarning for overflow
            wasserstein_aggregated = sum(cumm_aggregated_deltas[::-1])

            # Compute wassertein_influences
            wassertein_influences = []

            for j in range(len(all_loss_contributions_with[0])):

                aggregated_loss_contributions_with_tmp = np.zeros(
                    len(all_loss_contributions_with)
                )
                aggregated_loss_contributions_without_tmp = np.zeros(
                    len(all_loss_contributions_without)
                )

                for l_i in range(len(aggregated_loss_contributions_with)):
                    # remove element at j indix in all_loss_contributions_with[l] and split it in 2 arrays
                    pre_with = sum([l[1] for l in all_loss_contributions_with[l_i][:j]])
                    post_with = sum(
                        [l[1] * 2 for l in all_loss_contributions_with[l_i][j + 1 :]]
                    )
                    aggregated_loss_contributions_with_tmp[l_i] = pre_with + post_with

                    pre_without = sum(
                        [l[1] for l in all_loss_contributions_without[l_i][:j]]
                    )
                    post_without = sum(
                        [l[1] * 2 for l in all_loss_contributions_without[l_i][j + 1 :]]
                    )
                    aggregated_loss_contributions_without_tmp[l_i] = (
                        pre_without + post_without
                    )

                # Recalculate signed aggregated deltas
                temp_signed_aggregated_deltas = [
                    l_with - l_without
                    for l_with, l_without in zip(
                        aggregated_loss_contributions_with_tmp,
                        aggregated_loss_contributions_without_tmp,
                    )
                ]
                temp_cumm_signed_aggregated_deltas = np.cumsum(
                    temp_signed_aggregated_deltas
                )
                temp_cumm_aggregated_deltas = [
                    abs(l) for l in temp_cumm_signed_aggregated_deltas
                ]

                # Calculate the wasserstein influence for this exclusion
                wassertein_influence = sum(temp_cumm_aggregated_deltas[::-1])

                # Append the pair (index, influence)
                wassertein_influences.append(
                    (
                        all_loss_contributions_with[0][j][0],
                        wassertein_influence - wasserstein_aggregated,
                    )
                )

    except RuntimeWarning as e:
        print(f"Warning caught: {e}")
        raise e

    return wasserstein_aggregated, wassertein_influences


def compute_self_unormalized_average_waka_values_recomputable(
    training_data,
    training_labels,
    K,
    ray_parallelization=None,
    nbr_of_considered_points=100,
):
    """
    Compute self unormalized average WAKA values with the ability to recompute intermediate values.

    This function computes self WAKA values in a way that allows for recomputation of
    intermediate values, which can be useful for more detailed analysis.

    Args:
        training_data (numpy.ndarray): Training data points
        training_labels (numpy.ndarray): Labels for the training data
        K (int): Number of nearest neighbors to consider
        ray_parallelization (dict, optional): Parameters for ray parallelization
        nbr_of_considered_points (int, optional): Number of points to consider in the neighborhood

    Returns:
        list: List of tuples containing Wasserstein aggregated values and influence lists
             for each training point
    """

    N = len(training_data)

    training_labels = np.array(training_labels)

    training_indices = np.arange(0, len(training_labels))

    # knn_model = ray.get(knn_model_ref)
    knn_model = NearestNeighbors(n_neighbors=K)
    knn_model.fit(training_data, training_labels)

    test_waka_values = []

    zip_args = zip(training_data, training_indices, training_labels)
    total_size = len(training_indices)

    def t_iteration(data_point, data_point_idx, data_point_label):
        """
        Process a single training point for WAKA computation.

        Args:
            data_point (numpy.ndarray): A single training data point
            data_point_idx (int): Index of the data point
            data_point_label (any): Label of the data point

        Returns:
            tuple: Loss contributions with/without the target point and point index
        """

        all_loss_contributions_with, all_loss_contributions_without = (
            self_waka_t_iteration(
                training_labels,
                nbr_of_considered_points,
                knn_model,
                K,
                data_point,
                data_point_idx,
                data_point_label,
            )
        )

        return (
            all_loss_contributions_with,
            all_loss_contributions_without,
            data_point_idx,
        )

    if ray_parallelization is None:

        test_waka_values_decomposed = [
            t_iteration(data_point, data_point_idx, data_point_label)
            for i, (data_point, data_point_idx, data_point_label) in enumerate(zip_args)
        ]

    else:
        from rayPlus import (
            parallel_loop,
            parallel_loop_lazy,
            parallel_loop_lazy_with_progress,
        )

        n_tasks = ray_parallelization[
            "n_tasks"
        ]  # max(total_size // 10 + 1, ray_parallelization["n_tasks"])
        # Warning! if return_results=True, the results will be stored in memory and it can be a problem if the results are too big, it will crash your computer
        test_waka_values_decomposed = parallel_loop_lazy_with_progress(
            zip_args,
            total_size,
            t_iteration,
            return_results=True,
            n_tasks=n_tasks,
            object_store_memory=None,
        )

    test_waka_values = [
        (
            data_point_idx,
            self_waka_aggregate(
                all_loss_contributions_with, all_loss_contributions_without
            ),
        )
        for all_loss_contributions_with, all_loss_contributions_without, data_point_idx in test_waka_values_decomposed
    ]

    result = [None] * N
    for idx, waka_value in test_waka_values:
        result[idx] = waka_value

    # waka_values_average = np.mean(test_waka_values, axis=0)
    return result
