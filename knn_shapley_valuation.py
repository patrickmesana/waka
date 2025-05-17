import numpy as np
from sklearn.neighbors import NearestNeighbors
from numba import njit, prange

from utils import order_points_by_distance, parallel_order_points_by_distance
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from tqdm import tqdm
from rayPlus import parallel_loop


@dataclass
class UtilityResult:
    score: float
    model: any = None
    report: any = None
    data: any = None


def compute_knn_shapley_values(K, y_test, y, N, inverse_contrib=False):
    """
    Compute the Shapley values for k-NN using a loop.

    Parameters:
    -----------
    K : int
        The 'K' in k-NN, the number of neighbors to consider.
    y_test : any
        The test label.
    y : array-like
        An array of labels corresponding to the sorted distances to x_test.
    N : int
        The total number of elements in the dataset.
    inverse_contrib : bool, default=False
        If True, inverts the contribution calculation.

    Returns:
    --------
    list
        An array containing all Shapley values.
    """
    s = [0] * N  # Array to store the Shapley values

    # Initialize the base case (i = N)
    s[N - 1] = int(y[N - 1] == y_test) / N

    # Loop from N-1 down to 1 to populate the rest of s
    for i in range(N - 1, 0, -1):
        alpha_i = i - 1
        if inverse_contrib:
            contrib = int(y[alpha_i + 1] == y_test) - int(y[alpha_i] == y_test)
        else:
            contrib = int(y[alpha_i] == y_test) - int(y[alpha_i + 1] == y_test)
        s[alpha_i] = s[alpha_i + 1] + contrib / K * min(K, i) / i

    return s


def compute_leave_one_out(K, test_label, sorted_labels):
    """
    Compute leave-one-out values for k-NN.

    Parameters:
    -----------
    K : int
        The 'K' in k-NN, the number of neighbors to consider.
    test_label : any
        The label of the test point.
    sorted_labels : array-like
        Labels of training points sorted by distance to the test point.

    Returns:
    --------
    numpy.ndarray
        An array containing all leave-one-out values.
    """
    leave_one_out_values = np.zeros(len(sorted_labels))

    for i in range(len(sorted_labels)):
        if i > K - 1:
            # If i > K-1, set leave one out value to 0
            leave_one_out_values[i] = 0
        else:
            # Look at the label of the K-th point
            k_th_label = sorted_labels[K - 1]
            if sorted_labels[i] == k_th_label:
                # If labels are the same
                leave_one_out_values[i] = 0
            else:
                # Otherwise, set leave one out value to 1/K or -1/K
                leave_one_out_values[i] = (
                    1 / K if sorted_labels[i] == test_label else -1 / K
                )

    return leave_one_out_values


@njit(parallel=True)
def parallel_compute_training_average_shapley_values(
    training_data,
    training_labels,
    test_data,
    test_labels,
    K,
    only_average=False,
    inverse_contrib=False,
    sorted_indices=None,
):
    """
    Compute average Shapley values in parallel for multiple test points.

    Parameters:
    -----------
    training_data : array-like
        Training data points.
    training_labels : array-like
        Labels of training data points.
    test_data : array-like
        Test data points.
    test_labels : array-like
        Labels of test data points.
    K : int
        The 'K' in k-NN, the number of neighbors to consider.
    only_average : bool, default=False
        If True, returns only the average Shapley values.
    inverse_contrib : bool, default=False
        If True, inverts the contribution calculation.
    sorted_indices : array-like, default=None
        Pre-computed sorted indices by distance.

    Returns:
    --------
    numpy.ndarray
        Array of Shapley values for each test point or their average.
    """
    N = len(training_data)

    if sorted_indices is None:
        sorted_indices = parallel_order_points_by_distance(training_data, test_data)

    # Create array with |test_labels| rows and |training_labels| columns
    sorted_training_abels = training_labels[sorted_indices]

    tests_shapley_values = []

    for i in prange(len(test_labels)):
        ii_shapley_values = compute_knn_shapley_values(
            K, test_labels[i], sorted_training_abels[i], N, inverse_contrib
        )
        shapley_values = np.zeros_like(ii_shapley_values)
        shapley_values[sorted_indices[i]] = ii_shapley_values

        tests_shapley_values.append(shapley_values)

    if only_average:
        return np.mean(tests_shapley_values, axis=0)
    else:
        return np.array(tests_shapley_values)


def compute_training_average_shapley_values(
    training_data,
    training_labels,
    test_data,
    test_labels,
    K,
    only_average=False,
    inverse_contrib=False,
):
    """
    Compute average Shapley values for multiple test points.

    Parameters:
    -----------
    training_data : array-like
        Training data points.
    training_labels : array-like
        Labels of training data points.
    test_data : array-like
        Test data points.
    test_labels : array-like
        Labels of test data points.
    K : int
        The 'K' in k-NN, the number of neighbors to consider.
    only_average : bool, default=False
        If True, returns only the average Shapley values.
    inverse_contrib : bool, default=False
        If True, inverts the contribution calculation.

    Returns:
    --------
    numpy.ndarray
        Array of Shapley values for each test point or their average.
    """
    N = len(training_data)
    sorted_indices = order_points_by_distance(training_data, test_data)

    # Create array with |test_labels| rows and |training_labels| columns
    sorted_training_abels = training_labels[sorted_indices]

    tests_shapley_values = []
    for i, _ in enumerate(test_labels):
        ii_shapley_values = compute_knn_shapley_values(
            K, test_labels[i], sorted_training_abels[i], N, inverse_contrib
        )
        shapley_values = np.zeros_like(ii_shapley_values)
        shapley_values[sorted_indices[i]] = ii_shapley_values

        tests_shapley_values.append(shapley_values)

    if only_average:
        return np.mean(tests_shapley_values, axis=0)
    else:
        return np.array(tests_shapley_values)


def compute_training_average_leave_one_out(
    training_data, training_labels, test_data, test_labels, K, only_average=False
):
    """
    Compute average leave-one-out values for multiple test points.

    Parameters:
    -----------
    training_data : array-like
        Training data points.
    training_labels : array-like
        Labels of training data points.
    test_data : array-like
        Test data points.
    test_labels : array-like
        Labels of test data points.
    K : int
        The 'K' in k-NN, the number of neighbors to consider.
    only_average : bool, default=False
        If True, returns only the average leave-one-out values.

    Returns:
    --------
    numpy.ndarray
        Array of leave-one-out values for each test point or their average.
    """
    N = len(training_data)
    sorted_indices = order_points_by_distance(training_data, test_data)

    # Create array with |test_labels| rows and |training_labels| columns
    sorted_training_abels = training_labels[sorted_indices]

    tests_loo = []
    for i, _ in enumerate(test_labels):
        ii_shapley_values = compute_leave_one_out(
            K, test_labels[i], sorted_training_abels[i]
        )
        loos = np.zeros_like(ii_shapley_values)
        loos[sorted_indices[i]] = ii_shapley_values

        tests_loo.append(loos)

    if only_average:
        return np.mean(tests_loo, axis=0)
    else:
        return np.array(tests_loo)


def compute_self_training_average_shapley_values(
    training_data, training_labels, K, inverse_contrib=False
):
    """
    Compute average Shapley values using training data as both training and test set.

    Parameters:
    -----------
    training_data : array-like
        Training data points.
    training_labels : array-like
        Labels of training data points.
    K : int
        The 'K' in k-NN, the number of neighbors to consider.
    inverse_contrib : bool, default=False
        If True, inverts the contribution calculation.

    Returns:
    --------
    numpy.ndarray
        Array of self-training Shapley values.
    """
    N = len(training_data)
    sorted_indices = order_points_by_distance(training_data, training_data)

    # Create array with |training_labels| rows and |training_labels| columns
    sorted_training_abels = training_labels[sorted_indices]

    tests_shapley_values = []
    for i, _ in enumerate(training_labels):
        ii_shapley_values = compute_knn_shapley_values(
            K, training_labels[i], sorted_training_abels[i], N, inverse_contrib
        )
        tests_shapley_values.append(ii_shapley_values[0])

    return np.array(tests_shapley_values)


def compute_self_training_average_leave_one_out(training_data, training_labels, K):
    """
    Compute average leave-one-out values using training data as both training and test set.

    Parameters:
    -----------
    training_data : array-like
        Training data points.
    training_labels : array-like
        Labels of training data points.
    K : int
        The 'K' in k-NN, the number of neighbors to consider.

    Returns:
    --------
    numpy.ndarray
        Array of self-training leave-one-out values.
    """
    N = len(training_data)
    sorted_indices = order_points_by_distance(training_data, training_data)

    # Create array with |training_labels| rows and |training_labels| columns
    sorted_training_abels = training_labels[sorted_indices]

    self_loos = []
    for i, _ in enumerate(training_labels):
        ii_loos = compute_leave_one_out(K, training_labels[i], sorted_training_abels[i])
        self_loos.append(ii_loos[0])

    return np.array(self_loos)


def compute_knn_shapley_values__pydvl__(n_neighbors, y, yt, n, j, ii):
    """
    Compute Shapley values for k-NN using PyDVL algorithm with online averaging.

    Parameters:
    -----------
    n_neighbors : int
        The 'k' in k-NN, the number of neighbors to consider.
    y : any
        Label of test point.
    yt : array-like
        Labels of training points.
    n : int
        The total number of elements in the dataset.
    j : int
        The current iteration count for averaging.
    ii : array-like
        Indices of training points sorted by distance.

    Returns:
    --------
    numpy.ndarray
        Array of Shapley values.
    """
    values = np.zeros_like(yt, dtype=np.float_)

    value_at_x = int(yt[ii[-1]] == y) / n
    values[ii[-1]] += (value_at_x - values[ii[-1]]) / j

    # Farthest to closest, starting from n-2
    for i in range(n - 2, n_neighbors, -1):
        value_at_x = (
            values[ii[i + 1]] + (int(yt[ii[i]] == y) - int(yt[ii[i + 1]] == y)) / i
        )
        values[ii[i]] += (value_at_x - values[ii[i]]) / j

    # Farthest to closest for the nearest neighbors
    for i in range(n_neighbors, -1, -1):
        value_at_x = (
            values[ii[i + 1]]
            + (int(yt[ii[i]] == y) - int(yt[ii[i + 1]] == y)) / n_neighbors
        )
        values[ii[i]] += (value_at_x - values[ii[i]]) / j

    return values


def compute_knn_shapley_values_not_averaging__pydvl__(n_neighbors, y, yt, n, j, ii):
    """
    Compute Shapley values for k-NN using PyDVL algorithm without online averaging.

    Parameters:
    -----------
    n_neighbors : int
        The 'k' in k-NN, the number of neighbors to consider.
    y : any
        Label of test point.
    yt : array-like
        Labels of training points.
    n : int
        The total number of elements in the dataset.
    j : int
        The current iteration (unused in this version).
    ii : array-like
        Indices of training points sorted by distance.

    Returns:
    --------
    numpy.ndarray
        Array of Shapley values.
    """
    values = np.zeros_like(yt, dtype=np.float_)

    value_at_x = int(yt[ii[-1]] == y) / n
    values[ii[-1]] = value_at_x

    # Farthest to closest, starts at n-2
    for i in range(n - 2, n_neighbors, -1):
        value_at_x = (
            values[ii[i + 1]] + (int(yt[ii[i]] == y) - int(yt[ii[i + 1]] == y)) / i
        )
        values[ii[i]] = value_at_x

    # Farthest to closest for the nearest neighbors
    for i in range(n_neighbors, -1, -1):
        value_at_x = (
            values[ii[i + 1]]
            + (int(yt[ii[i]] == y) - int(yt[ii[i + 1]] == y)) / n_neighbors
        )
        values[ii[i]] = value_at_x

    return values


def knn_valuation(
    utility_fct,
    training_inputs,
    training_targets,
    valuation_inputs,
    valuation_targets,
    progress=True,
    seed=42,
    K=5,
):
    """
    Compute Shapley values for k-NN using the utility function approach.

    Parameters:
    -----------
    utility_fct : callable
        The utility function to use for valuation (unused in this implementation).
    training_inputs : array-like
        Training data points.
    training_targets : array-like
        Labels of training data points.
    valuation_inputs : array-like
        Data points for valuation.
    valuation_targets : array-like
        Labels of valuation data points.
    progress : bool, default=True
        If True, shows progress (unused in this implementation).
    seed : int, default=42
        Random seed for reproducibility (unused in this implementation).
    K : int, default=5
        The 'K' in k-NN, the number of neighbors to consider.

    Returns:
    --------
    numpy.ndarray
        Array of Shapley values.
    """
    nns = NearestNeighbors(n_neighbors=len(training_inputs), algorithm="ball_tree").fit(
        training_inputs
    )
    # Get indices of neighbors from closest to farthest
    _, indices = nns.kneighbors(valuation_inputs)

    N = len(training_inputs)
    t_shapley_values = []

    for j, (y_test, ii) in enumerate(zip(valuation_targets, indices)):
        y = training_targets[ii]
        ii_shapley_values = compute_knn_shapley_values(K, y_test, y, N)
        # Reorder values to match original indices
        shapley_values = np.zeros_like(ii_shapley_values)
        shapley_values[ii] = ii_shapley_values
        t_shapley_values.append(shapley_values)

    return np.array(t_shapley_values)
