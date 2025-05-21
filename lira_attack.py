from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

from itertools import combinations
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import log_loss


# def logit_transformation(probability):
#     """
#     Apply the logit transformation to a probability.

#     Parameters:
#     - probability: A float representing a probability (between 0 and 1).

#     Returns:
#     - float: The logit transformation of the probability.
#     """
#     # Check to avoid division by zero or taking log of zero
#     if probability <= 0:
#         return -np.inf  # or some large negative value
#     elif probability >= 1:
#         return np.inf   # or some large positive value
#     else:
#         return np.log(probability / (1 - probability))

def train_shadow_models_and_collect_scores(all_data, all_labels, data_point, data_point_label, N_shadow, K, seed=42, shadow_dataset_proportion=[0.9, 1], regularization_term=1e-6, ray_parallelization = None, default_distance_metric = 'euclidean'):
    np.random.seed(seed)  # Fix the seed for reproducibility


    dataset_size = len(all_data)

    distances = cdist(all_data, np.array([data_point]), default_distance_metric)
    
    # Find the index of the closest point
    closest_point_index = np.argmin(distances)
    
    # create new dataset without the closest point
    modified_data = np.copy(all_data)
    modified_labels = np.copy(all_labels)
    modified_data = np.delete(modified_data, closest_point_index, axis=0)
    modified_labels = np.delete(modified_labels, closest_point_index, axis=0)


    def shadow_iteration(shadow_i):
        # Randomly select a proportion within the given interval
        proportion = np.random.uniform(shadow_dataset_proportion[0], shadow_dataset_proportion[1])
        sample_size = int(dataset_size * proportion)  # Calculate sample size based on the selected proportion

        # Randomly sample a shadow dataset of the calculated size
        shadow_data, shadow_labels = resample(modified_data, modified_labels, n_samples=sample_size, replace=False) # FIXME replace=True


        # Train IN model
        in_data = np.vstack((shadow_data, data_point))  # Append data_point to shadow_data
        in_labels = np.append(shadow_labels, data_point_label) 
        model_in = KNeighborsClassifier(n_neighbors=K)
        model_in.fit(in_data, in_labels)

        
        # Find the K nearest neighbors of the first example in X_test (for demonstration)
        _, in_indices = model_in.kneighbors(np.array([data_point]), n_neighbors=K)

        # Use the indices to look up the labels of the nearest neighbors
        in_neighbor_labels =  in_labels[in_indices] #TODO :check if indices are not relative

        loss_in = 1- np.sum(in_neighbor_labels == data_point_label) / K

        # Train OUT model
        model_out = KNeighborsClassifier(n_neighbors=K)
        model_out.fit(shadow_data, shadow_labels)

        # out_label_pos = data_point_label if np.unique(shadow_labels).shape[0] > 1 else 0


        # Find the K nearest neighbors of the first example in X_test (for demonstration)
        _, out_indices = model_out.kneighbors(np.array([data_point]), n_neighbors=K)

        # Use the indices to look up the labels of the nearest neighbors
        out_neighbor_labels =  shadow_labels[out_indices]

        loss_out = 1- np.sum(out_neighbor_labels == data_point_label) / K    

        return loss_in, loss_out
    

    if ray_parallelization is None:
        losses_in_and_out = [shadow_iteration(i) for i in range(N_shadow)]
    
    else:
        from rayPlus import parallel_loop


        # Warning! if return_results=True, the results will be stored in memory and it can be a problem if the results are too big, it will crash your computer
        losses_in_and_out = parallel_loop(list(range(N_shadow)), 
                                shadow_iteration, 
                                return_results=True, 
                                n_tasks=ray_parallelization["n_tasks"], 
                                init_and_shutdown_ray=ray_parallelization["init_and_shutdown_ray"]
                            )


    losses_in = [loss_in for loss_in, loss_out in losses_in_and_out]
    losses_out = [loss_out for loss_in, loss_out in losses_in_and_out]
    return losses_in, losses_out



def logit_transformation(x):
    return np.log(x / (1 - x))


import warnings
def compute_likelihood_ratio_reg(confs_in, confs_out, conf_obs, regularization_term=1e-6, transform=False):
    # Calculate mean and regularized variance
    mu_in = np.mean([logit_transformation(c) if transform else c for c in confs_in])
    sigma_in = np.var(confs_in) + regularization_term
    mu_out = np.mean([logit_transformation(c) if transform else c for c in confs_out])
    sigma_out = np.var(confs_out) + regularization_term

    std_in = np.sqrt(sigma_in)
    std_out = np.sqrt(sigma_out)

    # Compute likelihoods
    likelihood_in = norm.pdf(logit_transformation(conf_obs) if transform else conf_obs, mu_in, std_in) + regularization_term
    likelihood_out = norm.pdf(logit_transformation(conf_obs) if transform else conf_obs, mu_out, std_out) + regularization_term

    with warnings.catch_warnings(record=True) as w:
        # Force the RuntimeWarning to trigger in this block
        warnings.simplefilter("always", RuntimeWarning)
        
        likelihood_ratio = likelihood_in / likelihood_out

        # If any warnings were raised, check them
        if w:
            for warning in w:
                if issubclass(warning.category, RuntimeWarning):
                    raise warning.message

    return likelihood_ratio


def compute_knn_distributions_uniform(all_data, all_labels, data_point, data_point_label, K=1):
    # Calculate distances from the data point to all points in all_data
    distances = np.linalg.norm(all_data - data_point, axis=1)
    
    # Find the index of the closest point
    closest_point_index = np.argmin(distances)
    
    # Replace the closest point with the data point in a new dataset
    modified_data = np.copy(all_data)
    modified_labels = np.copy(all_labels)
    modified_data[closest_point_index] = data_point
    modified_labels[closest_point_index] = data_point_label  # Ensure label consistency

    N = len(modified_data)  # Update N to reflect the modified dataset size
    
    losses_in = []
    losses_out = []
    
    for combo_indices in combinations(range(N), K):
        combo_labels = modified_labels[list(combo_indices)]
        
        # Calculate the frequency of each label
        # label_counts = np.bincount(combo_labels, minlength=np.max(modified_labels)+1)
        # most_frequent_label = np.argmax(label_counts)
        # probability_most_likely_label = label_counts[most_frequent_label] / K
        loss = np.sum(combo_labels == data_point_label) / K
        
        # Check if the combination includes the replaced point
        if closest_point_index in combo_indices:
            losses_in.append(loss)
        else:
            losses_out.append(loss)
    
    return losses_in, losses_out


def compute_knn_distributions_localsearch(all_data, all_labels, data_point, data_point_label, K=1, close_points_percent=0.05):

    # Calculate distances from the data point to all points in all_data
    distances = np.linalg.norm(all_data - data_point, axis=1)

    nbr_to_consider = max(len(distances) * close_points_percent, 2*K)

    # Find the close_points_percent closest points
    close_points_indices = np.argsort(distances)[:int(nbr_to_consider)]
    closest_point_index = close_points_indices[0]
    
    # Replace the closest point with the data point in a new dataset
    modified_data = np.copy(all_data)
    modified_labels = np.copy(all_labels)
    modified_data[closest_point_index] = data_point
    modified_labels[closest_point_index] = data_point_label  # Ensure label consistency

    N = len(modified_data)  # Update N to reflect the modified dataset size

    # indices_combinations should be taken out of close_points_indices
    indices_combinations = combinations(close_points_indices, K)
    
    losses_in = []
    losses_out = []
    
    for combo_indices in indices_combinations:
        combo_labels = modified_labels[list(combo_indices)]
        
        # Calculate the frequency of each label
        # label_counts = np.bincount(combo_labels, minlength=np.max(modified_labels)+1)
        # most_frequent_label = np.argmax(label_counts)
        # probability_most_likely_label = label_counts[most_frequent_label] / K
        loss = np.sum(combo_labels == data_point_label) / K
        
        # Check if the combination includes the replaced point
        if closest_point_index in combo_indices:
            losses_in.append(loss)
        else:
            losses_out.append(loss)
    
    return losses_in, losses_out

def lira_online_attack_reg(all_data, all_labels, target_model_loss, data_point, data_point_label, N_shadow=10, K=3, shadow_dataset_proportion=0.7, regularization_term=1e-6, ray_parallelization = None, default_distance_metric = 'euclidean'):
    # Train shadow models and collect confidence scores
    losses_in, losses_out = train_shadow_models_and_collect_scores(all_data, all_labels, data_point, data_point_label, N_shadow, K, shadow_dataset_proportion=shadow_dataset_proportion, regularization_term=regularization_term, ray_parallelization=ray_parallelization, default_distance_metric=default_distance_metric)



    # Compute likelihood ratio with regularization
    likelihood_ratio = compute_likelihood_ratio_reg(losses_in, losses_out, target_model_loss, regularization_term=regularization_term)
    return likelihood_ratio



def lira_vanilla_attack(all_data, all_labels, target_model_loss, data_point, data_point_label,challenge_point_idx, shadows, K=3, seed_list=[], ray_parallelization = None, default_distance_metric = 'euclidean'):

    losses_in = []
    losses_out = []

    sample_size = int(len(all_data)) // 2

    for i, models in enumerate(shadows):

        model1, model2 = models

        shadow_seed = seed_list[i]

        rng = np.random.RandomState(shadow_seed)   

        shadow_indices = rng.choice(len(all_data), sample_size, replace=True)

        non_shadow_indices = np.array([i for i in range(len(all_data)) if i not in shadow_indices])

       
        _, indices1 = model1.kneighbors(np.array([data_point]), n_neighbors=K)


        labels1 =  all_labels[shadow_indices[indices1]]

        loss1 = 1- np.sum(labels1 == data_point_label) / K

        _, indices2 = model2.kneighbors(np.array([data_point]), n_neighbors=K)


        labels2 =  all_labels[non_shadow_indices[indices2]]

        loss2 = 1- np.sum(labels2 == data_point_label) / K



        if challenge_point_idx in shadow_indices:
            losses_in.append(loss1)
            losses_out.append(loss2)
        else:

            losses_in.append(loss2)
            losses_out.append(loss1)



    # Compute likelihood ratio with regularization
    likelihood_ratio = compute_likelihood_ratio_reg(losses_in, losses_out, target_model_loss)
    return likelihood_ratio


def lira_vanilla_scores(all_data, all_labels, target_model_losses, K, N_shadow = 10,  seed = 42, target_indices = None):

    seed_list = list(range(seed, seed + N_shadow))
    # shadows = []
    sample_size = int(len(all_data)) // 2

    
    def shadow_iteration(shadow_seed):
       
        rng = np.random.RandomState(shadow_seed)   

        shadow_indices1 = rng.choice(len(all_data), sample_size, replace=True)
        
        # indices that are not shadow indices
        shadow_indices2 = np.array([i for i in range(len(all_data)) if i not in shadow_indices1])

        # train a KNN model on the shadow indices
        shadow_data = all_data[shadow_indices1]
        shadow_labels = all_labels[shadow_indices1]
        shadow_model1 = KNeighborsClassifier(n_neighbors=K)
        shadow_model1.fit(shadow_data, shadow_labels)

        # train a KNN model on the non shadow indices
        shadow_data2 = all_data[shadow_indices2]
        shadow_labels2 = all_labels[shadow_indices2]
        shadow_model2 = KNeighborsClassifier(n_neighbors=K)
        shadow_model2.fit(shadow_data2, shadow_labels2)


        _, indices1 = shadow_model1.kneighbors(all_data, n_neighbors=K)

        labels1 =  all_labels[shadow_indices1[indices1]] 

        losses1 = 1 - np.sum(labels1 == all_labels.reshape(-1,1), axis=1) /K

        _, indices2 = shadow_model2.kneighbors(all_data, n_neighbors=K)

        labels2 =  all_labels[shadow_indices2[indices2]]

        losses2 = 1 - np.sum(labels2 == all_labels.reshape(-1,1), axis=1) /K


        losses_in_1 = losses1[shadow_indices1]
        losses_out_1 = losses2[shadow_indices1]

        losses_in_2 = losses2[shadow_indices2]
        losses_out_2 = losses1[shadow_indices2]

        losses_in = np.zeros(len(all_data))
        losses_out = np.zeros(len(all_data))

        losses_in[shadow_indices1] = losses_in_1
        losses_out[shadow_indices1] = losses_out_1

        losses_in[shadow_indices2] = losses_in_2
        losses_out[shadow_indices2] = losses_out_2

        return losses_in, losses_out



    losses = [shadow_iteration(shadow_seed) for shadow_seed in seed_list]

    arrays_in = [t[0] for t in losses]
    losses_in = np.column_stack(arrays_in)
    arrays_out = [t[1] for t in losses]
    losses_out = np.column_stack(arrays_out)


    if target_indices is None:
        target_indices = range(len(all_data))

    likelihood_ratios = []
    for i, d in enumerate(target_indices):

        if target_indices is None:
            target_model_loss = target_model_losses[d]
        else:
            target_model_loss = target_model_losses[i]

        likelihood_ratio = compute_likelihood_ratio_reg(losses_in[d], losses_out[d], target_model_loss)
        likelihood_ratios.append(likelihood_ratio)



    return likelihood_ratios


from sklearn.base import clone


def lira_model_agnostic_scores(all_data, all_labels, target_model_losses, model_cls, model_params=None, N_shadow=10, seed=42, target_indices=None, should_logit_transform=False, model_loss_fct=None): 
    """
    Computes likelihood ratios for regression models using shadow models.
    
    Parameters:
    - all_data: array-like, shape (n_samples, n_features), Input data.
    - all_labels: array-like, shape (n_samples,), True labels for the input data.
    - target_model_losses: array-like, shape (n_samples,), Losses of the target model on the data.
    - model_cls: class, The scikit-learn model class to use (e.g., KNeighborsClassifier, LogisticRegression).
    - model_params: dict, Optional. Parameters to initialize the model.
    - N_shadow: int, Number of shadow models to use.
    - seed: int, Seed for random number generation.
    - target_indices: list or None, Indices of target data points. If None, use all data points.

    Returns:
    - likelihood_ratios: list of floats, Likelihood ratios for each target data point.
    """
    
    if model_params is None:
        model_params = {}

    seed_list = list(range(seed, seed + N_shadow))
    sample_size = len(all_data) // 2

    def shadow_iteration(shadow_seed):
        rng = np.random.RandomState(shadow_seed)

        shadow_indices1 = rng.choice(len(all_data), sample_size, replace=True)
        shadow_indices2 = np.array([i for i in range(len(all_data)) if i not in shadow_indices1])

        shadow_data1, shadow_labels1 = all_data[shadow_indices1], all_labels[shadow_indices1]
        shadow_data2, shadow_labels2 = all_data[shadow_indices2], all_labels[shadow_indices2]

        # Clone the model with the specified parameters
        shadow_model1 = clone(model_cls(**model_params))
        shadow_model1.fit(shadow_data1, shadow_labels1)

        shadow_model2 = clone(model_cls(**model_params))
        shadow_model2.fit(shadow_data2, shadow_labels2)

        losses1 = np.array([model_loss_fct(shadow_model1, all_data[i], all_labels[i]) for i in range(len(all_data))])
        losses2 = np.array([model_loss_fct(shadow_model2, all_data[i], all_labels[i]) for i in range(len(all_data))])

        losses_in_1 = losses1[shadow_indices1]
        losses_out_1 = losses2[shadow_indices1]
        losses_in_2 = losses2[shadow_indices2]
        losses_out_2 = losses1[shadow_indices2]

        losses_in = np.zeros(len(all_data))
        losses_out = np.zeros(len(all_data))

        losses_in[shadow_indices1] = losses_in_1
        losses_out[shadow_indices1] = losses_out_1

        losses_in[shadow_indices2] = losses_in_2
        losses_out[shadow_indices2] = losses_out_2

        return losses_in, losses_out

    losses = [shadow_iteration(shadow_seed) for shadow_seed in seed_list]

    arrays_in = [t[0] for t in losses]
    losses_in = np.column_stack(arrays_in)
    arrays_out = [t[1] for t in losses]
    losses_out = np.column_stack(arrays_out)

    if target_indices is None:
        target_indices = range(len(all_data))

    likelihood_ratios = []
    for i, d in enumerate(target_indices):
        if target_indices is None:
            target_model_loss = target_model_losses[d]
        else:
            target_model_loss = target_model_losses[i]

        likelihood_ratio = compute_likelihood_ratio_reg(losses_in[d], losses_out[d], target_model_loss, transform=should_logit_transform)
        likelihood_ratios.append(likelihood_ratio)
    
    return likelihood_ratios


def lira_knn_online_attack_uniform(all_data, all_labels, target_model_loss, data_point, data_point_label, K=3, regularization_term=1e-6):
    # Train shadow models and collect confidence scores
    # confs_in, confs_out = shadow_models_knn_and_collect_scores(all_data, all_labels, data_point, data_point_label, N_shadow, K, shadow_dataset_proportion=shadow_dataset_proportion, regularization_term=regularization_term)

    losses_in, losses_out = compute_knn_distributions_uniform(all_data, all_labels, data_point, data_point_label, K)


    # Compute likelihood ratio with regularization
    likelihood_ratio = compute_likelihood_ratio_reg(losses_in, losses_out, target_model_loss, regularization_term=regularization_term)
    return likelihood_ratio

def lira_knn_online_attack_localsearch(all_data, all_labels, target_model_loss, data_point, data_point_label, K=3, regularization_term=1e-6):
    # Train shadow models and collect confidence scores
    # confs_in, confs_out = shadow_models_knn_and_collect_scores(all_data, all_labels, data_point, data_point_label, N_shadow, K, shadow_dataset_proportion=shadow_dataset_proportion, regularization_term=regularization_term)

    losses_in, losses_out = compute_knn_distributions_localsearch(all_data, all_labels, data_point, data_point_label, K)


    # Compute likelihood ratio with regularization
    likelihood_ratio = compute_likelihood_ratio_reg(losses_in, losses_out, target_model_loss, regularization_term=regularization_term)
    return likelihood_ratio


def knn_distance_attack(data_point, nn_model, all_distances, K):
    # this is a basic attack that simply look at all distancs and use the average distance of a KNN model (not the same K as the initial model)
    # we normalize but its not clear if its really necessary since the ROC will do some normalization
    
    # Find the K+1 nearest neighbors (including the point itself)
    distances, indices_ = nn_model.kneighbors([data_point])
    
    # Compute the average distance excluding the first point which is the data point itself
    avg_distance = np.mean(distances[0][1:])

    
    # Normalize the average distance using MinMax scaling
    min_distance = np.min(all_distances)
    max_distance = np.max(all_distances)
    normalized_distance = (avg_distance - min_distance) / (max_distance - min_distance)
    
    return normalized_distance


def create_dataset_excluding_point(training_data, training_labels, excluded_index):
    """
    Create a new dataset excluding the data point at the specified index.

    Parameters:
    - training_data: Array of training data points
    - training_labels: Array of training labels
    - excluded_index: Index of the data point to exclude

    Returns:
    - new_data: New dataset excluding the specified data point
    - new_labels: New labels excluding the label of the specified data point
    """
    mask = np.ones(len(training_data), dtype=bool)
    mask[excluded_index] = False
    new_data = training_data[mask]
    new_labels = training_labels[mask]
    return new_data, new_labels

def lira_online_attacks(target_model, training_data, training_labels, N_shadow, K, shadow_dataset_proportion=0.7, default_distance_metric = 'euclidean'):
    scores = []
    for data_point_idx in range(len(training_data)):

        print(data_point_idx)

        # Create a new dataset excluding the current data point
        new_data, new_labels = create_dataset_excluding_point(training_data, training_labels, data_point_idx)

        # Perform LiRA attack on the excluded data point
        score = lira_online_attack_reg(new_data, new_labels, target_model, training_data[data_point_idx], training_labels[data_point_idx], N_shadow, K, shadow_dataset_proportion=shadow_dataset_proportion, default_distance_metric=default_distance_metric)
        scores.append(score)

    return scores



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import ast
    dataset_name = '/Users/patrickmesana/Dev/data-valuation-core/synthetic/'

    # Load the all_points.csv file
    df_training = pd.read_csv(dataset_name + 'all_points.csv')

    # Extracting training data and labels
    training_features = df_training[['x', 'y']].values
    training_labels = df_training['label'].values

    # Load parameters
    params_df = pd.read_csv(dataset_name + 'synthetic-data-parameters.csv')
    params = params_df.iloc[3]

    # Adjust this line based on the new format of your CSV
    # Example: if centers are listed as a stringified list in the 'centers' column
    test_points = ast.literal_eval(params['centers'])

    N = len(training_features)  # Total number of data points
    K = 1  # Parameter K in k-NN

            # Prepare target model
    target_model = KNeighborsClassifier(n_neighbors=K)
    target_model.fit(training_features, training_labels)

    # Run LiRA attack
    scores = lira_online_attacks(target_model, training_features, training_labels, N_shadow=1000, K=K, shadow_dataset_proportion=[0.6, 0.80])
    print(scores)

    # Create a bar chart of the scores
    plt.figure(figsize=(12, 7))
    plt.bar(np.arange(len(scores)), scores)
    plt.title("LiRA Attack Scores")
    plt.xlabel("Data Points")
    plt.ylabel("Score")
    plt.xticks(np.arange(len(scores)), np.arange(0, len(scores)))
    plt.grid(True)
    plt.show()
