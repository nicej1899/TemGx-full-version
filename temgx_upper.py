"""
This module demonstrates how to:
1. Load dynamic data from a JSON file that contains node information, labels, and feature values across multiple time slices.
2. Convert this data into a format suitable for learning Bayesian Networks (BN) and Dynamic Bayesian Networks (DBN) using pgmpy.
3. Perform structure learning using HillClimbSearch and BIC as the scoring method.
4. Fit parameters (CPDs) using MaximumLikelihoodEstimator.
5. Optionally, perturb labels based on feature differences and apply data augmentation to increase the training set size.

Make sure to install the required libraries before running:
    pip install pandas numpy pgmpy scikit-learn
"""

import pandas as pd
import numpy as np
import json
from pgmpy.models import BayesianNetwork, DynamicBayesianNetwork as DBN
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from sklearn.preprocessing import KBinsDiscretizer

# Set random seed for reproducibility
np.random.seed(42)


def load_json_data(json_file_path: str) -> dict:
    """
    Loads the JSON file from the specified path.

    Parameters
    ----------
    json_file_path : str
        The path to the JSON file containing the dynamic data.

    Returns
    -------
    dict
        A dictionary with the keys 'union_Vs', 'union_edges', and 'time_slices'.
    """
    with open(json_file_path, 'r') as f:
        data_json = json.load(f)
    return data_json


def collect_dataframes_from_time_slices(time_slices: list, all_nodes: list) -> pd.DataFrame:
    """
    Creates a list of DataFrames, one per time slice, ensuring that each DataFrame contains
    all nodes with their corresponding labels and mean feature values. These DataFrames are
    then concatenated into a single DataFrame.

    Parameters
    ----------
    time_slices : list
        A list of dictionaries, each representing a time slice with keys such as 'Vs', 'Vt',
        'node_labels', 'node_features', etc.
    all_nodes : list
        A list of all nodes (including Vt).

    Returns
    -------
    pd.DataFrame
        A combined DataFrame containing all time slices. Each row corresponds to a node in a
        given time slice, with columns [node, label, feature, snapshot].
    """
    data_snapshots = []
    num_time_slices = len(time_slices)

    for t in range(num_time_slices):
        time_slice = time_slices[t]
        node_labels = time_slice.get('node_labels', {})
        node_features = time_slice.get('node_features', {})
        nodes_in_time_slice = set(all_nodes)

        # Create a DataFrame for the current time slice
        df = pd.DataFrame({
            'node': list(nodes_in_time_slice),
            'label': [int(node_labels.get(str(node), 0)) for node in nodes_in_time_slice],
            'feature': [float(node_features.get(str(node), 0.0)) for node in nodes_in_time_slice],
            'snapshot': t
        })
        data_snapshots.append(df)

    # Concatenate into a single DataFrame
    data = pd.concat(data_snapshots, ignore_index=True)
    return data


def pivot_labels_for_bn(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots the label data into a wide format for BN learning. Each row corresponds to a time slice,
    and each column corresponds to a node's label.

    Parameters
    ----------
    data : pd.DataFrame
        The combined data containing columns [node, label, feature, snapshot].

    Returns
    -------
    pd.DataFrame
        A pivoted DataFrame where each node becomes a separate column of labels.
    """
    # Convert to wide format using pivot
    flattened_data = data.pivot(index="snapshot", columns="node", values="label")
    flattened_data.columns = [f"node_{int(col)}" for col in flattened_data.columns]
    flattened_data = flattened_data.reset_index(drop=True)

    # Handle missing values and ensure integer type
    flattened_data.fillna(0, inplace=True)
    flattened_data = flattened_data.astype(int)

    return flattened_data


def learn_bn_structure(flattened_data: pd.DataFrame, Vt: int) -> BayesianNetwork:
    """
    Learns the structure of a Bayesian Network (BN) using HillClimbSearch on the given
    label DataFrame. All nodes except Vt are forced to have edges pointing to Vt.

    Parameters
    ----------
    flattened_data : pd.DataFrame
        Each row is a time slice, and each column is a node label at that time slice.
    Vt : int
        The target node index.

    Returns
    -------
    BayesianNetwork
        A pgmpy BayesianNetwork model with the learned structure.
    """
    node_columns = flattened_data.columns.tolist()
    Vt_column = f"node_{Vt}"

    # Subset of columns excluding Vt
    subnode_columns = [col for col in node_columns if col != Vt_column]

    # Define fixed edges: all subnodes -> Vt
    fixed_edges = [(col, Vt_column) for col in subnode_columns]

    # Define a blacklist for all edges not in fixed_edges
    all_possible_edges = [(u, v) for u in node_columns for v in node_columns if u != v]
    blacklist = [edge for edge in all_possible_edges if edge not in fixed_edges]

    # Hill Climb Search for structure learning with BIC score
    hc = HillClimbSearch(flattened_data)
    model = hc.estimate(
        scoring_method=BicScore(flattened_data),
        fixed_edges=fixed_edges,
        blacklist=blacklist,
        max_iter=100
    )
    return model


def prepare_inter_snapshot_data(flattened_data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new DataFrame representing inter-snapshot transitions. Each row contains data
    from the current time slice (t0) and the next time slice (t1).

    Parameters
    ----------
    flattened_data : pd.DataFrame
        Each row is a time slice; columns are node labels.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing columns like node_X_t0, node_X_t1 for each node X,
        enabling time-slice-to-time-slice DBN learning.
    """
    num_time_slices = flattened_data.shape[0]
    node_columns = flattened_data.columns.tolist()
    inter_snapshot_data = []

    for t in range(num_time_slices - 1):
        snapshot_t = flattened_data.iloc[t]
        snapshot_t1 = flattened_data.iloc[t + 1]
        inter_data = {}
        for col in node_columns:
            inter_data[f"{col}_t0"] = snapshot_t[col]
            inter_data[f"{col}_t1"] = snapshot_t1[col]
        inter_snapshot_data.append(inter_data)

    inter_snapshot_df = pd.DataFrame(inter_snapshot_data)
    inter_snapshot_df.fillna(0, inplace=True)
    inter_snapshot_df = inter_snapshot_df.astype(int)

    return inter_snapshot_df


def filter_inter_snapshot_edges(edges: list, Vt_column: str) -> list:
    """
    Filters out invalid edges for inter-snapshot data, keeping only edges of the form:
    (node_X_t0 -> node_Vt_t1).

    Parameters
    ----------
    edges : list
        A list of tuples (u, v) representing edges learned by HillClimbSearch.
    Vt_column : str
        The target node column name without time suffix (e.g., 'node_20').

    Returns
    -------
    list
        A filtered list of valid edges for the DBN's time-slice transitions.
    """
    filtered_edges = []
    for (u, v) in edges:
        # Keep edges that end in Vt_t1 and start in *_t0
        if u.endswith('_t0') and v == f"{Vt_column}_t1":
            filtered_edges.append((u, v))
    return filtered_edges


def reassign_labels(train_df: pd.DataFrame,
                    feature_flat: pd.DataFrame,
                    discretizer: KBinsDiscretizer,
                    num_time_slices: int,
                    node_columns: list) -> pd.DataFrame:
    """
    Reassigns labels in the training DataFrame based on feature differences between time slices.
    The updated labels are inserted into *_t1 columns in the DBN training data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Original training data with columns like node_X_t0, node_X_t1.
    feature_flat : pd.DataFrame
        DataFrame of mean feature values, with each row as a time slice and columns as nodes.
    discretizer : KBinsDiscretizer
        Already fitted discretizer to transform feature differences into discrete bins.
    num_time_slices : int
        Total number of time slices in the dataset.
    node_columns : list
        A list of node column names (e.g., ['node_1', 'node_2', ...]).

    Returns
    -------
    pd.DataFrame
        A copy of the input train_df with updated labels in *_t1 columns according to feature deltas.
    """
    perturbed_train_df = train_df.copy()

    for t in range(num_time_slices - 1):
        current_means = feature_flat.iloc[t + 1]
        prev_means = feature_flat.iloc[t]
        delta = current_means - prev_means

        for col in node_columns:
            delta_value = delta[col]
            if np.isnan(delta_value):
                new_label = 0  # Default label if delta is NaN
            else:
                new_label = discretizer.transform([[delta_value]])[0][0]
            perturbed_train_df.at[t, f"{col}_t1"] = int(new_label)

    return perturbed_train_df


def build_dbn_structure(dbn: DBN,
                        bn_model: BayesianNetwork,
                        filtered_inter_edges: list,
                        Vt_column: str,
                        num_time_slices: int) -> None:
    """
    Incorporates both intra-slice edges (BN structure) and inter-slice edges (filtered edges)
    into the Dynamic Bayesian Network (DBN) object.

    Parameters
    ----------
    dbn : DBN
        An instance of pgmpy.models.DynamicBayesianNetwork.
    bn_model : BayesianNetwork
        The BayesianNetwork model containing intra-slice edges.
    filtered_inter_edges : list
        A list of tuples representing valid inter-slice edges (u -> v).
    Vt_column : str
        The target node column name without time suffix (e.g., 'node_20').
    num_time_slices : int
        The total number of time slices to replicate the BN structure.
    """
    # Add intra-slice edges (BN) for each time slice
    for t in range(num_time_slices):
        for u, v in bn_model.edges():
            dbn.add_edge(f"{u}_t{t}", f"{v}_t{t}")

    # Add inter-slice edges (e.g., node_X_t0 -> node_Vt_t1)
    for u, v in filtered_inter_edges:
        dbn.add_edge(u, v)


def augment_data(perturbed_train_df: pd.DataFrame, num_augment: int = 20) -> pd.DataFrame:
    """
    Generates additional training samples by randomly perturbing existing rows.

    Parameters
    ----------
    perturbed_train_df : pd.DataFrame
        The training data to augment.
    num_augment : int, optional
        Number of augmented samples to generate (default is 20).

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the original data plus the augmented samples.
    """
    augmented_data = []
    for _ in range(num_augment):
        augmented_sample = perturbed_train_df.sample(n=1, replace=True).copy()
        noise = np.random.randint(-1, 2, size=augmented_sample.shape)
        augmented_sample += noise
        augmented_sample = augmented_sample.clip(lower=0)  # Ensure no negative values
        augmented_data.append(augmented_sample)

    augmented_train_df = pd.concat(augmented_data, ignore_index=True)
    final_train_df = pd.concat([perturbed_train_df, augmented_train_df], ignore_index=True)
    return final_train_df


def main():
    """
    Main workflow to:
    1. Load data from a JSON file containing union_Vs, union_edges, and time_slices.
    2. Construct a combined DataFrame of all time slices, ensuring each node is represented in each slice.
    3. Pivot the label data for Bayesian Network structure learning.
    4. Learn the BN structure where all subnodes point to Vt.
    5. Prepare inter-snapshot data for DBN learning.
    6. Learn inter-snapshot edges and filter only valid edges (node_t0 -> Vt_t1).
    7. Create a DBN, replicate intra-slice edges, and add inter-slice edges.
    8. Reassign labels based on feature deltas using a discretizer.
    9. Augment the data with random perturbations and fit the DBN parameters.
    10. Retrieve CPDs for inspection.
    """
    # Define the path to the JSON file
    json_file_path = 'dbn_data_163_window0.json'

    # Load JSON data
    data_json = load_json_data(json_file_path)

    # Extract information
    union_Vs = data_json.get('union_Vs', [])
    union_edges = data_json.get('union_edges', [])
    time_slices = data_json.get('time_slices', [])
    num_time_slices = len(time_slices)

    # Determine target node (Vt) and collect all nodes
    if num_time_slices == 0:
        raise ValueError("No time slices found in the JSON data.")
    Vt = time_slices[0].get('Vt')
    if Vt is None:
        raise KeyError("Missing 'Vt' key in the first time slice.")

    all_nodes = set(union_Vs)
    all_nodes.add(Vt)
    all_nodes = list(all_nodes)

    # Collect time slice data into a single DataFrame
    data = collect_dataframes_from_time_slices(time_slices, all_nodes)

    # Pivot label data for BN structure learning
    flattened_data = pivot_labels_for_bn(data)

    # Learn BN structure
    bn_model = learn_bn_structure(flattened_data, Vt)

    # Prepare inter-snapshot data for DBN
    inter_snapshot_df = prepare_inter_snapshot_data(flattened_data)

    # Learn inter-snapshot edges
    node_columns = flattened_data.columns.tolist()
    Vt_column = f"node_{Vt}"
    hc_inter = HillClimbSearch(inter_snapshot_df)
    inter_model = hc_inter.estimate(
        scoring_method=BicScore(inter_snapshot_df),
        fixed_edges=[(f"{col}_t0", f"{Vt_column}_t1") for col in node_columns],
        blacklist=[],
        max_iter=100
    )

    # Filter valid inter-slice edges
    filtered_inter_edges = filter_inter_snapshot_edges(inter_model.edges(), Vt_column)

    # Create DBN and add intra-slice + inter-slice edges
    dbn = DBN()
    build_dbn_structure(dbn, bn_model, filtered_inter_edges, Vt_column, num_time_slices)

    # Prepare data for DBN parameter learning (train_df)
    # Each row: columns node_X_t0, node_X_t1 for all X
    train_data = []
    for t in range(num_time_slices - 1):
        snapshot_t = flattened_data.iloc[t]
        snapshot_t1 = flattened_data.iloc[t + 1]
        data_dict = {}
        for col in node_columns:
            data_dict[f"{col}_t0"] = snapshot_t[col]
            data_dict[f"{col}_t1"] = snapshot_t1[col]
        train_data.append(data_dict)
    train_df = pd.DataFrame(train_data)

    # Build feature means DataFrame
    feature_means = data.copy()
    feature_means['mean_feature'] = feature_means['feature']
    feature_flat = feature_means.pivot(index="snapshot", columns="node", values="mean_feature")
    feature_flat.columns = [f"node_{int(col)}" for col in feature_flat.columns]
    feature_flat = feature_flat.reset_index(drop=True)

    # Fit a KBinsDiscretizer on feature deltas (excluding Vt)
    subnode_columns = [col for col in node_columns if col != Vt_column]
    all_delta_values = []
    for t in range(1, num_time_slices):
        current_means = feature_flat.iloc[t]
        prev_means = feature_flat.iloc[t - 1]
        delta = current_means - prev_means
        for col in subnode_columns:
            delta_value = delta[col]
            all_delta_values.append([delta_value])

    # Clean up the delta values before fitting
    all_delta_values = np.array(all_delta_values)
    all_delta_values = all_delta_values[~np.isnan(all_delta_values)].reshape(-1, 1)

    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    discretizer.fit(all_delta_values)

    # Reassign labels in the training data based on feature deltas
    perturbed_train_df = reassign_labels(
        train_df, feature_flat, discretizer, num_time_slices, node_columns
    )
    perturbed_train_df = perturbed_train_df.astype(int)

    # Augment data by adding random perturbations
    final_train_df = augment_data(perturbed_train_df, num_augment=20)

    # Fit the DBN parameters using MaximumLikelihoodEstimator
    try:
        dbn.fit(final_train_df, estimator=MaximumLikelihoodEstimator)
    except ValueError as e:
        raise ValueError(f"Error during DBN parameter fitting: {e}")

    # The user can optionally retrieve CPDs using:
    # cpd = dbn.get_cpds(f"{Vt_column}_t1")
    # and then analyze or print it if needed.


if __name__ == "__main__":
    main()
