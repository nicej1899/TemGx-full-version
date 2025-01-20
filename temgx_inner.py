"""
This script demonstrates how to compute an approximate set of influential nodes (Vs) within L-hop neighborhoods
of a target node (Vt) in a temporal graph scenario, using Shapley values to measure their contributions to the model's
prediction on the target node. It also generates data suitable for Dynamic Bayesian Network (DBN) visualization
in JSON format.

Dependencies:
    - torch
    - torch_geometric
    - numpy
    - sklearn
    - json
    - random
    - time
    - os

You will also need:
    - A method `get_metr_la_dataset` to load your METR-LA time-series dataset.
    - A trained model checkpoint of `DCRNN` architecture.
    - Adjust the import paths according to your project structure.

Author: (Your Name)
Date: (Your Date)
"""

import torch
import numpy as np
from torch_geometric.data import Data
import time
import json
import random
import os
from sklearn.preprocessing import KBinsDiscretizer

# Adjust these import paths to match your project structure
from src.data import get_metr_la_dataset  # Make sure this function returns the METR-LA dataset
from src.model.dcrnn import DCRNN         # DCRNN model implementation


# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def get_neighbors(node, edge_index):
    """
    Given a node index and an edge_index tensor of shape (2, E),
    returns all direct neighbors of the node in the graph.

    Parameters:
    -----------
    node : int
        Node index.
    edge_index : torch.Tensor
        The edge index tensor of the graph, shape (2, number_of_edges).

    Returns:
    --------
    neighbors : set
        A set of all neighbors (node indices) of the specified node.
    """
    neighbors = set()
    # Move edge_index to CPU if needed
    edge_index = edge_index.cpu().numpy()
    for src, dst in zip(edge_index[0], edge_index[1]):
        if src == node:
            neighbors.add(dst)
        elif dst == node:
            neighbors.add(src)
    return neighbors


def get_L_hop_neighbors(node, edge_index, L):
    """
    Get all nodes within L-hop distance from a given node.
    For example, 1-hop neighbors, 2-hop neighbors, etc.

    Parameters:
    -----------
    node : int
        The node index for which we want to find neighbors.
    edge_index : torch.Tensor
        The edge index tensor of the graph, shape (2, number_of_edges).
    L : int
        The number of hops (distance).

    Returns:
    --------
    neighbors : set
        A set of node indices that are within L hops of the specified node.
    """
    neighbors = set([node])
    current_layer = set([node])
    for _ in range(L):
        next_layer = set()
        for n in current_layer:
            n_neighbors = get_neighbors(n, edge_index)
            next_layer.update(n_neighbors)
        # Exclude nodes already visited
        next_layer -= neighbors
        neighbors.update(next_layer)
        current_layer = next_layer
    return neighbors


def mask_graph(graph, nodes_to_mask, device):
    """
    Mask (overwrite) the features of specified nodes with the mean feature vector.

    Parameters:
    -----------
    graph : torch_geometric.data.Data
        The input graph containing x (node features) and edge_index.
    nodes_to_mask : list or tensor of int
        The node indices whose features will be replaced by the mean feature value.
    device : str
        The device to which the graph data should be moved ('cpu' or 'cuda').

    Returns:
    --------
    masked_graph : torch_geometric.data.Data
        A copy of the graph with masked node features for the specified nodes.
    """
    masked_x = graph.x.clone().to(device)
    # Convert nodes_to_mask to a tensor (if not already) on the correct device
    nodes_to_mask = torch.tensor(list(nodes_to_mask), dtype=torch.long).to(device)
    # Compute the mean feature vector (moved to device)
    mean_feature_value = graph.x.mean(dim=0).to(device)
    
    # Replace the features of the specified nodes with the mean
    masked_x[nodes_to_mask] = mean_feature_value

    # Return a new Data object with masked features
    return Data(x=masked_x, edge_index=graph.edge_index.to(device))


def approximate_shapley_values(test_node, candidate_nodes, model, graph, device, epsilon=0.001):
    """
    Approximate the Shapley values of candidate_nodes in terms of their contribution
    to the prediction on the test_node. Each node's contribution is measured as
    (baseline_prediction - masked_prediction).

    Parameters:
    -----------
    test_node : int
        The node index for which we want the prediction.
    candidate_nodes : list of int
        Nodes to consider for Shapley value calculation.
    model : torch.nn.Module
        The trained model (e.g., DCRNN).
    graph : torch_geometric.data.Data
        The graph data at the current time step.
    device : str
        The computation device ('cpu' or 'cuda').
    epsilon : float, optional (default=0.001)
        Threshold for including a node in the set Vs (significant contributors).

    Returns:
    --------
    shapley_values : dict
        A dictionary mapping node -> Shapley contribution (float).
    Vs : set
        A set of nodes whose contribution is greater than epsilon in absolute value.
    """
    shapley_values = {}
    Vs = set()
    base_graph = graph.clone().to(device)
    edge_index = base_graph.edge_index.to(device)

    # For the GNN forward pass, we can maintain an edge_weight of all ones
    edge_weight = torch.ones(edge_index.size(1)).to(device)

    # Compute baseline prediction (no masking)
    with torch.no_grad():
        base_output = model(base_graph.x.unsqueeze(0), edge_index, edge_weight)
        # base_output shape might be (1, num_nodes, prediction_dimension)
        # We take the mean over the last dimension for simplicity
        base_output = base_output.squeeze(0).mean(dim=-1).cpu()
    initial_prediction = base_output[test_node].item()

    # For each candidate node, compute contribution by masking it individually
    for node in candidate_nodes:
        masked_graph = mask_graph(graph, [node], device)
        with torch.no_grad():
            output = model(masked_graph.x.unsqueeze(0), masked_graph.edge_index, edge_weight)
            output = output.squeeze(0).mean(dim=-1).cpu()
        prediction = output[test_node].item()
        contribution = initial_prediction - prediction

        shapley_values[node] = contribution
        if abs(contribution) > epsilon:
            Vs.add(node)

    return shapley_values, Vs


def process_single_window(graphs, L, device, model, test_node, epsilon=0.01, window_size=12):
    """
    Process a single time window from a list of temporal graphs to:
    1. Compute the set Vs of influential nodes for each time slice using approximate Shapley values.
    2. Collect feature deltas to fit a KBinsDiscretizer.
    3. Assign discrete labels to nodes based on these feature deltas.
    4. Construct a JSON structure for DBN visualization with:
       - union of Vs nodes across all time slices
       - union of edges (Vs -> Vt)
       - factual vs. counterfactual predictions
       - shapley values, etc.

    Parameters:
    -----------
    graphs : list of torch_geometric.data.Data
        Temporal graphs, one per time slice.
    L : int
        Number of hops to consider for candidate nodes.
    device : str
        The computation device ('cpu' or 'cuda').
    model : torch.nn.Module
        The trained model.
    test_node : int
        The target node Vt for which we compute predictions.
    epsilon : float, optional (default=0.01)
        Threshold for selecting nodes in Vs based on Shapley values.
    window_size : int, optional (default=12)
        The number of time slices in a window.

    Returns:
    --------
    output_data : dict
        A dictionary containing the final union of Vs, union edges, and per-time-slice details.
    """
    print("Processing a single window to compute Vs...")
    num_snapshots = len(graphs)
    assert window_size <= num_snapshots, "Window size exceeds the total number of graphs."

    # For demonstration, we only process the first window
    window_idx = 0
    window_start = window_idx * window_size
    window_end = window_start + window_size
    window_graphs = graphs[window_start:window_end]

    time_slices_data = []
    all_Vs = set()
    all_edges = set()

    # Initialize a KBinsDiscretizer
    # n_bins=5 with 'quantile' strategy can be a start; can be adjusted as needed
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    # First pass: compute Shapley values and collect feature deltas for discretizer
    all_delta_values = []
    shapley_info = []  # store (shapley_values, Vs) for each time slice
    print("First pass: Computing Shapley values and collecting delta features for discretizer...")

    for t, graph in enumerate(window_graphs):
        # Node features (shape: [num_nodes, num_features])
        node_features = graph.x.cpu().numpy()
        # Get L-hop neighbors of test_node
        candidate_nodes = get_L_hop_neighbors(test_node, graph.edge_index, L)
        # Exclude the target node from candidates
        candidate_nodes = [node for node in candidate_nodes if node != test_node]

        # Approximate Shapley values
        shapley_values, Vs = approximate_shapley_values(
            test_node, candidate_nodes, model, graph, device, epsilon
        )
        # Remove the target node from Vs if present
        Vs.discard(test_node)
        shapley_info.append((shapley_values, Vs))
        all_Vs.update(Vs)

        # Collect feature differences (delta) compared to the previous time slice
        if t > 0:
            prev_node_features = window_graphs[t - 1].x.cpu().numpy()
            delta_features = node_features - prev_node_features
        else:
            # For the first time slice, there's no previous slice, so just use current
            delta_features = node_features

        # For each node in Vs, collect the average feature delta
        for node in Vs:
            if node < delta_features.shape[0]:
                delta_value = delta_features[node].mean()
                # Scale the delta to emphasize changes
                scaled_delta = delta_value * 1000
                all_delta_values.append([scaled_delta])

    # If no delta values have been collected (Vs was empty), fit on entire set of nodes
    if not all_delta_values:
        all_delta_values = []
        for t in range(len(window_graphs)):
            node_features = window_graphs[t].x.cpu().numpy()
            if t > 0:
                prev_node_features = window_graphs[t - 1].x.cpu().numpy()
                delta_features = node_features - prev_node_features
            else:
                delta_features = node_features
            scaled_delta = delta_features * 1000
            # Flatten the 2D array into a list of [value], for discretizer
            all_delta_values.extend(scaled_delta.reshape(-1, 1))
        discretizer.fit(all_delta_values)
        print("Discretizer fitted on all node delta features.")
    else:
        discretizer.fit(all_delta_values)
        print("Discretizer fitted on collected scaled delta features.")

    # Second pass: assign discrete labels and build JSON data
    print("Second pass: Assigning labels and constructing JSON data...")
    for t, graph in enumerate(window_graphs):
        shapley_values, Vs = shapley_info[t]
        Vs.discard(test_node)

        # We want to include all nodes that appeared in any Vs plus the target node
        nodes_in_current_slice = all_Vs.union({test_node})

        # Compute feature deltas for the current slice
        node_features = graph.x.cpu().numpy()
        if t > 0:
            prev_node_features = window_graphs[t - 1].x.cpu().numpy()
            delta_features = node_features - prev_node_features
        else:
            delta_features = node_features

        # Move the current graph to the device
        graph = graph.to(device)

        Vs_features = {}
        Vs_labels_dict = {}

        # Assign discrete labels to nodes_in_current_slice
        for node in nodes_in_current_slice:
            if node < delta_features.shape[0]:
                delta_value = delta_features[node].mean()
                scaled_delta = delta_value * 1000
                try:
                    discrete_label = int(discretizer.transform([[scaled_delta]])[0][0])
                    Vs_features[str(node)] = float(scaled_delta)
                    Vs_labels_dict[str(node)] = discrete_label
                    print(f"Time {t}, Node {node}, Scaled Delta Feature: {scaled_delta}, Label: {discrete_label}")
                except Exception as e:
                    print(f"Error discretizing scaled delta feature for node {node}, Time {t}: {e}")
                    Vs_features[str(node)] = float(scaled_delta)
                    Vs_labels_dict[str(node)] = 0  # default label
            else:
                # If the node index is out of bounds for delta_features
                print(f"Warning: Node {node} is out of bounds for delta_features with shape {delta_features.shape}")
                Vs_features[str(node)] = 0.0
                Vs_labels_dict[str(node)] = 0

        # Extract edges from Vs to the test_node (Vs -> Vt)
        edges = []
        for node in Vs:
            if node != test_node:
                edge = [int(node), int(test_node)]
                # Store edges uniquely (sort them to avoid duplicates)
                if tuple(sorted(edge)) not in all_edges:
                    edges.append(edge)
                    all_edges.add(tuple(sorted(edge)))

        # Factual prediction for the current time slice
        with torch.no_grad():
            factual_output = model(
                graph.x.unsqueeze(0),
                graph.edge_index,
                torch.ones(graph.edge_index.size(1)).to(device)
            )
            factual_output = factual_output.squeeze(0).mean(dim=-1).cpu()
        factual_prediction = factual_output[test_node].item()

        # Counterfactual prediction: mask all nodes in Vs and predict again
        if Vs:
            masked_graph = mask_graph(graph, Vs, device)
            with torch.no_grad():
                counterfactual_output = model(
                    masked_graph.x.unsqueeze(0),
                    masked_graph.edge_index,
                    torch.ones(masked_graph.edge_index.size(1)).to(device)
                )
                counterfactual_output = counterfactual_output.squeeze(0).mean(dim=-1).cpu()
            counterfactual_prediction = counterfactual_output[test_node].item()
        else:
            # If Vs is empty, the counterfactual is the same as the factual
            counterfactual_prediction = factual_prediction

        # Compute the prediction change
        prediction_change = counterfactual_prediction - factual_prediction

        # Collect data for the time slice
        time_slice_data = {
            'time_slice': t,
            'Vs': list(map(int, Vs)),
            'Vt': int(test_node),
            'node_features': Vs_features,
            'node_labels': Vs_labels_dict,
            'edges': edges,
            'shapley_values': {
                str(node): shapley_values.get(node, 0.0) for node in nodes_in_current_slice
            },
            'factual_prediction': factual_prediction,
            'counterfactual_prediction': counterfactual_prediction,
            'prediction_change': prediction_change
        }
        time_slices_data.append(time_slice_data)
        print(f"Window {window_idx}, Time {t}, Vs: {Vs}, Prediction Change: {prediction_change:.6f}")

    # For DBN, the graph structure is assumed to be consistent across time slices
    # So we take the union of all Vs plus the target node for final JSON data
    union_Vs = list(map(int, all_Vs.union({test_node})))
    union_edges = [list(edge) for edge in all_edges]

    # Build the final output data structure
    output_data = {
        'union_Vs': union_Vs,
        'union_edges': union_edges,
        'time_slices': time_slices_data
    }

    # Save as JSON file
    output_filename = f'dbn_data_{test_node}_window{window_idx}.json'
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"DBN data saved to {output_filename}")

    print("Processing completed.")
    return output_data


def main_first_layer():
    """
    Main function to demonstrate usage of the above utilities:
    1. Load the METR-LA dataset.
    2. Load a pre-trained DCRNN model.
    3. Convert the dataset into a list of PyG Data objects (temporal graphs).
    4. Call the `process_single_window` function with specified parameters.
    """
    print("Loading dataset...")
    dataset = get_metr_la_dataset()  # Replace with your actual dataset loading function
    print("Dataset loaded successfully.")

    # Update this path to your model checkpoint
    checkpoint_path = "/home/mxl1171/CSE_MSE_RXF131/cradle-members/mdle/mxl1171/tgnn_explain/runs/model_checkpoint_dcrnn_no_skip_LA.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize your model according to the expected architecture
    model = DCRNN(node_features=1, out_channels=32, K=3).to(device)
    
    # Load the model state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Build temporal graphs from the dataset
    # Assuming dataset.features is a list of node feature arrays over time
    # and dataset.edge_index is consistent across time
    temporal_graphs = [
        Data(
            x=torch.tensor(dataset.features[i], dtype=torch.float32),
            edge_index=torch.tensor(dataset.edge_index, dtype=torch.long)
        )
        for i in range(len(dataset.features))
    ]
    print(f"Total graphs loaded: {len(temporal_graphs)}")

    # Choose a test node (Vt)
    test_node = 20
    print(f"Selected test node: {test_node}")

    # Hyperparameters
    L = 2         # Number of hops
    epsilon = 0.005  # Threshold for Shapley value significance
    window_size = 2  # Time window size

    # Ensure there are enough graphs for at least one window
    assert len(temporal_graphs) >= window_size, "Not enough graphs for the specified window_size."

    start_time = time.time()
    # Run the process on a single window
    output_data = process_single_window(
        graphs=temporal_graphs,
        L=L,
        device=device,
        model=model,
        test_node=test_node,
        epsilon=epsilon,
        window_size=window_size
    )
    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds.")
    print("All processes completed successfully.")


if __name__ == "__main__":
    main_first_layer()
