# TemGX: Generating Temporal Graph Explanations for Temporal Graph Neural Networks
## Overview
Inner Layer (temGX - Subgraph Generation)
Uses a GNN (e.g., a DCRNN) to approximate Shapley values for each node’s contribution to a target node’s prediction over time.
Collects these influential nodes (Vs) and feature deltas as subgraphs, then exports a JSON file for further DBN analysis.
Outer Layer (temGX - DBN Construction)

Reads the generated JSON file.
Learns a Bayesian Network (intra-slice) structure using HillClimbSearch and BIC.
Extends it to a Dynamic Bayesian Network (DBN) by adding inter-slice edges.
Optionally reassigns labels based on discretized feature deltas and augments the dataset.
Fits final CPDs with MaximumLikelihoodEstimator.
## Requirements
Python 3.7+
PyTorch, torch_geometric
numpy, pandas, scikit-learn
pgmpy
json, time, random, os (standard libraries)
Install via:
```
pip install torch torch_geometric numpy pandas scikit-learn pgmpy
```
Usage
Prepare  Dataset

Ensure you have a function get_metr_la_dataset() (or similar) returning the temporal features and edge indices.
Train or load a pre-trained DCRNN model checkpoint.
Run Inner Layer
Execute ```temgx_inner.py``` to build the DBN
Update the checkpoint path and dataset loading as needed.
Execute the script  to generate json file.
Run Outer Layer

Point the script to the generated JSON file .
Execute ```temgx_upper.py``` to build the DBN, learn structures/parameters, and optionally retrieve CPDs.
Notes
You can modify hyperparameters (L, epsilon, window_size, discretization bins, etc.) to suit your dataset.
The DBN edges and Shapley-based explanations can be inspected or visualized for further analysis.
