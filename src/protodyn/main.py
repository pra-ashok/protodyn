import os
from preprocessing import *

# Initialize the ProteinGraphBuilder
pdb_path = "/content/drive/MyDrive/Datasets/atlas_trajectories/1a62_A_analysis/1a62_A.pdb"
xtc_path = "/content/drive/MyDrive/Datasets/atlas_trajectories/1a62_A_analysis/1a62_A_R2.xtc"

builder = ProteinGraphBuilder(pdb_path, xtc_path, selection="protein")

# Compute all features and edges
builder.compute_all()

# Retrieve the data
data = builder.get_data()

# Access individual components
backbone_features = data['backbone_node_features']  # (Frames, n_residues, features)
sidechain_features = data['sidechain_node_features']  # (Frames, n_residues, features)
backbone_edges = data['backbone_edges']  # List of backbone edges [n_residues, 2]
sidechain_edges = data['sidechain_edges']  # (Frames, edge_index, 2)
sidechain_edge_attrs = data['side_chain_edge_attrs']  # (Frames, edge_attrs, 2) - (min_dist, COM_dist)
centroid = data['centroid']  # (x, y, z)
protein_sequence = data['protein_sequence']  # String

print("Backbone Features shape: ", backbone_features.shape)
print("Sidechain Features shape: ", sidechain_features.shape)
print("Peptide edges: ", len(backbone_edges))
print("Sidechain edges: ", len(sidechain_edges))