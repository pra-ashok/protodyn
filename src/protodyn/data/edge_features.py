import MDAnalysis as mda 
from MDAnalysis.analysis.dihedrals import Ramachandran, calc_dihedrals
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
from protodyn.data.utils import *

# Takes 30 seconds for each frame for a protein of size 415aa
def compute_sidechain_edges_with_com(protein, filtered_pairs, coms, min_dist_threshold=5.0):
    """
    Compute sidechai~n edges for a chunk using cached Cα pairs, including sidechain COM distances.

    Parameters:
    -----------
    protein : MDAnalysis AtomGroup
        AtomGroup representing the protein.
    filtered_pairs : list of tuple
        List of residue pairs (i, j) within the Cα distance threshold.
    min_dist_threshold : float
        Distance threshold (Å) for sidechain atom pair distances to form edges.

    Returns:
    --------
    edges : list of tuple
        List of residue pairs (i, j) forming edges based on sidechain distances.
    attrs : list of tuple
        List of attributes for each edge, including:
        - Minimum distance between sidechain atoms.
        - Distance between sidechain COMs.
    """
    edges = []
    attrs = []
    # print(f"Computing sidechain edges for {len(filtered_pairs)} pairs...")
    filtered_pairs = list((i,j) for i,j in filtered_pairs)
    try:
        for i, j in filtered_pairs:
            if i >= j:  # Avoid duplicates and self-comparisons
                continue

            # Extract residues
            residue_i = protein.residues[i]
            residue_j = protein.residues[j]

            # Compute COM and CB vectors
            # Redundancy Compute the com and cb only once
            com_i, com_j = coms[i], coms[j]

            # Skip if sidechain COMs cannot be computed
            if com_i is None or com_j is None:
                continue

            # Extract sidechain atoms for the residue pair
            sidechain_atoms_i = residue_i.atoms.select_atoms("not name N C O CA")
            sidechain_atoms_j = residue_j.atoms.select_atoms("not name N C O CA")

            # Skip if any residue doesn't have sidechain atoms
            if len(sidechain_atoms_i) == 0 or len(sidechain_atoms_j) == 0:
                continue

            # Compute pairwise distances between sidechain atoms
            pairwise_distances = distance_matrix(sidechain_atoms_i.positions, sidechain_atoms_j.positions)

            # Compute the minimum distance between sidechains
            min_distance = pairwise_distances.min()
            if min_distance < min_dist_threshold:
                # Compute COM distance
                com_distance = np.linalg.norm(com_i - com_j)

                # Append edge and attributes
                edges.append((i, j))
                attrs.append((min_distance, com_distance))

    except Exception as e:
        print(f"An error occurred while computing sidechain edges: {e}")
        # print("Filtered Pairs from :", filtered_pairs[:10])
        exit()

    return (edges, attrs)