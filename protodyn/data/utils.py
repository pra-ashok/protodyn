import MDAnalysis as mda 
from MDAnalysis.analysis.dihedrals import Ramachandran, calc_dihedrals
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix

def gather_chi_quads(protein):
    """
    Build a list of quadruplet atom indices (relative to protein.atoms) for chi angles.
    Also track the residue index that each quadruplet belongs to.

    Parameters:
    -----------
    protein : MDAnalysis AtomGroup
        The protein atom group selected from an MDAnalysis Universe. This should
        contain residues and their associated atoms.

    Returns:
    --------
    chi_quadruplets : list of tuple
        A list of quadruplets (4-tuples) of atom indices. Each quadruplet corresponds
        to a chi-angle definition for a residue in the protein. Indices are relative
        to the `protein` AtomGroup.
    residue_index_list : list of int
        A list of residue indices corresponding to each quadruplet in `chi_quadruplets`.
        This maps each chi-angle quadruplet to the residue it belongs to.

    Notes:
    ------
    - Chi (χ) angles describe the torsional rotations around bonds in side chains
      of amino acids.
    - Not all residues have chi angles (e.g., glycine and alanine).
    - This function ensures that only valid chi-angle definitions are included.
    """
    # Initialize the outputs: lists to store chi-angle quadruplets and residue indices
    chi_quadruplets = []
    residue_index_list = []

    # Get the global indices of all atoms in the protein selection
    protein_indices = protein.atoms.indices.tolist()

    # Iterate through all residues in the protein
    for i, residue in enumerate(protein.residues):
        # Get the residue name and convert it to uppercase (to match CHI_DEFINITIONS)
        resname = residue.resname.upper()

        # Skip residues that do not have chi-angle definitions
        if resname not in CHI_DEFINITIONS:
            continue

        # Get the list of chi-angle definitions for the current residue
        chi_list = CHI_DEFINITIONS[resname]

        # Process each chi-angle definition (list of 4 atoms) for the residue
        for chi_atoms in chi_list:
            quadruplet = []  # Temporarily store indices for the current chi angle

            # Find the indices of the 4 atoms needed for the chi angle
            for aname in chi_atoms:
                # Select the atom with the given name in the current residue
                sel = residue.atoms.select_atoms(f"name {aname}")

                # If the atom is not uniquely found, skip this quadruplet
                if len(sel) != 1:
                    quadruplet = []
                    break

                # Get the global index of the selected atom
                global_idx = protein_indices.index(sel.indices[0])
                quadruplet.append(global_idx)

            # If a valid quadruplet was formed, add it to the outputs
            if len(quadruplet) == 4:
                chi_quadruplets.append(tuple(quadruplet))  # Add as a tuple
                residue_index_list.append(i)  # Record the residue index

    # Return the list of chi-angle quadruplets and their corresponding residue indices
    return chi_quadruplets, residue_index_list


def precompute_ca_distances(protein, threshold=15.0):
    """
    Precompute and cache the Cα distance matrix as a sparse matrix.

    Parameters:
    -----------
    protein : MDAnalysis AtomGroup
        AtomGroup representing the protein.
    threshold : float
        Distance threshold (Å) for filtering residue pairs based on Cα distances.

    Returns:
    --------
    filtered_pairs : list of tuple
        List of residue pairs (i, j) within the distance threshold.
    sparse_matrix : scipy.sparse.coo_matrix
        Sparse matrix representation of the Cα distance matrix.
    """
    # Step 1: Compute Cα coordinates
    ca_coords = np.array([res.atoms.select_atoms("name CA").positions[0] for res in protein.residues])

    # Step 2: Compute pairwise Cα distances
    ca_distances = distance_matrix(ca_coords, ca_coords)

    # Step 3: Create sparse matrix for Cα distances below the threshold
    mask = ca_distances < threshold
    sparse_matrix = coo_matrix((ca_distances[mask], np.where(mask)))

    # Step 4: Extract filtered pairs
    filtered_pairs = list(zip(sparse_matrix.row, sparse_matrix.col))

    return filtered_pairs, sparse_matrix


if __name__ == "__main__":
    universe = mda.Universe(pdb_path, xtc_path,in_memory=True)
    protein = universe.select_atoms("protein")
    filtered_pairs, sparse_matrix = precompute_ca_distances(protein)
    print("Length of Filtered_pairs: ", len(filtered_pairs)) # Computes in microseconds

    edges, attrs =  compute_sidechain_edges_with_com(
        protein,
        filtered_pairs,
        min_dist_threshold=5.0
    )

    print("Edges and Attrs length: ",len(edges), len(attrs)) # Computes in 29seconds)

    print("Edge[:10]:->",edges[:10], type(edges))

    print("Attrs[:10]:->",attrs[:10],type(attrs))

    