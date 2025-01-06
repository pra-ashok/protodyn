import MDAnalysis as mda 
from MDAnalysis.analysis.dihedrals import Ramachandran, calc_dihedrals
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix

def calculate_ramachandran_angles(pdb_path, xtc_path, selection="protein"):
    """
    Calculate Ramachandran (phi and psi) angles for a protein across an MD trajectory,
    including padding for the first and last residues.

    Parameters:
    - pdb_path (str): Path to the PDB file.
    - xtc_path (str): Path to the XTC trajectory file.
    - selection (str): Atom selection string (default: "protein").

    Returns:
    - np.ndarray: Array of shape (n_frames, n_residues + 2, 2) containing phi and psi angles in radians.
                  The first and last residues are padded with NaN values.
    """
    try:
        # Load the universe with the specified PDB and trajectory
        universe = mda.Universe(pdb_path, xtc_path, in_memory=True)

        # Select the desired atoms (default: protein)
        protein = universe.select_atoms(selection)

        if len(protein) == 0:
            raise ValueError("No atoms selected with the given selection string.")

        # Perform Ramachandran analysis
        ramachandran = Ramachandran(protein)
        ramachandran.run()
        phi_psi_angles = ramachandran.results.angles  # Shape: (n_frames, n_residues, 2)

        # Convert angles from degrees to radians
        phi_psi_angles_radians = np.radians(phi_psi_angles)

        # Create padding with NaN for the first and last residues
        padding = np.full((phi_psi_angles_radians.shape[0], 1, 2), np.nan)

        # Concatenate padding to the beginning and end of the residue dimension
        padded_phi_psi_angles = np.concatenate([padding, phi_psi_angles_radians, padding], axis=1)

        return padded_phi_psi_angles

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def compute_sidechain_chis(pdb_file, xtc_file, start_frame, end_frame, degrees=False):
    """
    Compute dihedral angles for sidechain chi angles in a protein.
    Calls `gather_chi_quads` to identify chi quadruplets and their residue indices.

    Parameters:
    -----------
    pdb_file : str
        Path to the PDB file containing the protein structure.
    xtc_file : str
        Path to the XTC file containing the trajectory.
    start_frame : int
        The first frame in the trajectory to process.
    end_frame : int
        The last frame (exclusive) in the trajectory to process.
    degrees : bool, optional
        Whether to return the computed dihedral angles in degrees (default is False).

    Returns:
    --------
    chi_padded : numpy.ndarray
        A (n_subset_frames, n_residues, max_chi) array, where max_chi is the maximum
        number of chi angles for any residue.
    """
    u = mda.Universe(pdb_file, xtc_file, in_memory=True)
    protein = u.select_atoms("protein")
    n_residues = len(protein.residues)
    n_subset_frames = end_frame - start_frame

    # Gather chi quadruplets and their residue indices
    chi_quadruplets, residue_index_list = gather_chi_quads(protein)

    # Determine the maximum number of chi angles per residue
    max_chi = max(residue_index_list.count(i) for i in set(residue_index_list))

    # Initialize a zero-padded array to store chi angles for all residues
    chi_padded = np.zeros((n_subset_frames, n_residues, max_chi), dtype=np.float32)

    if not chi_quadruplets:
        return chi_padded

    # Track how many chis have been processed for each residue
    chi_counts = {res_idx: 0 for res_idx in set(residue_index_list)}

    # Process trajectory frames
    dihedrals_all = []
    for ts in u.trajectory[start_frame:end_frame]:
        coords = protein.positions
        coords1 = np.array([coords[quadruplet[0]] for quadruplet in chi_quadruplets])
        coords2 = np.array([coords[quadruplet[1]] for quadruplet in chi_quadruplets])
        coords3 = np.array([coords[quadruplet[2]] for quadruplet in chi_quadruplets])
        coords4 = np.array([coords[quadruplet[3]] for quadruplet in chi_quadruplets])

        dihedrals_rad = calc_dihedrals(coords1, coords2, coords3, coords4)
        if degrees:
            dihedrals_all.append(np.degrees(dihedrals_rad))
        else:
            dihedrals_all.append(dihedrals_rad)

    # Convert the list of dihedrals to a numpy array
    dihedrals_all = np.array(dihedrals_all, dtype=np.float32)

    # Map chi angles to their respective residues
    for dihedral_idx, residue_idx in enumerate(residue_index_list):
        chi_idx = chi_counts[residue_idx]  # Determine the chi angle index
        chi_padded[:, residue_idx, chi_idx] = dihedrals_all[:, dihedral_idx]
        chi_counts[residue_idx] += 1  # Increment the chi angle index for the residue

    return chi_padded

def compute_sidechain_com_and_cb_vector(residue):
    """
    Compute the center of mass (COM) of the sidechain and the vector from Cα to Cβ for a residue.

    Parameters:
    -----------
    residue : MDAnalysis Residue
        Residue from which to compute the sidechain COM and Cβ vector.

    Returns:
    --------
    com : np.ndarray
        Center of mass of the sidechain atoms (3D vector).
    cb_vector : np.ndarray
        Vector from Cα to Cβ (3D vector). Returns None if Cβ is not present.
    """
    # Select sidechain atoms (exclude backbone atoms)
    sidechain_atoms = residue.atoms.select_atoms("not name N C O CA")
    if len(sidechain_atoms) == 0:
        return None, None

    # Compute center of mass of the sidechain
    com = sidechain_atoms.center_of_mass()

    # Extract Cα (CA) and Cβ (CB) atom positions
    ca = residue.atoms.select_atoms("name CA")
    cb = residue.atoms.select_atoms("name CB")
    if len(ca) != 1 or len(cb) != 1:
        return com, None

    # Compute the vector from Cα to Cβ
    cb_vector = cb.positions[0] - ca.positions[0]

    return com, cb_vector