import numpy as np
import MDAnalysis as mda
from typing import Dict, Any
from multiprocessing import Pool
from MDAnalysis.analysis.dihedrals import Ramachandran
from protodyn.constants import *
from protodyn.data.node_features import *
from protodyn.data.edge_features import *

class ProteinGraphBuilder:
    def __init__(self, pdb_file: str, xtc_file: str, selection: str = "protein", threshold: float = 15.0, min_sidechain_dist: float = 5.0, n_jobs: int = None):
        """
        Initialize the ProteinGraphBuilder class.

        Parameters:
        -----------
        pdb_file : str
            Path to the PDB file.
        xtc_file : str
            Path to the XTC trajectory file.
        selection : str
            Atom selection string (default: "protein").
        threshold : float
            Distance threshold (Å) for Cα pairs (default: 15.0).
        min_sidechain_dist : float
            Minimum distance threshold (Å) for sidechain edges (default: 5.0).
        n_jobs : int
            Number of parallel processes to use (default: all available cores).
        """
        self.pdb_file = pdb_file
        self.xtc_file = xtc_file
        self.selection = selection
        self.threshold = threshold
        self.min_sidechain_dist = min_sidechain_dist
        self.n_jobs = n_jobs or os.cpu_count()

        # Load universe and protein
        self.universe = mda.Universe(pdb_file, xtc_file, in_memory=True)
        self.protein = self.universe.select_atoms(selection)

        if len(self.protein) == 0:
            raise ValueError("No atoms found with the given selection string.")

        # Compute initial centroid and sequence
        self.centroid = self.protein.center_of_mass()
        self.protein_sequence = "".join(res.resname.upper() for res in self.protein.residues)

        # Precompute Cα distances and backbone edges
        self.filtered_pairs, self.ca_sparse_matrix = precompute_ca_distances(self.protein, threshold=self.threshold)
        self.backbone_edges = self._compute_backbone_edges()

        # Precompute sidechain COMs and CB vectors
        self.sidechain_data = [
            compute_sidechain_com_and_cb_vector(res)
            for res in self.protein.residues
        ]

        # Placeholder for computed features and edges
        self.backbone_node_features = None
        self.sidechain_node_features = None
        self.sidechain_edges = None
        self.sidechain_edge_attrs = None

    def _compute_backbone_edges(self):
        """
        Compute backbone edges by connecting consecutive residues.

        Returns:
        --------
        backbone_edges : list of tuple
            List of tuples (i, i+1) for consecutive residues.
        """
        return [(i, i + 1) for i in range(len(self.protein.residues) - 1)]

    def _precompute_features(self):
        """
        Precompute static features such as Ramachandran angles and chi angles.
        """
        print("Precomputing static features...")
        self.ram_angles = calculate_ramachandran_angles(self.pdb_file, self.xtc_file)
        self.chi_angles = compute_sidechain_chis(self.pdb_file, self.xtc_file, 0, len(self.universe.trajectory))

    def _process_frame(self, frame_idx):
        """
        Process a single frame for backbone and sidechain features.

        Parameters:
        -----------
        frame_idx : int
            Frame index to process.

        Returns:
        --------
        frame_data : dict
            Dictionary containing frame-specific data.
        """
        self.universe.trajectory[frame_idx]

        # Compute Cα coordinates relative to centroid
        ca_coords = np.array([
            res.atoms.select_atoms("name CA").positions[0] for res in self.protein.residues
        ]) - self.centroid

        # Extract precomputed Cβ vectors and sidechain COMs
        cb_vectors = np.array([
            data[1] if data[1] is not None else [np.nan, np.nan, np.nan]
            for data in self.sidechain_data
        ])

        sidechain_coms = np.array([
            data[0] if data[0] is not None else [np.nan, np.nan, np.nan]
            for data in self.sidechain_data
        ]) - self.centroid

        # Extract Ramachandran and chi angles for the frame
        frame_ram_angles = self.ram_angles[frame_idx]
        frame_chi_angles = self.chi_angles[frame_idx]

        return {
            'ca_coords': ca_coords,
            'cb_vectors': cb_vectors,
            'sidechain_coms': sidechain_coms,
            'frame_ram_angles': frame_ram_angles,
            'frame_chi_angles': frame_chi_angles
        }

    def _process_sidechain_edges(self, frame_idx):
        """
        Compute sidechain edges for a single frame.

        Parameters:
        -----------
        frame_idx : int
            Frame index to process.

        Returns:
        --------
        edges_attrs : tuple
            Tuple containing edges and attributes for the frame.
        """
        self.universe.trajectory[frame_idx]
        sidechain_coms = np.array([
            data[0] if data[0] is not None else [np.nan, np.nan, np.nan]
            for data in self.sidechain_data
        ])

        return compute_sidechain_edges_with_com(
            self.protein,
            sidechain_coms,
            self.filtered_pairs,
            self.min_sidechain_dist
        )

    def compute_all(self):
        """
        Compute all features and edges.
        """
        print("Precomputing features...")
        self._precompute_features()

        print("Processing backbone and sidechain features in parallel...")
        n_frames = len(self.universe.trajectory)

        # Parallel processing of frame features
        with Pool(processes=self.n_jobs) as pool:
            frame_results = pool.map(self._process_frame, range(n_frames))

        # Assign backbone and sidechain features
        n_residues = len(self.protein.residues)
        self.backbone_node_features = np.zeros((n_frames, n_residues, 8), dtype=np.float32)
        self.sidechain_node_features = np.zeros((n_frames, n_residues, 7), dtype=np.float32)

        for frame_idx, frame_data in enumerate(frame_results):
            self.backbone_node_features[frame_idx, :, :3] = frame_data['ca_coords']
            self.backbone_node_features[frame_idx, :, 3:6] = frame_data['cb_vectors']
            self.backbone_node_features[frame_idx, :, 6:] = frame_data['frame_ram_angles']
            self.sidechain_node_features[frame_idx, :, :3] = frame_data['sidechain_coms']
            self.sidechain_node_features[frame_idx, :, 3:] = frame_data['frame_chi_angles']

        print("Processing sidechain edges in parallel...")

        # Parallel processing of sidechain edges
        with Pool(processes=self.n_jobs) as pool:
            edge_results = pool.map(self._process_sidechain_edges, range(n_frames))

        # Assign edges and attributes
        self.sidechain_edges = [result[0] for result in edge_results]
        self.sidechain_edge_attrs = [result[1] for result in edge_results]

    def get_data(self) -> Dict[str, Any]:
        """
        Retrieve all computed data.

        Returns:
        --------
        data : dict
            Dictionary containing all computed data.
        """
        return {
            'backbone_node_features': self.backbone_node_features,
            'sidechain_node_features': self.sidechain_node_features,
            'backbone_edges': self.backbone_edges,
            'sidechain_edges': self.sidechain_edges,
            'side_chain_edge_attrs': self.sidechain_edge_attrs,
            'centroid': self.centroid,
            'protein_sequence': self.protein_sequence,
        }
