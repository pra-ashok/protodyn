import numpy as np
import os, pickle, logging
import MDAnalysis as mda
from typing import Dict, Any
from multiprocessing import Pool
from MDAnalysis.analysis.dihedrals import Ramachandran
from protodyn.constants import *
from protodyn.data.node_features import *
from protodyn.data.edge_features import *
from protodyn.data.utils import *
from utils import find_pdb_xtc_pairs


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# File Logger
file_handler = logging.FileHandler("protodyn_preprocess.log")
file_handler.setLevel(logging.DEBUG)

# Console Logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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
        logger.info("Precomputing Ramachandran and chi angles...")
        self.ram_angles = calculate_ramachandran_angles(self.pdb_file, self.xtc_file)
        self.chi_angles = compute_sidechain_chis(self.pdb_file, self.xtc_file, 0, len(self.universe.trajectory))
        logger.info("Precomputed Ramachandran and chi angles.")

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

        logger.debug(f"Processing frame {frame_idx}...")
        # Compute Cα coordinates relative to centroid
        logger.debug("Computing Cα coordinates...")
        ca_coords = np.array([
            res.atoms.select_atoms("name CA").positions[0] for res in self.protein.residues
        ]) - self.centroid
        logger.info(f"Computed Cα coordinates for {len(ca_coords)} residues.")

        logger.debug("Extracting sidechain features...")
        # Extract precomputed Cβ vectors and sidechain COMs
        logger.debug("Extracting Cβ vectors")
        cb_vectors = np.array([
            data[1] if data[1] is not None else [np.nan, np.nan, np.nan]
            for data in self.sidechain_data
        ])

        logger.debug("Extracting sidechain COMs")
        sidechain_coms = np.array([
            data[0] if data[0] is not None else [np.nan, np.nan, np.nan]
            for data in self.sidechain_data
        ]) - self.centroid
        logger.info(f"Extracted sidechain features for {len(sidechain_coms)} residues for {frame_idx}th frame.")
        
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
        logger.debug(f"Processing sidechain edges for frame {frame_idx}...")
        self.universe.trajectory[frame_idx]
        sidechain_coms = np.array([
            data[0] if data[0] is not None else [np.nan, np.nan, np.nan]
            for data in self.sidechain_data
        ])
        logger.info(f"Extracted sidechain COMs for {len(sidechain_coms)} residues for {frame_idx}th frame.")

        return compute_sidechain_edges_with_com(
            self.protein,
            self.filtered_pairs,
            sidechain_coms,
            self.min_sidechain_dist
        )

    def compute_all(self):
        """
        Compute all features and edges.
        """
        
        self._precompute_features()

        logger.info("Computing backbone and sidechain features...")
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

        logger.info("Processed backbone and sidechain features.")

        logger.info("Computing sidechain edges...")
        # Parallel processing of sidechain edges
        with Pool(processes=self.n_jobs) as pool:
            edge_results = pool.map(self._process_sidechain_edges, range(n_frames))
        logger.info("Processed sidechain edges.")

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
            "threshold": self.threshold,
            "min_sidechain_dist": self.min_sidechain_dist
        }
    
    def save(self, output_path: str):
        """
        Save computed data to a file.

        Parameters:
        -----------
        output_file : str
            Path to the output file.
        """
        filename = os.path.basename(self.xtc_file).split(".")[0]
        output_file = os.path.join(output_path, f"{filename}.pkl")
        data = self.get_data()
        with open(output_file, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved computed data to {output_file}.")
        return

def preprocess_protein(pdb_file_path: str, xtc_file_path: str, output_path: str, selection: str = "protein", chain:str = "A", threshold: float = 15.0, min_sidechain_dist: float = 5.0, n_jobs: int = None):
    """ 
    Preprocess a protein trajectory and save the computed data.
    
    Parameters:
    -----------
    pdb_path : str
        Path to the PDB file.
    xtc_path : str
        Path to the XTC trajectory file.
    output_path : str
        Path to save the computed data.

    """
    builder = ProteinGraphBuilder(pdb_file_path, xtc_file_path)
    builder.compute_all()
    builder.save(output_path)
    
    return

def preprocess_protein_bulk(raw_data_path: str, output_path: str, selection: str = "protein", chain:str = "A", threshold: float = 15.0, min_sidechain_dist: float = 5.0, n_jobs: int = None):
    """
    Preprocess multiple protein trajectories in bulk.

    Parameters:
    -----------
    pdb_xtc_pairs : list
        List of tuples containing (pdb_file, xtc_file) paths.
    output_path : str
        Path to save the computed data.
    selection : str
        Atom selection string (default: "protein").
    chain : str
        Chain identifier (default: "A").
    threshold : float
        Distance threshold (Å) for Cα pairs (default: 15.0).
    min_sidechain_dist : float
        Minimum distance threshold (Å) for sidechain edges (default: 5.0).
    n_jobs : int
        Number of parallel processes to use (default: all available cores).
    """
    # Find all .pdb and .xtc files in the directory
    pdb_xtc_pairs = find_pdb_xtc_pairs(raw_data_path)
    logger.info(f"Found {len(pdb_xtc_pairs)} protein trajectories for preprocessing.")

    # Preprocess each protein trajectory
    for pdb_file, xtc_file in pdb_xtc_pairs:
        logger.info(f"Preprocessing {pdb_file} and {xtc_file}...")
        preprocess_protein(pdb_file, xtc_file, output_path)
        logger.info(f"Preprocessing complete for {pdb_file} and {xtc_file}.")

    logger.info("Preprocessing complete.")
    
    return
