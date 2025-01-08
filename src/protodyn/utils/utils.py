import math
import os, shutil

def list_pkl_files_in_directory(directory):
    try:
        # List all files in the given directory with .pkl extension
        pkl_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.pkl')]
        return pkl_files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def list_analysis_directories(parent_directory):
    """
    Lists all directories ending with 'analysis' in the given parent directory.

    Parameters:
    parent_directory (str): The path of the parent directory to search within.

    Returns:
    list: A list of paths to directories ending with 'analysis'.
    """
    # List all directories in the parent directory
    analysis_directories = [
        os.path.join(parent_directory, d) 
        if os.path.isdir(os.path.join(parent_directory, d)) and d.endswith('analysis') 
        else None
        for d in os.listdir(parent_directory)
    ]
    
    # Filter out any None values
    return [d for d in analysis_directories if d]

def find_pdb_xtc_pairs(parent_directory):
    """
    Finds and returns a list of tuples, each containing the path to a .pdb file
    and a corresponding .xtc file in the same subdirectory where the .pdb file
    name is a substring of the .xtc file name.

    Parameters:
    parent_directory (str): The path of the parent directory to search within.

    Returns:
    list: A list of tuples, where each tuple is (pdb_file, xtc_file).
    """
    pdb_xtc_pairs = []

    # Traverse through the subdirectories
    for root, dirs, files in os.walk(parent_directory):
        # Find the .pdb file
        pdb_files = [f for f in files if f.endswith('.pdb')]
        
        # Proceed if there's exactly one .pdb file
        if pdb_files:
            pdb_file = os.path.join(root, pdb_files[0])  # Assume only one .pdb file per directory
            pdb_filename = os.path.splitext(pdb_files[0])[0]  # Get the pdb file name without extension

            # Find all matching .xtc files
            xtc_files = [f for f in files if f.endswith('.xtc') and pdb_filename in f]

            # Pair the .pdb file with each matching .xtc file
            for xtc_file in xtc_files:
                xtc_file_path = os.path.join(root, xtc_file)
                pdb_xtc_pairs.append((pdb_file, xtc_file_path))

    return pdb_xtc_pairs

def get_first_pdb_file(directory_path):
    """
    Returns the full path of the first .pdb file found in the given directory.

    Parameters:
    directory_path (str): The path of the directory to search for .pdb files.

    Returns:
    str: The full path of the first .pdb file, or None if no .pdb file is found.
    """
    # List all .pdb files in the directory
    pdb_files = [f for f in os.listdir(directory_path) if f.endswith('.pdb')]
    
    # Return the first .pdb file found, or None if no files are found
    if pdb_files:
        return os.path.join(directory_path, pdb_files[0])
    else:
        return None

def create_tmp_frames_subfolder(directory_path):
    """
    Creates a subfolder named 'tmp_frames' within the specified directory.

    Parameters:
    directory_path (str): The path of the directory where the subfolder will be created.

    Returns:
    str: The full path of the created 'tmp_frames' subfolder.
    """
    # Define the path for the new subfolder
    tmp_frames_path = os.path.join(directory_path, 'tmp_frames')
    
    # Create the subfolder if it doesn't already exist
    os.makedirs(tmp_frames_path, exist_ok=True)
    
    return tmp_frames_path


def delete_folder(directory_path):
    """
    Cleans the 'tmp_frames' subfolder within the specified directory.

    Parameters:
    directory_path (str): The path of the directory where the cleanup will occur.

    Returns:
    None
    """
    # Define the path for the 'tmp_frames' subfolder
    # tmp_frames_path = os.path.join(directory_path, 'tmp_frames')
    
    # Check if 'tmp_frames' exists and remove it along with its contents
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
        print(f"'tmp_frames' subfolder and its contents have been removed from: {directory_path}")
    else:
        print(f"No 'tmp_frames' subfolder found at: {directory_path}")

def get_xtc_files(directory_path):
    """
    Returns a list of all .xtc files in the given directory.

    Parameters:
    directory_path (str): The path of the directory to search for .xtc files.

    Returns:
    list: A list of full paths to .xtc files found in the directory.
    """
    # List all .xtc files in the directory
    xtc_files = [f for f in os.listdir(directory_path) if f.endswith('.xtc')]
    
    # Get the full path for each .xtc file
    xtc_file_paths = [os.path.join(directory_path, f) for f in xtc_files]
    
    return xtc_file_paths

def determine_optimal_pair(total_processors):
    """
    Determines the optimal pair of (processors per protein, number of proteins processed at a time)
    based on the total processors available.
    
    :param total_processors: Total number of available processors.
    :return: Tuple of (processors per protein, number of proteins processed at a time)
    """
    min_processors_per_trajectory = 10
    max_processors_per_trajectory = 20
    best_difference = total_processors
    best_pair = (min_processors_per_trajectory, 1)

    for processors_per_protein in range(min_processors_per_trajectory, max_processors_per_trajectory + 1):
        proteins_per_stage = total_processors // processors_per_protein
        used_processors = processors_per_protein * proteins_per_stage
        difference = total_processors - used_processors

        if difference < best_difference:
            best_difference = difference
            best_pair = (processors_per_protein, proteins_per_stage)
        
        # If an exact match is found, break early
        if difference == 0:
            break

    return best_pair

if __name__ == '__main__':
    total_processors = 135
    print("Optimal Pair:",determine_optimal_pair(total_processors))

    print(find_pdb_xtc_pairs("/worskpace/data/raw_data"))
