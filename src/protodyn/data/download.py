import os
import requests
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
import signal
import logging
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)

# progress = Progress(
#     TextColumn("[bold green]{task.fields[filename]}", justify="right"),
#     BarColumn(bar_width=None),
#     "[progress.percentage]{task.percentage:>3.1f}%",
#     "•",
#     DownloadColumn(),
#     "•",
#     TransferSpeedColumn(),
#     "•",
#     TimeRemainingColumn(),
# )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def handle_sigint(signum, frame):
    """Handle interrupt signals to gracefully shutdown the process pool."""
    print("Interrupted by user, shutting down...")
    raise SystemExit


signal.signal(signal.SIGINT, handle_sigint)


def download_traj_data(text_file, destination_folder,thread_count):
    # Initialize an empty list
    pdb_list = []

    # Open the file and read the lines
    with open(text_file, 'r') as file:
        lines = file.readlines()

        # Process each line except the header
        for line in lines:
            line = line.strip()
            if line and line != "all_pdbs.txt":
                pdb_list.append(line)
    
    # Creating a custom progress bar
    with Progress(
        # TextColumn("[progress.description] {task.description}"),
        # BarColumn(bar_width=None, complete_style="green", finished_style="green bold"),
        # TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        # transient=True  # The progress bar disappears after the task is complete
        TextColumn("[progress.description] {task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        BarColumn(bar_width=None, complete_style="green", finished_style="green bold"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn()
    ) as progress:
        task_id = progress.add_task("[cyan]Downloading and Unzipping..", total=len(pdb_list))
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(
                download_and_unzip, pdb_id, destination_folder) 
                for pdb_id in pdb_list]
            results = []
            for future in futures:
                result = future.result()  # Wait for each future to complete
                results.append(result)
                progress.update(task_id, advance=1)

        # download_and_unzip(pdb_id, destination_folder)
    return

def download_and_unzip(pdb_id, destination_folder):
    """
    Download the zip file from the API and unzip it into a specific folder.

    :param pdb_id: The PDB ID.
    :param destination_folder: The folder where the zip file will be saved and unzipped.
    """
    # Define the URL for the zip file
    url = f'https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/{pdb_id}'
    
    # -----------------------------------
    def unzip_folder(zip_filename):
        # Create a specific folder for the unzipped contents
        specific_folder = os.path.join(destination_folder, f"{pdb_id}_analysis")
        os.makedirs(specific_folder, exist_ok=True)

        # Unzip the file into the specific folder
        with ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(specific_folder)
        print(f"Unzipped {zip_filename} to {specific_folder}")
        return
    # -------------------------------------------------------
    # Define the local filename for the downloaded zip file
    zip_filename = os.path.join(destination_folder, f"{pdb_id}_analysis.zip")
    try:
        # Skip download if the directory already exists
        if not os.path.exists(zip_filename):
            # Download the zip file
            response = requests.get(url)
            if response.status_code == 200:
                with open(zip_filename, 'wb') as zip_file:
                    zip_file.write(response.content)
                logging.info(f"Downloaded {zip_filename}")
                # unzip the zip
                unzip_folder(zip_filename)
            # -------------------------------------------------------------
            else:
                print(f"Failed to download {url}")
                return
        else:
            logging.info(f"Skipping download and unzip for {pdb_id} as it already exists.")
            return
        
    except Exception as err:
        logging.error(f"Error occurred while processing - {pdb_id} Error-> {err}")
    
    return
