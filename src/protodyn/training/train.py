import torch
import torch.nn as nn
import torch.optim as optim
from protodyn.gnn.protodyn_model import ProtodynModel
from protodyn.constants import RES_EMBEDS
from protodyn.training.loss import calculate_loss
import json, os, pickle

# Training logger for the model
import logging
from datetime import datetime

# Define the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define the file handler
log_file = f"train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Define the console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define the formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def train_on_gpu(epochs, model_save_path, log_file_path, cuda_device, list_of_files):
    """
    Train the model on the GPU

    Args:
    - epochs: Number of epochs to train the model
    - model_save_path: Path to save the trained model
    - log_file_path: Path to save the loss
    - cuda_device: GPU device ID
    - list_of_files: List of files to train the model
    """

    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    # Load the model
    # Instantiate model
    model = ProtodynModel(
        sc_node_feature_size=7,         # random placeholder
        bb_node_feature_size=8,         # random placeholder
        sidechain_edge_attrs_size=3,    # random placeholder
        residue_embeddings=RES_EMBEDS,  # pass the single-letter embeddings
        dim_h=8,
        dim_h_edge=4,
        num_layers=8,
        dropout_p=0.1
    ).to(device)
    
    # Define the loss function and optimizer
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model.train()
    running_loss = 0.0
    log_file_dir = os.path.dirname(log_file_path)

    for epoch in range(epochs): # loop over the Epochs
        epoch_loss = 0.0

        for pkl_file in list_of_files: # loop over the files
            with open(pkl_file, 'rb') as fdp:
                data = pickle.load(fdp)

            logger.info(f"Training on file: {pkl_file}")
            
            file_loss = 0.0 # loss for the file
            seq = [ch for ch in data["protein_sequence"]] # list of single-letter amino acid codes
            backbone_edges = [[],[]]

            for i,j in data["backbone_edges"]:
                backbone_edges[0].append(i)
                backbone_edges[1].append(j)

            input_data,target = None, None
            for ts in range(len(data["backbone_node_features"])-2): # loop over the timesteps in the trajectory
                sidechain_edges = [[],[]]
                traj_loss = 0.0 # loss for the trajectory
                for i,j in data["sidechain_edges"][ts]:
                    sidechain_edges[0].append(i)
                    sidechain_edges[1].append(j)

                input_data = {
                    'sequence': seq,
                    'sidechain_node_features': torch.tensor(data["sidechain_node_features"][ts],dtype=torch.float32,device=device),  
                    'backbone_node_features':  torch.tensor(data["backbone_node_features"][ts],dtype=torch.float32,device=device), 
                    'sidechain_edge_attrs':    torch.tensor(data["side_chain_edge_attrs"][ts],dtype=torch.float32,device=device),
                    'sidechain_edges':         torch.tensor(sidechain_edges, dtype=torch.long, device=device),
                    'backbone_edges':          torch.tensor(backbone_edges, dtype=torch.long, device=device),
                }

                target_data = {
                    "sequence": seq,
                    "sidechain_node_features": torch.tensor(data["sidechain_node_features"][ts+1],dtype=torch.float32,device=device),
                    "backbone_node_features":  torch.tensor(data["backbone_node_features"][ts+1],dtype=torch.float32,device=device),
                    "sidechain_edge_attrs":    torch.tensor(data["side_chain_edge_attrs"][ts+1],dtype=torch.float32,device=device),
                    "sidechain_edges":         torch.tensor(sidechain_edges, dtype=torch.long, device=device),
                    "backbone_edges":          torch.tensor(backbone_edges, dtype=torch.long, device=device),
                }
                        
                optimizer.zero_grad()
                
                outputs = model(input_data)
                loss, loss_dict = calculate_loss(outputs, target_data)
                loss.backward()
                optimizer.step()
                
                traj_loss += loss.item()
                if ts % 40 == 1:
                    traj_loss_1 = traj_loss / ts
                    logger.info(f"loss - ts: {traj_loss_1} - {ts}")
                    with open(os.path.join(log_file_dir,"loss.json"), 'a') as fd_json:
                        json.dump(loss_dict, fd_json)

                file_loss = traj_loss / len(data["backbone_node_features"])
                logger.info(f"loss - file: {file_loss} - {pkl_file}")

            epoch_loss += file_loss
    
        running_loss = epoch_loss / len(list_of_files)
    
        # Save the model
        torch.save(model.state_dict(), model_save_path + f"protein-{epoch}.pth")
        logger.info(f"Model saved at: {model_save_path}-epoch-{epoch}")
        
    
    logger.info("Finished Training")
