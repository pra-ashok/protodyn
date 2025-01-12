import torch
import torch.nn.functional as F

def calculate_loss(output_data, target_data):
    """
    Calculate the loss between the output and target data
    output_data (tuple): The output data from the model (chi1, chi2, chi3, chi4, V_com, phi, psi, X_c_alpha, V_beta)
    """
    
    # Calculate the chi1 loss
    chi_1_output = output_data[0]
    chi_1_target = target_data["sidechain_node_features"][:,3]
    chi_1_loss = 1 - torch.cos(chi_1_output - chi_1_target).mean()

    # Calculate the chi2 loss
    chi_2_output = output_data[1]
    chi_2_target = target_data["sidechain_node_features"][:,4]
    chi_2_loss = 1 - torch.cos(chi_2_output - chi_2_target).mean()

    # Calculate the chi3 loss
    chi_3_output = output_data[2]
    chi_3_target = target_data["sidechain_node_features"][:,5]
    chi_3_loss = 1 - torch.cos(chi_3_output - chi_3_target).mean()

    # Calculate the chi4 loss
    chi_4_output = output_data[3]
    chi_4_target = target_data["sidechain_node_features"][:,6]
    chi_4_loss = 1 - torch.cos(chi_4_output - chi_4_target).mean()

    # Calculate the  phi-psi loss
    phi_output = output_data[5]
    phi_target = target_data["backbone_node_features"][:,6]
    phi_loss = 1 - torch.cos(phi_output - phi_target).mean()

    psi_output = output_data[6]
    psi_target = target_data["backbone_node_features"][:,7]
    psi_loss = 1 - torch.cos(psi_output - psi_target).mean()

    # Calculate the coords loss
    X_c_alpha_output = output_data[7]
    X_c_alpha_target = target_data["backbone_node_features"][:,:3]
    print(X_c_alpha_output[:10])
    print(X_c_alpha_target[:10])
    coords_loss = F.mse_loss(X_c_alpha_output, X_c_alpha_target)
    # Calculate V_beta loss
    V_beta_output = output_data[8]
    V_beta_target = target_data["backbone_node_features"][:,3:6]
    V_beta_loss = F.mse_loss(V_beta_output, V_beta_target)
    # Calculate the V_com loss
    V_com_output = output_data[4]
    V_com_target = target_data["backbone_node_features"][:,:3]
    V_com_loss = F.mse_loss(V_com_output, V_com_target)

    # Calculate the total loss
    total_loss = chi_1_loss + chi_2_loss + chi_3_loss + chi_4_loss + phi_loss + psi_loss + coords_loss + V_beta_loss + V_com_loss
    loss_dict = {
        "chi_1_loss": chi_1_loss.item(),
        "chi_2_loss": chi_2_loss.item(),
        "chi_3_loss": chi_3_loss.item(),
        "chi_4_loss": chi_4_loss.item(),
        "phi_loss": phi_loss.item(),
        "psi_loss": psi_loss.item(),
        "coords_loss": coords_loss.item(),
        "V_beta_loss": V_beta_loss.item(),
        "V_com_loss": V_com_loss.item(),
        "total_loss": total_loss.item()
    }

    return total_loss, loss_dict