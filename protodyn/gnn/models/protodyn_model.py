import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

# Import your building blocks
from protodyn.gnn.layers.residue_embedding import ResidueEmbedding
from protodyn.gnn.layers.layer_norm_linear import LayerNormLinear
from protodyn.gnn.modules.sidechain_gcn import SideChainGCN
from protodyn.gnn.modules.backbone_gcn import BackboneGCN

class ProtoDynGNN(nn.Module):
    """
    SideChain Edges represent non-convalent interactions
    Example architecture:
      1) Initialize node features (side-chain, backbone).
      2) (Side-chain) For L layers:
         - GCN pass -> update hat_h
         - Then update hat_e based on new hat_h
      3) (Backbone) For 2 layers:
         - GCN pass -> update h
         - (Optional) edge updates
         - SC-BB interaction
      4) Produce coarse-grained outputs
    """

    def __init__(self,
                 num_res_types,
                 residue_embed_dim,
                 sidechain_in_dim, 
                 backbone_in_dim,
                 node_hidden_dim,
                 edge_hidden_dim, 
                 num_sidechain_layers=3):
        super().__init__()

        # Residue Embedding for side-chain edges
        self.res_embed = ResidueEmbedding(num_res_types, residue_embed_dim)

        # Node initialization
        self.sidechain_init = LayerNormLinear(sidechain_in_dim, node_hidden_dim)
        self.backbone_init  = LayerNormLinear(backbone_in_dim, node_hidden_dim)

        # Edge initialization
        sc_edge_input_dim = (1 + 1) + (2 * residue_embed_dim)  # e.g. d_sidechain_com + d_min + embed(R_i,R_j)
        self.sidechain_edge_init = nn.Sequential(
            nn.LayerNorm(sc_edge_input_dim),
            nn.Linear(sc_edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim)
        )

        # Edge update: hat{e}_{ij}^{(l)} += Linear( cat(hat{h}_i, hat{h}_j) )
        self.sidechain_edge_update = nn.Linear(node_hidden_dim*2, edge_hidden_dim)

        # GCN for side-chain node updates
        self.sidechain_gcn = SideChainGCN(node_hidden_dim, node_hidden_dim)

        # GCN for backbone node updates
        self.backbone_gcn = BackboneGCN(node_hidden_dim, node_hidden_dim)

        # SC-BB interaction M_{scbb} = Linear(ReLU(h_i, hat_h_i))
        self.scbb_interact = nn.Sequential(
            nn.Linear(node_hidden_dim*2, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, node_hidden_dim)
        )

        self.num_sidechain_layers = num_sidechain_layers

        # Output heads for coarse-grained angles & coords
        self.sc_angle_head = nn.Linear(node_hidden_dim, 1)
        self.bb_angle_head = nn.Linear(node_hidden_dim, 1)
        self.sc_coord_head = nn.Linear(node_hidden_dim, 3)
        self.bb_coord_head = nn.Linear(node_hidden_dim, 3)

    def forward(self,
                # Node features (side-chain)
                V_com, SBF_chi,
                # Node features (backbone)
                X_c_alpha, V_c_beta, SBF_phi, SBF_psi,
                # Distances & Residue info for side-chain edges
                d_sidechain_com, d_min, residue_idx_i, residue_idx_j,
                edge_index_sc,
                # For backbone edges, let's assume we only have adjacency
                edge_index_bb,
                # Old angles & coords (for offset)
                old_chi, old_phi, old_psi,
                old_cbeta_coords, old_calpha_coords):

        # 1) Initialize node embeddings
        sc_in = torch.cat([V_com, SBF_chi], dim=-1)
        hat_h = self.sidechain_init(sc_in)

        bb_in = torch.cat([X_c_alpha, V_c_beta, SBF_phi, SBF_psi], dim=-1)
        h = self.backbone_init(bb_in)

        # 2) Initialize side-chain edges
        res_pair_emb = self.res_embed(residue_idx_i, residue_idx_j)
        if d_sidechain_com.dim() == 1:
            d_sidechain_com = d_sidechain_com.unsqueeze(-1)
        if d_min.dim() == 1:
            d_min = d_min.unsqueeze(-1)
        sc_edge_in = torch.cat([d_sidechain_com, d_min, res_pair_emb], dim=-1)
        hat_e = self.sidechain_edge_init(sc_edge_in)

        e_bb = torch.ones(edge_index_bb.size(1), device=edge_index_bb.device)

        # 3) Side-chain L-layer updates
        for _ in range(self.num_sidechain_layers):
            # Node update
            edge_weight_sc = hat_e.mean(dim=-1)  # reduce to scalar
            hat_h_new = self.sidechain_gcn(hat_h, edge_index_sc, edge_weight=edge_weight_sc)
            hat_h_new = F.relu(hat_h_new)
            hat_h = hat_h + hat_h_new

            # Edge update
            sc_src, sc_dst = edge_index_sc
            e_update_input = torch.cat([hat_h[sc_src], hat_h[sc_dst]], dim=-1)
            e_update = self.sidechain_edge_update(e_update_input)
            hat_e = hat_e + e_update

        # 4) Backbone 2-layer updates
        for _ in range(2):
            # SC-BB interaction
            M_scbb = self.scbb_interact(torch.cat([h, hat_h], dim=-1))

            # Backbone GCN pass
            h_new = self.backbone_gcn(h, edge_index_bb, edge_weight=e_bb)
            h_new = F.relu(h_new)

            # Combine
            h = h + h_new + M_scbb

        # 5) Coarse-Grained Outputs
        sc_act = F.relu(hat_h)
        delta_chi = self.sc_angle_head(sc_act).squeeze(-1)

        chi_new = []
        for k in range(len(old_chi)):
            chi_old_k = old_chi[k]
            updated_k = (chi_old_k + delta_chi + 3.14159) % (2 * 3.14159) - 3.14159
            chi_new.append(updated_k)

        bb_act = F.relu(h)
        delta_phi = self.bb_angle_head(bb_act).squeeze(-1)
        delta_psi = self.bb_angle_head(bb_act).squeeze(-1)
        phi_new = (old_phi + delta_phi + 3.14159) % (2 * 3.14159) - 3.14159
        psi_new = (old_psi + delta_psi + 3.14159) % (2 * 3.14159) - 3.14159

        delta_cbeta = self.sc_coord_head(sc_act)
        V_cbeta_new = old_cbeta_coords + delta_cbeta

        delta_calpha = self.bb_coord_head(bb_act)
        X_calpha_new = old_calpha_coords + delta_calpha

        return {
            "chi_new": chi_new,
            "phi_new": phi_new,
            "psi_new": psi_new,
            "V_cbeta_new": V_cbeta_new,
            "X_calpha_new": X_calpha_new
        }
