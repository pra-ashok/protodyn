import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import GATv2Conv

SIDECHAIN_GROUPS_1LETTER = {
    'A': [False, False, False, False],
    'R': [True,  True,  True,  True ],
    'N': [True,  True,  False, False],
    'D': [True,  True,  False, False],
    'C': [True,  False, False, False],
    'Q': [True,  True,  True,  False],
    'E': [True,  True,  True,  False],
    'G': [False, False, False, False],
    'H': [True,  True,  False, False],
    'I': [True,  True,  False, False],
    'L': [True,  True,  False, False],
    'K': [True,  True,  True,  True ],
    'M': [True,  True,  True,  False],
    'F': [True,  True,  False, False],
    'P': [False, False, False, False],
    'S': [True,  False, False, False],
    'T': [True,  False, False, False],
    'W': [True,  True,  False, False],
    'Y': [True,  True,  False, False],
    'V': [True,  False, False, False],
}


class ResidueRelation(nn.Module):
    """
    Encapsulates residue-based edge feature computation:
    Takes in concatenated residue-embeddings (res_i || res_j) for an edge,
    transforms them, and outputs a learned feature (e.g. 1D).
    """
    def __init__(self, residue_embedding_dim, dim_h_edge=16, dropout_p=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(residue_embedding_dim, dim_h_edge),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim_h_edge, 1),
            nn.Sigmoid()
        )

    def forward(self, concatenated_res_embeds):
        """
        concatenated_res_embeds: shape (num_edges, residue_embedding_dim)
        Returns: shape (num_edges, 1)
        """
        return self.mlp(concatenated_res_embeds)


class NodeEmbedding(nn.Module):
    """
    Generic node embedding module that can be used for both
    sidechain and backbone node initialization.
    """
    def __init__(self, in_features, out_features, dropout_p=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(in_features, out_features),
            ReLU(),
            Dropout(dropout_p)
        )

    def forward(self, x):
        """
        x: shape (num_nodes, in_features)
        Returns: shape (num_nodes, out_features)
        """
        return self.mlp(x)


class EdgeUpdate(nn.Module):
    """
    Update layer for edge features (sidechain edges).
    Uses residual connections: edge_feature <- MLP(edge_feature) + edge_feature
    """
    def __init__(self, edge_dim_in, dim_h_edge=16, dropout_p=0.5):
        super().__init__()
        self.edge_dim_in = edge_dim_in
        self.dim_h_edge = dim_h_edge
        self.dropout_p = dropout_p

        self.mlp = nn.Sequential(
            Linear(edge_dim_in, dim_h_edge),
            BatchNorm1d(dim_h_edge),
            ReLU(),
            Dropout(dropout_p),
            Linear(dim_h_edge, edge_dim_in)
        )

    def forward(self, edge_attr):
        """
        edge_attr: shape (num_edges, edge_dim_in)
        Returns: shape (num_edges, edge_dim_in)
        """
        return self.mlp(edge_attr) + edge_attr  # Residual connection


class BackboneNodeUpdate(nn.Module):
    """
    Applies MLP to sidechain node representation before adding it as a residual
    update to the backbone node.
    """
    def __init__(self, dim_h, dropout_p=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(dim_h, dim_h),
            BatchNorm1d(dim_h),
            ReLU(),
            Dropout(dropout_p)
        )

    def forward(self, sidechain_nodes):
        """
        sidechain_nodes: shape (num_sc_nodes, dim_h)
        Returns: shape (num_sc_nodes, dim_h)
        """
        return self.mlp(sidechain_nodes)


class AttentionFusion(nn.Module):
    """
    Computes a scalar attention weight for gating sidechain representation
    into backbone representation.
    """
    def __init__(self, dim_h, dropout_p=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(dim_h, 1),
            BatchNorm1d(1),
            nn.Sigmoid(),
            Dropout(dropout_p)
        )

    def forward(self, x):
        """
        x: shape (num_nodes, dim_h)
        returns: shape (num_nodes, 1)
        """
        return self.mlp(x)


class SidechainOutputHeads(nn.Module):
    """
    Collect all sidechain outputs (chi angles, coordinates).
    We rely on a mask for each angle, rather than learned attention.
    """
    def __init__(self, dim_h):
        super().__init__()

        self.sc_chi_1_hidden = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.ReLU()
        )
        self.sc_chi_2_hidden = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.ReLU()
        )
        self.sc_chi_3_hidden = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.ReLU()
        )
        self.sc_chi_4_hidden = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.ReLU()
        )

        # Final angle decoder for each chi
        self.angle_decoder = nn.Sequential(
            nn.Linear(dim_h, dim_h, bias=False),
            nn.ReLU(),
            nn.Linear(dim_h, 1, bias=False)
        )

        # v_com
        self.v_com = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, 3)
        )

    def forward(self, sidechain_nodes, residue_types):
        """
        sidechain_nodes: shape (num_sc_nodes, dim_h)
        residue_types:   list of length num_sc_nodes (each is a single-letter code)
        Returns:
          chi_1, chi_2, chi_3, chi_4, v_com
          each chi: (num_sc_nodes, 1)
          v_com:    (num_sc_nodes, 3)
        """
        # Build mask for each node => shape (num_sc_nodes, 4)
        mask_list = []
        print("residue_types", len(residue_types))
        for rtype in residue_types:
            ### MODIFIED ###
            chi_mask_bools = SIDECHAIN_GROUPS_1LETTER.get(rtype, [False, False, False, False])
            mask_list.append(chi_mask_bools)
        angle_mask = torch.tensor(mask_list, device=sidechain_nodes.device, dtype=torch.float32)
        angle_mask = angle_mask.unsqueeze(2)  # Shape becomes (num_nodes, 4, 1)

        # print("angle_mask.shape", angle_mask.shape)
        # print("sidechain_nodes.shape", sidechain_nodes.shape)

        # Chi 1
        chi_1_mask = angle_mask[:, 0].expand_as(sidechain_nodes)
        sc_chi_1_h = self.sc_chi_1_hidden(sidechain_nodes) * chi_1_mask
        chi_1 = self.angle_decoder(sc_chi_1_h) 

        # Chi 2
        chi_2_mask = angle_mask[:, 1].expand_as(sidechain_nodes)
        sc_chi_2_h = self.sc_chi_2_hidden(sidechain_nodes) * chi_2_mask
        chi_2 = self.angle_decoder(sc_chi_2_h)

        # Chi 3
        chi_3_mask = angle_mask[:, 2].expand_as(sidechain_nodes)
        sc_chi_3_h = self.sc_chi_3_hidden(sidechain_nodes) * chi_3_mask
        chi_3 = self.angle_decoder(sc_chi_3_h)

        # Chi 4
        chi_4_mask = angle_mask[:, 3].expand_as(sidechain_nodes)
        sc_chi_4_h = self.sc_chi_4_hidden(sidechain_nodes) * chi_4_mask
        chi_4 = self.angle_decoder(sc_chi_4_h)

        # v_com (no masking; all residues have a sidechain COM)
        v_com = self.v_com(sidechain_nodes)

        return chi_1, chi_2, chi_3, chi_4, v_com


class BackboneOutputHeads(nn.Module):
    """
    Collect all backbone outputs (phi, psi, X_c_alpha, V_c_beta).
    """
    def __init__(self, dim_h):
        super().__init__()
        # phi
        self.bb_phi = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, 1, bias=False)
        )

        # psi
        self.bb_psi = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, 1, bias=False)
        )

        # X_c_alpha
        self.X_c_alpha = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, 3)
        )

        # V_c_beta
        self.V_c_beta = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, 3)
        )

    def forward(self, backbone_nodes):
        """
        backbone_nodes: shape (num_bb_nodes, dim_h)
        Returns: phi, psi, X_c_alpha, V_c_beta
        """
        phi = self.bb_phi(backbone_nodes)
        psi = self.bb_psi(backbone_nodes)
        X_c_alpha = self.X_c_alpha(backbone_nodes)
        V_c_beta = self.V_c_beta(backbone_nodes)
        return phi, psi, X_c_alpha, V_c_beta


class ProtodynModel(nn.Module):
    def __init__(self,
                 sc_node_feature_size,
                 bb_node_feature_size,
                 sidechain_edge_attrs_size,
                 residue_embeddings,   ### MODIFIED: must be single-letter => 1D embedding
                 dim_h=64,
                 dim_h_edge=16,
                 num_layers=12,
                 dropout_p=0.5):
        super().__init__()

        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.dim_h = dim_h

        # Residue embeddings (single-letter dictionary)
        self.residue_embeddings = residue_embeddings

        # Node embedding modules
        self.sidechain_node_layer = NodeEmbedding(
            in_features=sc_node_feature_size,
            out_features=dim_h,
            dropout_p=dropout_p
        )

        self.backbone_node_layer = NodeEmbedding(
            in_features=bb_node_feature_size,
            out_features=dim_h,
            dropout_p=dropout_p
        )

        # For building an additional edge feature from residue types:
        #   The dimension is 2 * the embedding dimension (concat of two residue embeddings).
        example_key = next(iter(self.residue_embeddings))
        single_embed_dim = len(self.residue_embeddings[example_key])
        residue_embedding_dim = 2 * single_embed_dim

        self.residue_relation = ResidueRelation(
            residue_embedding_dim=residue_embedding_dim,
            dim_h_edge=dim_h_edge,
            dropout_p=dropout_p * 0.5
        )

        # GAT layers for sidechain
        self.sidechain_gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=dim_h,
                out_channels=dim_h,
                edge_dim=sidechain_edge_attrs_size,
                dropout=dropout_p,
                residual=True,
                negative_slope=0.2
            )
            for _ in range(num_layers)
        ])

        # GAT layers for backbone
        self.backbone_gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=dim_h,
                out_channels=dim_h,
                dropout=dropout_p,
                residual=True,
                negative_slope=0.2
            )
            for _ in range(num_layers)
        ])

        # Edge update layers (for sidechain edges)
        self.edge_update_layers = nn.ModuleList([
            EdgeUpdate(
                edge_dim_in=sidechain_edge_attrs_size,
                dim_h_edge=dim_h_edge,
                dropout_p=dropout_p
            )
            for _ in range(num_layers)
        ])

        # Backbone node update module (from sidechain)
        self.bb_node_update = BackboneNodeUpdate(dim_h=dim_h, dropout_p=dropout_p)

        # Attention for sidechain -> backbone gating
        self.attention = AttentionFusion(dim_h=dim_h, dropout_p=dropout_p)

        # Output heads
        self.sidechain_outputs = SidechainOutputHeads(dim_h=dim_h)
        self.backbone_outputs = BackboneOutputHeads(dim_h=dim_h)

    def forward(self, data):
        """
        data: dict with keys:
          - sequence: list of single-letter residue codes (e.g. ['G','L','Y','S','E',...])
          - sidechain_node_features: shape (num_sc_nodes, sc_node_feature_size) # float tensor
          - backbone_node_features:  shape (num_bb_nodes, bb_node_feature_size) # float tensor
          - sidechain_edge_attrs:    shape (num_sc_edges, sidechain_edge_attrs_size) # float tensor
          - sidechain_edges:         (2, num_sc_edges) adjacency # long tensor
          - backbone_edges:          (2, num_bb_edges) adjacency # long tensor
        Returns: 
          chi_1, chi_2, chi_3, chi_4, v_com, phi, psi, X_c_alpha, V_c_beta
        """
        # 1) Build residue embedding vectors (one per node).
        #    If your sidechain/backbone nodes each correspond to exactly one residue,
        #    you can do a single embedding per node, indexed by residue.
        #    (Adjust as needed if your graph layout differs.)
        ### MODIFIED ###
        device = data['sidechain_node_features'].device
        residue_embeds_list = [
            torch.tensor(self.residue_embeddings[aa], dtype=torch.float32)
            for aa in data['sequence']
        ]
        residue_embeds = torch.stack(residue_embeds_list, dim=0)  # shape = (num_nodes, embed_dim)
        residue_embeds = residue_embeds.to(device)

        # 2) Extract node/edge features from data
        sidechain_node_features = data['sidechain_node_features']   # (num_sc_nodes, sc_node_feature_size)
        backbone_node_features = data['backbone_node_features']     # (num_bb_nodes,  bb_node_feature_size)
        sidechain_edge_attrs = data['sidechain_edge_attrs']         # (num_sc_edges, sidechain_edge_attrs_size)
        edge_index_sc = data['sidechain_edges']                     # (2, num_sc_edges)
        edge_index_bb = data['backbone_edges']                      # (2, num_bb_edges)

        # 3) Initialize node embeddings
        sidechain_nodes = self.sidechain_node_layer(sidechain_node_features)
        backbone_nodes = self.backbone_node_layer(backbone_node_features)
        print("sidechain_nodes.shape at 3", sidechain_nodes.shape)
        print("backbone_nodes.shape at 3", backbone_nodes.shape)
        # print("backbone_nodes.shape", backbone_nodes.shape)
        print("backbone_nodes at 3--->", backbone_nodes)

        # 4) Residue-embedding-based edge features for sidechain
        sc_edge_pairs = edge_index_sc.t().tolist()  # shape (num_sc_edges, 2)
        list_concat_res_embeds = []
        for (i, j) in sc_edge_pairs:
            # i, j are sidechain node indices (each corresponds to a residue).
            # Adjust if your indexing is different.
            list_concat_res_embeds.append(
                torch.cat((residue_embeds[i], residue_embeds[j]), dim=0)  # (2*embed_dim,)
            )
        concatenated_res_tensor = torch.stack(list_concat_res_embeds, dim=0)  # (num_sc_edges, 2*embed_dim)

        # Build the residue-based edge feature
        residue_edge_features = self.residue_relation(concatenated_res_tensor)
        # print("residue_edge_features.shape", residue_edge_features.shape)
        # print("sidechain_edge_attrs.shape", sidechain_edge_attrs.shape)
        # Combine them with the existing sidechain_edge_attrs (Concatenation)
        edge_features = torch.cat((sidechain_edge_attrs,residue_edge_features),dim=1)
        # print("edge_features.shape")
        # print(edge_features.shape)

        # 5) GAT layers
        for i in range(self.num_layers):
            # Sidechain node update
            sidechain_nodes = self.sidechain_gat_layers[i](
                x=sidechain_nodes, 
                edge_index=edge_index_sc,
                edge_attr=edge_features
            )

            # Sidechain edge update
            edge_features = self.edge_update_layers[i](edge_features)

            # Backbone node update
            backbone_nodes = self.backbone_gat_layers[i](backbone_nodes, edge_index_bb)
            print("sidechain_nodes.shape", sidechain_nodes.shape)
            print("Sidechain_nodes", sidechain_nodes)
            print("backbone_nodes.shape", backbone_nodes.shape)
            print("backbone_nodes", backbone_nodes)
            # Cross-attention: sidechain -> backbone
            attn_weights = self.attention(sidechain_nodes)  # (num_sc_nodes, 1)
            sidechain_to_bb = self.bb_node_update(sidechain_nodes)
            backbone_nodes = backbone_nodes + attn_weights * sidechain_to_bb

        # 6) Sidechain outputs
        chi_1, chi_2, chi_3, chi_4, v_com = self.sidechain_outputs(
            sidechain_nodes, 
            data['sequence']  # pass the single-letter residue types for masking
        )

        # 7) Backbone outputs
        phi, psi, X_c_alpha, V_c_beta = self.backbone_outputs(backbone_nodes)

        # Adjust Sidechain outputs
        chi_1 = (chi_1 + sidechain_node_features[:, 3].unsqueeze(1) + torch.pi)%(2*torch.pi) - torch.pi
        chi_2 = (chi_2 + sidechain_node_features[:, 4].unsqueeze(1) + torch.pi)%(2*torch.pi) - torch.pi
        chi_3 = (chi_3 + sidechain_node_features[:, 5].unsqueeze(1) + torch.pi)%(2*torch.pi) - torch.pi
        chi_4 = (chi_4 + sidechain_node_features[:, 6].unsqueeze(1) + torch.pi)%(2*torch.pi) - torch.pi


        # Adjust Backbone outputs
        phi = (phi + backbone_node_features[:, 6].unsqueeze(1) + torch.pi)%(2*torch.pi) - torch.pi
        psi = (psi + backbone_node_features[:, 7].unsqueeze(1) + torch.pi)%(2*torch.pi) - torch.pi

        # Adjust C-alpha
        print("X_c_alpha.shape from Protodyn_Model", X_c_alpha.shape)
        print("backbone_node_features.shape from Protodyn_Model", backbone_node_features[:,0:3].shape)
        X_c_alpha = X_c_alpha + backbone_node_features[:, 0:3]

        return chi_1, chi_2, chi_3, chi_4, v_com, phi, psi, X_c_alpha, V_c_beta


# Test the model
def test_model(data, device):
    import numpy as np

    # ### MODIFIED ###
    # Suppose you have your single-letter-coded embeddings in a dict (5D here).
    encoded_residues = {
        "A": [2.9167426, 2.258971, 0.6644938, -0.7949218, -6.6698585],
        "R": [33.930508, 10.69357, 12.420517, 12.451375, -15.758884],
        "N":[10.2937155, 1.8236568, 18.31367, -10.806467, -23.719212],
        "D": [19.966536, 2.1608086, 20.85456, -19.048147, -25.287539],
        "C": [5.3353815, 2.037181, 6.915355, -0.88399404, -10.008201],
        "E": [31.911568, 8.223736, 20.926216, -13.377687, -26.336746],
        "Q": [16.671667, 3.2160518, 23.479643, -4.1124864, -27.329218],
        "G": [1.8641269, 2.1092076, 0.05890134, -1.5898108, -5.6500163], 
        "H": [9.721873, -7.388795, 0.35414073, 8.12512, -3.964906], 
        "I": [28.599907, -2.7908437, 19.758871, 11.232325, -28.33977],
        "L": [12.087123, -3.34021, 10.024552, -2.1133611, -17.273863],
        "K": [31.23165, 6.870693, 13.786241, 11.163855, -20.361973], 
        "M": [18.46588, -1.2178062, 16.90781, 1.3556464, -19.953312],
        "F": [7.45427, -28.123909, -20.33646, -10.951878, 15.136281],
        "P": [5.310894, 12.2118, 4.0236216, -25.813725, -0.50325525],
        "S": [4.74718, 3.160784, 6.8495703, -2.7912028, -10.299657], 
        "T": [8.488678, 2.9467175, 7.9047804, -4.8888583, -12.196921],
        "W": [10.40995, -34.584442, -18.562786, -3.2649193, 10.704782],
        "Y": [12.30078, -35.035423, 9.127891, -11.849123, -8.555511],
        "V": [9.013404, 1.0820607, 11.967609, 3.0916648, -18.459846],
        "O": [9.721873, -7.388795, 0.35414073, 8.12512, -3.964906],
        "U": [5.3353815, 2.037181, 6.915355, -0.88399404, -10.008201],
        "X": [0,0,0,0,0]
    }

    # Instantiate model
    model = ProtodynModel(
        sc_node_feature_size=7,         # random placeholder
        bb_node_feature_size=8,         # random placeholder
        sidechain_edge_attrs_size=3,    # random placeholder
        residue_embeddings=encoded_residues,  # pass the single-letter embeddings
        dim_h=8,
        dim_h_edge=4,
        num_layers=8,
        dropout_p=0.1
    ).to(device)
    
    sidechain_edges = [[],[]]
    for i,j in data["sidechain_edges"][0]:
        sidechain_edges[0].append(i)
        sidechain_edges[1].append(j)
    
    backbone_edges = [[],[]]
    for i,j in data["backbone_edges"]:
        backbone_edges[0].append(i)
        backbone_edges[1].append(j)
    
    data = {
        'sequence': [ch for ch in data["protein_sequence"]],
        'sidechain_node_features': torch.tensor(data["sidechain_node_features"][0],dtype=torch.float32,device=device),  # shape = (num_sc_nodes, sc_node_feat_size)
        'backbone_node_features':  torch.tensor(data["backbone_node_features"][0],dtype=torch.float32,device=device),  # shape = (num_bb_nodes,  bb_node_feat_size)
        'sidechain_edge_attrs':    torch.tensor(data["side_chain_edge_attrs"][0],dtype=torch.float32,device=device),  # shape = (num_sc_edges, sidechain_edge_attrs_size)
        'sidechain_edges':         torch.tensor(sidechain_edges, dtype=torch.long, device=device),
        'backbone_edges':          torch.tensor(backbone_edges, dtype=torch.long, device=device),
    }
    print("data['sequence']", data['sequence']) # Error in preprocessing protein sequence
    print("data['sidechain_node_features']", data['sidechain_node_features'].shape)
    print("data['backbone_node_features']", data['backbone_node_features'].shape)
    # Forward pass
    (chi_1, chi_2, chi_3, chi_4, v_com,
     phi, psi, X_c_alpha, V_c_beta) = model(data)

    # Print the shapes
    print("chi_1:", chi_1.shape)      # (4, 1)
    print("chi_2:", chi_2.shape)      # (4, 1)
    print("chi_3:", chi_3.shape)      # (4, 1)
    print("chi_4:", chi_4.shape)      # (4, 1)
    print("v_com:", v_com.shape)      # (4, 3)
    print("phi:", phi.shape)          # (4, 1)
    print("psi:", psi.shape)          # (4, 1)
    print("X_c_alpha:", X_c_alpha.shape)  # (4, 3)
    print("V_c_beta:", V_c_beta.shape)    # (4, 3)
    return (chi_1, chi_2, chi_3, chi_4, v_com, phi, psi, X_c_alpha, V_c_beta)

def test():
    import pickle

    with open("/workspace/protodyn_2/outputs/preprocessed/1zd7_B_R3.pkl", "rb") as f:
        data = pickle.load(f)

     # Create some dummy data
    device = torch.device('cuda:0')
    results = test_model(data, device)
    return results