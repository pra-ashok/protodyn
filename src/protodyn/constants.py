import numpy as np

CHI_DEFINITIONS = {
    'ALA': [],
    'ARG': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'CD'],
        ['CB', 'CG', 'CD', 'NE'],
        ['CG', 'CD', 'NE', 'CZ']
    ],
    'ASN': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'OD1']
    ],
    'ASP': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'OD1']
    ],
    'CYS': [
        ['N', 'CA', 'CB', 'SG']
    ],
    'GLN': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'CD'],
        ['CB', 'CG', 'CD', 'OE1']
    ],
    'GLU': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'CD'],
        ['CB', 'CG', 'CD', 'OE1']
    ],
    'GLY': [],
    'HIS': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'ND1']
    ],
    'ILE': [
        ['N', 'CA', 'CB', 'CG1'],
        ['CA', 'CB', 'CG1', 'CD']
    ],
    'LEU': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'CD1']
    ],
    'LYS': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'CD'],
        ['CB', 'CG', 'CD', 'CE'],
        ['CG', 'CD', 'CE', 'NZ']
    ],
    'MET': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'SD'],
        ['CB', 'CG', 'SD', 'CE']
    ],
    'PHE': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'CD1']
    ],
    'PRO': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'CD']
    ],
    'SER': [
        ['N', 'CA', 'CB', 'OG']
    ],
    'THR': [
        ['N', 'CA', 'CB', 'OG1']
    ],
    'TRP': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'CD1']
    ],
    'TYR': [
        ['N', 'CA', 'CB', 'CG'],
        ['CA', 'CB', 'CG', 'CD1']
    ],
    'VAL': [
        ['N', 'CA', 'CB', 'CG1']
    ]
}

RESIDUE_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

RES_INDEX = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
    "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
    "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19
}


RES_EMBEDS = {
    "A": np.array([2.9167426, 2.258971, 0.6644938, -0.7949218, -6.6698585], dtype=np.float32),
    "R": np.array([33.930508, 10.69357, 12.420517, 12.451375, -15.758884], dtype=np.float32),
    "N": np.array([10.2937155, 1.8236568, 18.31367, -10.806467, -23.719212], dtype=np.float32),
    "D": np.array([19.966536, 2.1608086, 20.85456, -19.048147, -25.287539], dtype=np.float32),
    "C": np.array([5.3353815, 2.037181, 6.915355, -0.88399404, -10.008201], dtype=np.float32),
    "E": np.array([31.911568, 8.223736, 20.926216, -13.377687, -26.336746], dtype=np.float32),
    "Q": np.array([16.671667, 3.2160518, 23.479643, -4.1124864, -27.329218], dtype=np.float32),
    "G": np.array([1.8641269, 2.1092076, 0.05890134, -1.5898108, -5.6500163], dtype=np.float32),
    "H": np.array([9.721873, -7.388795, 0.35414073, 8.12512, -3.964906], dtype=np.float32),
    "I": np.array([28.599907, -2.7908437, 19.758871, 11.232325, -28.33977], dtype=np.float32),
    "L": np.array([12.087123, -3.34021, 10.024552, -2.1133611, -17.273863], dtype=np.float32),
    "K": np.array([31.23165, 6.870693, 13.786241, 11.163855, -20.361973], dtype=np.float32),
    "M": np.array([18.46588, -1.2178062, 16.90781, 1.3556464, -19.953312], dtype=np.float32),
    "F": np.array([7.45427, -28.123909, -20.33646, -10.951878, 15.136281], dtype=np.float32),
    "P": np.array([5.310894, 12.2118, 4.0236216, -25.813725, -0.50325525], dtype=np.float32),
    "S": np.array([4.74718, 3.160784, 6.8495703, -2.7912028, -10.299657], dtype=np.float32),
    "T": np.array([8.488678, 2.9467175, 7.9047804, -4.8888583, -12.196921], dtype=np.float32),
    "W": np.array([10.40995, -34.584442, -18.562786, -3.2649193, 10.704782], dtype=np.float32),
    "Y": np.array([12.30078, -35.035423, 9.127891, -11.849123, -8.555511], dtype=np.float32),
    "V": np.array([9.013404, 1.0820607, 11.967609, 3.0916648, -18.459846], dtype=np.float32)
}
