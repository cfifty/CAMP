import copy
import numpy as np
import sys
import itertools

from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix

from pyprojroot import here as project_root


sys.path.insert(0, str(project_root()))


def flatten(l):
    return [item for sublist in l for item in sublist]

def adjacency_lists_to_distance_matrix(adjacency_lists, num_atoms):
    adjacency_lists = np.array(flatten(adjacency_lists))
    rows = adjacency_lists[:, 0]
    cols = adjacency_lists[:, 1]
    data = np.ones(cols.shape)
    return csr_matrix((data, (rows, cols)), shape=(num_atoms, num_atoms))