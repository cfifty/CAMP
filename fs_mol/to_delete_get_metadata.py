# Load the saved model from disk.
import os
import sys
import torch

from dpu_utils.utils import RichPath

from rdkit.Chem import (
    MolFromSmiles,
)

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data.fsmol_dataset import FSMolDataset, DataFold
# from fs_mol.data.binding_data_multitask import MultitaskTaskSampleBatchIterable
from fs_mol.models import gnn_multitask
from fs_mol.models import abstract_torch_fsmol_model
from fs_mol.preprocessing.featurisers.molgraph_utils import *



metadata_pth = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "preprocessing/utils/helper_files/")
metapath = RichPath.create(metadata_pth)
path = metapath.join("metadata.pkl.gz")
metadata = path.read_by_file_suffix()
atom_feature_extractors = metadata["feature_extractors"]
print(atom_feature_extractors)
num = 3
print(atom_feature_extractors[num]._encode_as_onehot)
print(atom_feature_extractors[num]._min_known_num)
print(atom_feature_extractors[num]._max_known_num)
print(atom_feature_extractors[num]._metadata_initialised)