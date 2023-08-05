import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

TOPOLOGY_POS_EMB = 'topology_hparam_search'
MPNN_POS_EMB = 'mpnn_hparam_search'
SGC_POS_EMB = 'sgc_hparam_search'
TM_TRANSFORMER = 'TMTransformer'
MPNN_CONCAT = 'mpnn_concat_hparam_search'


def get_valid_losses(path):
    valid_losses = []
    with open(path, 'r') as f:
        for line in f:
            if 'Mean valid loss' in line:
                valid_losses.append(float(line[line.rfind(':') + 1:].strip()))
    return valid_losses


def get_train_losses(path):
    train_losses = []
    with open(path, 'r') as f:
        for line in f:
            if 'Mean train loss' in line:
                train_losses.append(float(line[line.rfind(':') + 1:].strip()))
    return train_losses


def extract_values(path):
    params_path = path.split('/')[1]
    # prefix = [params_path.split('_')[0]]
    model_type = path.split('/')[0]
    if 'topology' in model_type:
        prefix = 'TOPOLOGY'
    elif 'mpnn' in model_type:
        prefix = 'MPNN'
    elif 'sgc' in model_type:
        prefix = 'SGC'
    elif 'TMTransformer' in model_type:
        prefix = 'TMTransformer'
    else:
        raise Exception('Unknown model type chose.')
    print(params_path)
    model_params = [prefix] + params_path.split('_')[2:-3]
    valid_losses = get_valid_losses(path)
    train_losses = get_train_losses(path)
    return model_params, valid_losses, train_losses


def compare_with_plot(info_1, info_2):
    (_, v_1, _), (_, v_2, _) = info_1, info_2
    min_valid_len = min(len(v_1), len(v_2))
    for ((m, v, t), (c_1, c_2)) in zip([info_1, info_2], [('blue', 'navy'), ('orange', 'red')]):
        x = np.arange(len(t))
        plt.plot(x, t, label=f'{"_".join(m)}_train_loss', color=c_1)
        x = np.arange(len(v))
        plt.plot(x, v, label=f'{"_".join(m)}_valid_loss', color=c_2)
        print(f'{"_".join(m)}_valid_loss: {v[:min_valid_len]}')
    plt.legend()
    plt.ylim([0.8, 1.6])
    plt.show()


def process_training_logs(path):
    """Processes all training logs in a given path"""
    for filepath in [os.path.join(dp, f) for dp, dn, filenames in
                     os.walk(os.path.join(str(project_root()), path)) for f in filenames]:
        if os.path.splitext(filepath)[1] == '.log':
            model_params = filepath.split('_')[5:-3]
            valid_losses = get_valid_losses(filepath)
            train_losses = get_train_losses(filepath)
            print(f' model parameters: {model_params}')
            print(f'train losses: {train_losses}')
            print(f' valid losses : {valid_losses}')
            print('\n')
            x = np.arange(len(valid_losses))
            plt.title('__'.join(model_params))
            plt.plot(x, train_losses, label='train_loss', color='blue')
            plt.plot(x, valid_losses, label='valid_loss', color='navy')
            plt.legend()
            plt.show()


def compare_training_logs(dir_1, log_1, dir_2, log_2):
    info_1 = extract_values('/'.join([dir_1, log_1, 'train.log']))
    info_2 = extract_values('/'.join([dir_2, log_2, 'train.log']))
    compare_with_plot(info_1, info_2)


if __name__ == '__main__':
    # process_training_logs(TOPOLOGY_POS_EMB)
    # Best topology: TOPOLOGY_POS_EMB, 'AdamWbase_topology_5e-05_0.0_0.0_100_256_MoleculeTransformer_2023-02-28_13-21-18',
    # Best MPNN: MPNN_POS_EMB, 'AdamW_base_mpnn_5e-05_0.0_0.0_100_256_MoleculeTransformer_2023-03-01_14-43-32',
    # Best SGC: SGC_POS_EMB, 'No_Train_Pos_base_sgc_5e-05_0.0_0.0_100_256_MoleculeTransformer_2023-03-03_16-58-01',
    # BEST TMPTransformer: TM_TRANSFORMER, 'AdamW_base_topology_5e-05_0.1_0.03_100_256_TMTransformer_2023-03-02_19-12-18',
    # BEST MPNN_CONCAT: MPNN_CONCAT, 'Train_Pos_base_mpnn_concat_5e-05_0.0_0.0_100_256_MoleculeTransformer_2023-03-07_01-27-55'

    compare_training_logs(MPNN_POS_EMB, 'AdamW_base_mpnn_5e-05_0.0_0.0_100_256_MoleculeTransformer_2023-03-01_14-43-32',
                          TOPOLOGY_POS_EMB, 'AdamWbase_topology_5e-05_0.0_0.0_100_256_MoleculeTransformer_2023-02-28_13-21-18',)

#
#
# no_emb_train = [4.57486,
#                 1.59447,
#                 1.46016,
#                 1.39454,
#                 1.3486,
#                 1.32184,
#                 1.29519,
#                 1.2697,
#                 1.24941,
#                 1.23236,
#                 1.22169,
#                 1.2043,
#                 1.19125,
#                 1.17921,
#                 1.16594,
#                 1.15397,
#                 1.13991,
#                 1.13016,
#                 1.11983,
#                 1.11191]
# no_emb_valid = [1.92219,
#                 1.76845,
#                 1.48641,
#                 1.48142,
#                 1.38663,
#                 1.39551,
#                 1.38954,
#                 1.32828,
#                 1.33537,
#                 1.29374,
#                 1.28397,
#                 1.2919,
#                 1.25647,
#                 1.24364,
#                 1.2708,
#                 1.23624,
#                 1.20579,
#                 1.20979,
#                 1.20231,
#                 1.18867, ]
#
# top_emb_valid = [2.22896,
#                  1.51248,
#                  1.37802,
#                  1.29618,
#                  1.24713,
#                  1.19465,
#                  1.18009,
#                  1.15016,
#                  1.13006,
#                  1.11164,
#                  1.09478,
#                  1.08842,
#                  1.07398,
#                  1.06864,
#                  1.05772,
#                  1.05718,
#                  1.03945,
#                  1.02999,
#                  1.04519,
#                  1.02085,
#                  1.02004,
#                  1.00243,
#                  1.00348,
#                  0.99594,
#                  0.99395,
#                  0.99272,
#                  0.98047,
#                  0.97668,
#                  0.97551,
#                  0.96873,
#                  0.96637,
#                  0.9614,
#                  0.96229,
#                  0.96546,
#                  0.95848,
#                  0.95289,
#                  0.9522,
#                  0.94808,
#                  0.94758,
#                  0.95832, ]
#
# top_emb_train = [4.2138,
#                  1.49153,
#                  1.35866,
#                  1.30027,
#                  1.24514,
#                  1.20515,
#                  1.17921,
#                  1.15474,
#                  1.1335,
#                  1.1117,
#                  1.10133,
#                  1.08536,
#                  1.07421,
#                  1.05819,
#                  1.05244,
#                  1.03699,
#                  1.02447,
#                  1.01887,
#                  1.00633,
#                  0.99956,
#                  0.98898,
#                  0.98366,
#                  0.97258,
#                  0.96849,
#                  0.95808,
#                  0.9489,
#                  0.94404,
#                  0.93664,
#                  0.92839,
#                  0.92097,
#                  0.91933,
#                  0.91287,
#                  0.90468,
#                  0.89788,
#                  0.89331,
#                  0.88541,
#                  0.87697,
#                  0.87423,
#                  0.86841,
#                  0.86352]
#
# sin_emb_valid = [2.30791,
#                  1.68614,
#                  1.50381,
#                  1.47373,
#                  1.45105,
#                  1.41758,
#                  1.376,
#                  1.38289, ]
#
# sin_emb_train = [7.84443,
#                  1.67058,
#                  1.45397,
#                  1.40094,
#                  1.37691,
#                  1.3517,
#                  1.3208,
#                  1.30836, ]
#
# mpnn_valid = [6.19848,
#               2.08695,
#               1.58868,
#               1.36705,
#               1.37619,
#               1.23929,
#               1.19642,
#               1.1604,
#               1.16405,
#               1.16734,
#               1.11489,
#               1.09241,
#               1.07458,
#               1.05721,
#               1.06508,
#               1.04123,
#               1.03522,
#               1.02365,
#               1.03313,
#               1.03262,
#               1.02506,
#               1.00826,
#               0.99782,
#               1.00204,
#               0.98644,
#               0.98949,
#               0.97863, ]
#
# mpnn_train = [6.74937,
#               1.80948,
#               1.32684,
#               1.22445,
#               1.15565,
#               1.1156,
#               1.09073,
#               1.06755,
#               1.05479,
#               1.03768,
#               1.02745,
#               1.01459,
#               1.00331,
#               0.99444,
#               0.98748,
#               0.97585,
#               0.97153,
#               0.96392,
#               0.95875,
#               0.95378,
#               0.94489,
#               0.94245,
#               0.94014,
#               0.93209,
#               0.92809,
#               0.92481,
#               0.9247, ]
#
# # plt.style.use(['dark_background', 'presentation'])
# # plt.rcParams["figure.figsize"] = (20,10)
# #
# # plt.title('Top Embeddings vs. No Embeddings')
# # x = np.arange(len(top_emb_train))
# # plt.plot(x, np.array(top_emb_train), label='top_emb_train', linewidth=3.0 , color='maroon')
# # plt.plot(x, np.array(top_emb_valid), label='top_emb_valid', linewidth=3.0, color='tomato')
# #
# # plt.plot(np.arange(len(no_emb_train)), no_emb_train, label='no_emb_train', linewidth=3.0 , color='blue')
# # plt.plot(np.arange(len(no_emb_train)), no_emb_valid, label='no_emb_valid', linewidth=3.0, color='navy')
# # plt.xlabel('Number of Epochs')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()
#
# # plt.rcParams["figure.figsize"] = (20,10)
# #
# # plt.title('Top Embeddings vs. Sin Embeddings')
# # x = np.arange(len(top_emb_train))
# # plt.plot(x, np.array(top_emb_train), label='top_emb_train', linewidth=3.0 , color='maroon')
# # plt.plot(x, np.array(top_emb_valid), label='top_emb_valid', linewidth=3.0, color='tomato')
# # #
# # plt.plot(np.arange(len(sin_emb_train)), sin_emb_train, label='sin_emb_train', linewidth=3.0 , color='darkgreen')
# # plt.plot(np.arange(len(sin_emb_train)), sin_emb_valid, label='sin_emb_valid', linewidth=3.0, color='green')
# # plt.xlabel('Number of Epochs')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()
# #
# #
# # plt.rcParams["figure.figsize"] = (20,10)
# #
# # plt.title('Sin Embeddings vs. No Embeddings')
# # x = np.arange(len(top_emb_train))
# # plt.plot(np.arange(len(no_emb_train)), no_emb_train, label='no_emb_train', linewidth=3.0 , color='blue')
# # plt.plot(np.arange(len(no_emb_train)), no_emb_valid, label='no_emb_valid', linewidth=3.0, color='navy')
# #
# # plt.plot(np.arange(len(sin_emb_train)), sin_emb_train, label='sin_emb_train', linewidth=3.0 , color='darkgreen')
# # plt.plot(np.arange(len(sin_emb_train)), sin_emb_valid, label='sin_emb_valid', linewidth=3.0, color='green')
# # plt.xlabel('Number of Epochs')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()
# #
#
# plt.rcParams["figure.figsize"] = (20,10)
#
# plt.title('Top Embeddings vs. MPNN')
# x = np.arange(len(top_emb_train))
# plt.plot(x, np.array(top_emb_train), label='top_emb_train', linewidth=3.0 , color='maroon')
# plt.plot(x, np.array(top_emb_valid), label='top_emb_valid', linewidth=3.0, color='tomato')
# #
# plt.plot(np.arange(len(mpnn_train)), mpnn_train, label='mpnn_train', linewidth=3.0 , color='gold')
# plt.plot(np.arange(len(mpnn_valid)), mpnn_valid, label='mpnn_valid', linewidth=3.0, color='darkgoldenrod')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
