import copy
import numpy as np
import torch

from typing import List

"""
num_nodes = 23
num_edges = 25
adjacency_lists=[tensor([[ 0,  1],
        [ 2,  3],
        [ 3,  4],
        [ 4,  5],
        [ 4,  6],
        [ 7,  8],
        [ 9, 10],
        [ 9, 12],
        [ 4, 13],
        [ 3, 14],
        [14, 16],
        [17, 18],
        [19, 20],
        [20, 22],
        [ 5,  1],
        [11,  6],
        [21, 16]], device='cuda:0'), tensor([[ 1,  2],
        [ 6,  7],
        [ 8,  9],
        [10, 11],
        [14, 15],
        [16, 17],
        [18, 19],
        [20, 21]], device='cuda:0'), tensor([], device='cuda:0', size=(0, 2), dtype=torch.int64)]
node_features = torch.ones((23,128))
"""


def get_data_b():
    num_nodes = 23
    adjacency_lists = [torch.tensor([[0, 1],
                                     [2, 3],
                                     [3, 4],
                                     [4, 5],
                                     [4, 6],
                                     [7, 8],
                                     [9, 10],
                                     [9, 12],
                                     [4, 13],
                                     [3, 14],
                                     [14, 16],
                                     [17, 18],
                                     [19, 20],
                                     [20, 22],
                                     [5, 1],
                                     [11, 6],
                                     [21, 16]], device='cuda:0'),
                       torch.tensor([[1, 2],
                                     [6, 7],
                                     [8, 9],
                                     [10, 11],
                                     [14, 15],
                                     [16, 17],
                                     [18, 19],
                                     [20, 21]], device='cuda:0')]
    edge_matrix = torch.concat(adjacency_lists, dim=0).cpu().numpy()
    node_features = torch.ones((23, 128))
    return num_nodes, node_features, edge_matrix


def get_data_a():
    node_features = torch.ones((8, 128))
    num_nodes = 8
    adjacency_lists = [torch.tensor([[0, 1],
                                     [0, 5],
                                     [1, 2],
                                     [2, 3],
                                     [3, 4],
                                     [4, 5],
                                     [5, 6],
                                     [6, 7]], device='cuda:0')]
    edge_matrix = torch.concat(adjacency_lists, dim=0).cpu().numpy()
    return num_nodes, node_features, edge_matrix


def get_paths_dict(num_nodes, edge_matrix):
    paths_dict = {i: [] for i in range(num_nodes)}
    for i in range(edge_matrix.shape[0]):
        entry = edge_matrix[i]
        paths_dict[entry[0]].append(entry[1])
        paths_dict[entry[1]].append(entry[0])
    return paths_dict


def dfs(path, paths_dict, seen, all_paths, length):
    if len(path) == length:
        return all_paths.append(copy.deepcopy(path))
    last_node = path[-1]
    for neighbor in paths_dict[last_node]:
        if neighbor not in seen:
            path.append(neighbor)
            seen.add(neighbor)
            dfs(path, paths_dict, seen, all_paths, length)
            path.pop()
            seen.remove(neighbor)


def get_all_paths(paths_dict, length):
    """Return a list of all paths of the specified length -- use DFS + backtracking."""
    num_nodes = len(paths_dict.keys())
    all_paths = []
    for node_id in range(num_nodes):
        dfs([node_id], paths_dict, {node_id}, all_paths, length)
    return all_paths


def paths_to_feature_paths(all_paths, node_embeddings):
    node_emb_paths = []
    for path in all_paths:
        path_embs_list = [node_embeddings[node] for node in path]
        node_emb_paths.append(np.stack(path_embs_list))
    return node_emb_paths


def get_all_paths_from_node(paths_dict, node, node_embeddings, length) -> List[np.array]:
    all_paths = []
    dfs([node], paths_dict, {node}, all_paths, length)
    return paths_to_feature_paths(all_paths, node_embeddings)


def graph_to_paths(num_nodes, node_features, edge_matrix, length) -> List[np.array]:
    paths_dict = get_paths_dict(num_nodes, edge_matrix)
    all_paths = get_all_paths(paths_dict, length)
    return paths_to_feature_paths(all_paths, node_features)


def path_convolutional_similarity(a, b, length):
    """With this implementation, regions of high connectivity are going to be over expressed.

    As we consider all paths in a, if there's a lot of branching at a vertex, this will give it more weight.

    Can reduce it to a per-node level so that we reduce_max along each node in the graph to remove this
    weighing."""
    num_nodes, node_features, edge_matrix = a
    paths_dict = get_paths_dict(num_nodes, edge_matrix)

    # List[np.array]
    b_all_paths = graph_to_paths(*b, length)

    output = np.zeros((num_nodes,), dtype=np.float32)
    for node_id in range(num_nodes):
        node_emb_paths = get_all_paths_from_node(paths_dict, node_id, node_features, length)
        for node_emb_path in node_emb_paths:
            for b_node_emb_path in b_all_paths:
                conv_output = np.sum(node_emb_path * b_node_emb_path) / length
                output[node_id] = max(output[node_id], conv_output)
    return np.sum(output)


def score_fn(a, b):
    rtn = 0
    for path_length in range(2, 9):
        rtn += path_convolutional_similarity(a, b, path_length)
    return rtn


if __name__ == '__main__':
    # a = get_data_a()
    # b = get_data_b()
    # print(score_fn(a, b))
    num_nodes, f, e = get_data_a()
    print(get_all_paths(get_paths_dict(8, e), 4))