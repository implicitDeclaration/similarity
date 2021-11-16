import torch
import os
import tqdm
import numpy as np
import networkx as nx
import heapq
from scipy.stats import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import data

cfgs_vgg = {
        'cvgg16_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
                          'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40'],
        }
cfgs_res = {
    'cres18': [2, 2, 2, 2],
    'cres34': [3, 4, 6, 3],
    'cres50': [3, 4, 6, 3],
    'cres101': [3, 4, 23, 3],
    'cres152': [3, 8, 36, 3],
    'ires18': [2, 2, 2, 2],
    'ires34': [3, 4, 6, 3],
    'ires50': [3, 4, 6, 3],
    'ires101': [3, 4, 23, 3],
    'ires152': [3, 8, 36, 3],
}

def get_inner_feature_for_vgg(model, hook, args):
    cfg = cfgs_vgg[args.arch]
    print('cfg:', cfg)
    count = 0
    for idx, m in enumerate(model.named_modules()):
        name, module = m[0], m[1]
        if count < len(cfg):
            if name == cfg[count]:
                print(module)
                handle = module.register_forward_hook(hook)
                count += 1
        else:
            break


def get_inner_feature_for_resnet(model, hook):
    
    cfg = cfgs_res["cres18"]
    print('cfg:', cfg)
    
    handle = model.conv1.register_forward_hook(hook)
    # handle.remove()  # free memory
    for i in range(cfg[0]):
        handle = model.layer1[i].register_forward_hook(hook)

    for i in range(cfg[1]):
        handle = model.layer2[i].register_forward_hook(hook)

    for i in range(cfg[2]):
        handle = model.layer3[i].register_forward_hook(hook)

    for i in range(cfg[3]):
        handle = model.layer4[i].register_forward_hook(hook)

def Cosine_Similarity(x, y):
    r'''
    Args:
        x: feature
        y: feature
    Returns: the similarity between x and y
    '''
    return torch.cosine_similarity(x, y)


def calculate_similarity(feat, topk, similarity_function='cosine'):
    r'''

    Args:
        feat: features extracted by pretrained CNN
        topk: topk samples
        similarity_function: which indicators to use

    Returns: similarity_matrix, edges_list

    '''
    edges_list = []
    n_samples = feat.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    # similarity_matrix -= np.eye(n_samples)  # to avoid edges less than expectation for using Pearson

    for i in range(n_samples):
        for j in range(n_samples):
            if i < j:
                if similarity_function == 'cosine':  # already change to gpu
                    similarity = Cosine_Similarity(feat[i].reshape(1, -1), feat[j].reshape(1, -1))
                elif similarity_function == 'Pearson':  # note that use Pearson will make edges less than expectation
                    similarity, _ = stats.pearsonr(feat[i].cpu().numpy(), feat[j].cpu().numpy())  # not right

                similarity_matrix[i][j] = similarity_matrix[j][i] = similarity

    for i in range(n_samples):  # note that: nodes start from 1
        k_indice = heapq.nlargest(topk, range(len(similarity_matrix[i])), similarity_matrix[i].take)

        for j in range(len(k_indice)):
            b = int(k_indice[j]+1)
            a = (int(i+1), b, float(similarity_matrix[i][k_indice[j]]))
            edges_list.append(a)

    return similarity_matrix, edges_list


def directly_draw_undirected_weighted_network(edges_list, whole_label, target, layer, batch_size):  # checked
    r'''

    Args:
        edges_list: direcetd weighted networks' edges list, a tuple like: (source node, target node, weight)
        whole_label: a list of all samples' ground true & predicted top5 labels
        target: ground true label
        layer: draw which layer

    Returns: undirected weighted network, the adj of undirected weighted network

    '''
    adj = np.zeros((batch_size, batch_size))
    
    # directed adj
    for ii in range(len(edges_list)):
        a, b = int(edges_list[ii][0] - 1), int(edges_list[ii][1] - 1)
        adj[a, b] = edges_list[ii][2]
    
    adj = (adj + adj.T) / 2
    
    undirected_weighted_network = nx.from_numpy_matrix(adj)
    
    # whole_label = whole_label.cpu().numpy()
    # for i in range(batch_size):
    #     undirected_weighted_network.nodes[i]['label'] = whole_label[i]

    return undirected_weighted_network, adj


def get_layer(arch, block):
    if block == 0:
        return 0
    sum = 0
    for i in range(block):
        sum += arch[i]
    return sum

def LSim(G1, G2):
    r'''

    Args:
        G1: undirected graph
        G2: undirected graph

    Returns: layer similarity, the similarity of nodes between layer a and b
    ref: 'Measuring similarity for clarifying layer difference in multiplex ad hoc duplex information networks' Page 3
    '''
    adj1 = nx.adjacency_matrix(G1).todense()
    adj2 = nx.adjacency_matrix(G2).todense()
    lsim = 0
    NSim = []
    nodes = nx.number_of_nodes(G1)
    for i in range(nodes):
        k_i_1 = adj1[i]
        k_i_2 = adj2[i]
        NSim_i = cosine_similarity(k_i_1, k_i_2)
        NSim.append(NSim_i)
        lsim += NSim_i / nodes

    return lsim, NSim


def sanity_check(model_graphs, args):
    model_graphs = model_graphs # get_graphs(graph_save, args)
    model_num = len(model_graphs)
    total_acc = []
    for i in range(model_num):
        for j in range(i+1, model_num):  # use i_th model to find the corresponding layer of the rest j models
            # above is the iteration of models, below is the iteration of layers
            current_layer = 0
            correct_cnt = 0
            for layer_i in model_graphs[i]:
                sim_of_layers = []
                for layer_j in model_graphs[j]:
                    sim_of_layers.append(LSim(layer_i, layer_j))
                found_layer = sim_of_layers.index(max(sim_of_layers))
                if found_layer == current_layer:
                    correct_cnt +=1
                current_layer += 1
            acc = correct_cnt/current_layer
            total_acc.append(acc)
            print("model {i} and model {j}, acc is {acc}".format(i=i, j=j, acc=acc))
    print("total acc is {ta}".format(ta=np.mean(total_acc)))
    return np.mean(total_acc)
