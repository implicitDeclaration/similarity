import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import pandas
import seaborn as sns
from scipy.stats import stats
from sklearn.metrics.pairwise import cosine_similarity
np.seterr(divide='ignore', invalid='ignore')  # solve mat 0 problem
import numpy as np

def calculate_characterization(G_list):
    r'''

    Args:
        G_list: undirected graahs

    Returns:  parms

    '''
    G_degree = []
    for i in range(len(G_list)):
        G_degree.append(nx.degree(G_list[i]))


def full_pattern_of_correlations(G1, G2):
    r'''

    Args:
        G1: undirected graph
        G2: undirected graph

    Returns: full pattern of correlations between G1 and G2, type: np.array
    ref: 'The structure and dynamics of multilayer networks' Page 21

    '''
    graph1 = []
    graph2 = []
    nodes1 = nx.number_of_nodes(G1)
    nodes2 = nx.number_of_nodes(G2)
    matrix = np.zeros((nodes1, nodes2))
    assert nodes1 == nodes2, 'G1、G2 must have same number of nodes'
    degree1 = nx.degree(G1)
    degree2 = nx.degree(G2)
    for _, data in enumerate(degree1):
        graph1.append(data[1])

    for _, data in enumerate(degree2):
        graph2.append(data[1])

    pairs = itertools.product(graph1, graph2)
    for pair in pairs:  # get position
        # print(pair[0], pair[1])
        matrix[pair[0]][pair[1]] += 1

    matrix = matrix / nodes1
    # print(matrix)
    return matrix


def Average_degree(G):
    r'''
    Args:
        G: undirected graph

    Returns: Average degree of G
    '''
    graph = []
    degree = nx.degree(G)
    nodes = nx.number_of_nodes(G)
    for _, data in enumerate(degree):
        graph.append(data[1])

    average_degree = (np.array(graph) / nodes).tolist()
    return average_degree


def change_networkx_degree_to_np_array(G):
    r'''

    Args:
        G: undirected graph

    Returns: degree (type: np.array)
    '''
    new_degree = []
    degree = nx.degree(G)
    for _, data in enumerate(degree):
        new_degree.append(data[1])

    new_degree = np.array(new_degree)
    return new_degree


def average_degree_of_a_conditioned_on_b(G1, G2):
    r'''

    Args:
        G1: undirected graph
        G2: undirected graph

    Returns: Average degree in layer a conditioned on the degree of the node in layer b
    ref: 'The structure and dynamics of multilayer networks' Page 21
    attention: output like this, [nan nan nan  3.  6.  5. nan nan nan nan]
    '''
    degree1 = change_networkx_degree_to_np_array(G1)
    degree1 = degree1[np.newaxis, ...]  # expand the dim 0 for mul
    P_ab = full_pattern_of_correlations(G1, G2)
    a = np.sum(degree1 * P_ab, axis=0)
    b = np.sum(P_ab, axis=0)
    # print(a, b)
    return a / b


def total_overlap(G1, G2):
    r'''

    Args:
        G1: undirected graph
        G2: undirected graph

    Returns: overlap between G1 and G2
    ref: 'The structure and dynamics of multilayer networks' Page 22

    '''
    adj1 = nx.adjacency_matrix(G1).todense()
    adj2 = nx.adjacency_matrix(G2).todense()
    O_ab = np.multiply(adj1, adj2)
    return O_ab


def local_overlap(G1, G2):
    r'''

    Args:
        G1: undirected graph
        G2: undirected graph

    Returns: the total number of neighbors of node i that are neighbors in both layer a and layer b, type: np.array
    ref: 'The structure and dynamics of multilayer networks' Page 22

    '''
    adj1 = nx.adjacency_matrix(G1).todense()
    adj2 = nx.adjacency_matrix(G2).todense()
    Oi_ab = np.sum(np.multiply(adj1, adj2), axis=1)
    return Oi_ab


def Pearson_correlation_coefficient(G1, G2):
    r'''

    Args:
        G1: undirected graph
        G2: undirected graph

    Returns: Pearson correlation coefficient between G1 and G2
    ref: 'The structure and dynamics of multilayer networks' Page 22
    0.8-1.0 very strong relevant
    0.6-0.8 strong relevant
    0.4-0.6 medium relevant
    0.2-0.4 weak relevant
    0.0-0.2 very weak relevant or not relevant
    '''
    degree1 = change_networkx_degree_to_np_array(G1)
    degree2 = change_networkx_degree_to_np_array(G2)
    Pearson, pvalue = stats.pearsonr(degree1, degree2)
    return Pearson, pvalue


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


def draw_layer_similarity_heatmap(g_list, args):
    r'''

    Args:
        g_list: undirected graph list,{G1, G2,... ,G_m}

    Returns: None
    ref: 'Measuring similarity for clarifying layer difference in multiplex ad hoc duplex information networks' Page 3
    '''
    num = len(g_list)
    matrix = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            if j > i:
                lsim, _ = LSim(g_list[i], g_list[j])
                matrix[i][j] = matrix[j][i] = lsim

    matrix = matrix + np.eye(num)
    fig = plt.figure()
    sns_plot = sns.heatmap(matrix, annot=False)
    if args.pretrained:
        fig.savefig("graph_save/{data}-{model}-pretrain.pdf".format(data=args.set, model=args.arch), bbox_inches='tight')
    else:
        fig.savefig("graph_save/{data}-{model}-rand.pdf".format(data=args.set, model=args.arch), bbox_inches='tight')

    print("saved!")


def get_layer(arch, block):
    if block == 0:
        return 0
    sum = 0
    for i in range(block):
        sum += arch[i]
    return sum


def similarity_advs(ad_list, layer, args):
    num_ofattacks = len(ad_list)
    sim_layers = np.zeros((num_ofattacks, num_ofattacks))
    for i in range(num_ofattacks):
        for j in range(num_ofattacks):
            if i < j:
                lsim, _ = LSim(ad_list[i][layer], ad_list[j][layer])
                sim_layers[i][j] = sim_layers[j][i] = lsim
    sim_layers = sim_layers + np.eye(num_ofattacks)
    print("similarity for layer: {l}".format(l=layer))
    print(sim_layers)
    return sim_layers

def similarity_per_layer(edges1, edges2, args):
    sim_layers = []
    for e1, e2 in zip(edges1, edges2):
        lsim, _ = LSim(e1, e2)
        sim_layers.append(float(lsim))
    print("similarity per layer: ")
    print(sim_layers)
    return sim_layers


def draw_block_similarity(g_listofmodels, listofarchs, args):
    '''
    draw the similarity between the first layers of each block
    :param : g_listA: undirected graph list,{G1, G2,... ,G_m}
    :param : a_arch: block size of the corresponding model, e.g., [2, 2, 2, 3]
    :return: None
    '''
    model_num = len(g_listofmodels)

    for layer in range(len(listofarchs[0])): # iter of different layers
        matrix = np.zeros((model_num, model_num))
        print("block : {bb}".format(bb=layer))
        xcnt = 0
        for imodel, iarch in zip(g_listofmodels, listofarchs): # iter of different models
            ycnt =0
            for jmodel, jarch in zip(g_listofmodels, listofarchs):
                if ycnt > xcnt:
                    lsim, _ = LSim(imodel[get_layer(iarch, layer)], jmodel[get_layer(jarch, layer)])
                    matrix[xcnt][ycnt] = matrix[ycnt][xcnt] = lsim
                    print("get similarity of model : {m1} and model : {m2} at block : {b}".format(m1=iarch, m2=jarch, b=layer))
                ycnt += 1
            xcnt += 1
        matrix = matrix + np.eye(model_num)
        fig = plt.figure()
        sns_plot = sns.heatmap(matrix, annot=False)
        cres = ['Res18', 'Res34', 'Res50', 'Res101', 'Res152']
        x_axis = [0.5, 1.5, 2.5, 3.5, 4.5]
        plt.xticks(x_axis, cres,)  # rotation=30,  fontproperties='Arial', size=12
        plt.yticks(x_axis, cres, )
        fig.savefig("./graph_save/{data}-{model}-block-{bl}.pdf".format(data=args.set, model='vgg' if 'vgg' in args.arch else 'res', bl=layer+1),
                        bbox_inches='tight')
        
def sanity_check(model, data):
    pass

    
    
if __name__ == "__main__":
    # graph_save = "./graph_save/graph_of_sanity/"
    # seed = 23
    # print(graph_save, "{init}-{arch}-{layer}-sd{seed}_ed_list.npy".format(
    #                                 init="init", arch="res", layer=1, seed=seed if seed is not None else "0"))
    #
    logfile = open(os.path.join('./', 'ablation_log.csv'), 'a')
    graph_N = [5, 10, 50, 100, 200, 300, 400, 500]
    top_k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    logfile.write("top-k, degree, acc, model-type\n")
    for k in top_k:
        
        k_acc = k+0.1
        logfile.write("{topk}, {deg}, {acc}, {model}\n".format(topk=k, deg=500, acc=k_acc, model='args.arch'))
        logfile.flush()
    pass
