import os
import torch
import torch.nn as nn
import numpy as np
import tqdm
import heapq
import networkx as nx

from metric.motifs import ESU_BFS
from train import get_model, get_dataset, set_gpu, get_pretrained
from args import args
from utils.net_utils import *
from utils.design_for_hook import *
import importlib
import models
import data
import trainers.default as trainer

def motif_stitch_acc(top_model, botm_model, stitch_model, fea_ind, layer_ind,):
    data = get_dataset(args)
    criterion = nn.CrossEntropyLoss().cuda()
    if args.stitch:
        top_model, botm_model, model = get_model(args)
    else:
        model = get_model(args)
    #  validate the accuracy of top model and bottom model
    ttop1, ttop5 = trainer.validate(data.val_loader, top_model, criterion, args, None, -1)
    btop1, btop5 = trainer.validate(data.val_loader, botm_model, criterion, args, None, -1)
    print('top model acc : {ttop1} bottom model acc : {btop1}')
    fea_ind, layer_ind = get_fl_index(args.arch, args.stitch_loc)
    acc1, acc5 = trainer.validate(botm_model, top_model, model, data.val_loader, args, fea_ind, layer_ind, device='cuda:%s' % args.gpu)
    print(f'stitch acc is {acc1}')


def func_sim(args):
    data = get_dataset(args)
    criterion = nn.CrossEntropyLoss().cuda()
    # archs = ['cresnet18', 'cresnet34', 'cresnet50', 'cresnet101', 'cresnet152', 'cvgg11_bn', 'cvgg13_bn', 'cvgg16_bn', 'cvgg19_bn']
    info = {'set': 'cifar10', 'bs': args.batch_size, 'topk': args.topk, 'acc1': 0}
    # for a in archs:
    #     args.arch = a
    model = get_model(args)
    if not args.pretrained:
        raise ValueError('you need assign a pretrained model')
    model = get_pretrained(args, model)
    model = set_gpu(args, model)

    save_dir = './graph_save/function'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    device = torch.device("cuda:{}".format(args.gpu))
    get_edg_list(model, data.val_loader, save_dir, device, arch=args.arch, topk=args.topk, degree=args.batch_size)
    get_motif(save_dir, arch=args.arch, log_dir='./', log_info=info)


def get_edg_list(model, val_loader, graph_save, device, arch, topk=5, degree=200):
    if not os.path.exists(graph_save):
        os.makedirs(graph_save)

    for i, (images, target) in tqdm.tqdm(enumerate(val_loader), ascii=True, total=len(val_loader)):
        inter_feature = []

        def hook(module, input, output):
            inter_feature.append(output.clone().detach())

        model.eval()
        images = images.to(device)
        target = target.to(device)
        with torch.no_grad():
            if 'res' in arch.lower():
                get_inner_feature_for_resnet(model, hook, arch=arch.lower())
            if 'vgg' in arch.lower():
                get_inner_feature_for_vgg(model, hook, arch=arch.lower())
            else:
                get_inner_feature_for_cnn(model, hook)

            output = model(images)
            label_check = os.path.join(graph_save, "{arch}-N{de}_label.npy".format(arch=arch, de=degree))
            np.save(label_check, target.cpu().detach().numpy())
            for m in range(len(inter_feature)):
                if len(inter_feature[m].shape) != 2:
                    inter_feature[m] = inter_feature[m].reshape(degree, -1)
                file_check = os.path.join(graph_save,
                        "{arch}-{layer}-top{k}-N{de}_edge.npy".format(arch=arch, layer=m, k=topk, de=degree))

                if os.path.exists(file_check):
                    print(file_check + " ****exist skip**** !")
                    continue
                else:
                    similarity_matrix, edges_list = calculate_cosine_similarity_matrix(inter_feature[m], topk)
                    np.save(file_check, edges_list)
                    print(file_check + " saved !")
        break  # one batch is enough


def graph_from_edges(edges_list, labels, batch_size):  # checked
    r'''

    Args:
        edges_list: direcetd weighted networks' edges list, a tuple like: (source node, target node, weight)
        whole_label: a list of all samples' ground true & predicted top5 labels
        label: ground true label
    Returns: undirected weighted network, the adj of undirected weighted network

    '''
    adj = np.zeros((batch_size, batch_size))

    # directed adj
    for ii in range(len(edges_list)):
        a, b = int(edges_list[ii][0] - 1), int(edges_list[ii][1] - 1)
        adj[a, b] = edges_list[ii][2]

    adj = (adj + adj.T) / 2

    undirected_weighted_network = nx.from_numpy_matrix(adj)
    for i in range(batch_size):
        undirected_weighted_network.nodes[i]['label'] = labels[i]

    return undirected_weighted_network, adj


def calculate_cosine_similarity_matrix(h_emb, topk, eps=1e-8):
    r'''
        h_emb: (N, M) hidden representations
    '''
    # normalize
    edges_list = []
    h_emb = torch.tensor(h_emb)
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    sim_matrix = sim_matrix.cpu().numpy()
    row, col = np.diag_indices_from(sim_matrix)  # Do not consider self-similarity
    sim_matrix[row, col] = 0
    n_samples = h_emb.shape[0]

    for i in range(n_samples):  # note that: nodes start from 1
        k_indice = heapq.nlargest(topk, range(len(sim_matrix[i])), sim_matrix[i].take)
        for j in range(len(k_indice)):
            b = int(k_indice[j]+1)
            a = (int(i+1), b, float(sim_matrix[i][k_indice[j]]))
            edges_list.append(a)
    return sim_matrix, edges_list


def get_motif(graph_save, arch, log_dir, log_info):
    '''

    :param graph_save: where saves graphs file
    :param arch: model structure
    :param log_dir: where to save log file
    :param log_info: dict e.g., {'set': 'cifar10', 'bs': 128, 'topk': 5, 'acc1': 00.0}
    :return:
    '''
    archl = arch.lower()
    if 'res' in archl:
        layer_num = sum(cfgs[archl])
    elif 'vgg' in archl:
        layer_num = len(cfgs[archl])
    file_path = os.path.join(log_dir, 'motif_log.csv')
    logfile = open(file_path, 'a')  # vgg
    logfile.flush()
    if os.path.getsize(file_path) == 0:
        logfile.write('arch, set, batch_size, topk, layer, acc1, type1, type2, type3\n')
        logfile.flush()
    label_check = os.path.join(graph_save, "{arch}-N{de}_label.npy".format(arch=arch, de=log_info['bs']))
    label = np.load(label_check)
    for m in range(layer_num):
        file_check = os.path.join(graph_save,
                                  "{arch}-{layer}-top{k}-N{de}_edge.npy".format(arch=arch, layer=m,
                                                                                k=log_info['topk'], de=log_info['bs']))
        edge = np.load(file_check)
        graph, adj = graph_from_edges(edge, label, log_info['bs'])
        motif = ESU_BFS(adj, label)
        logfile.write('{arch}, {set}, {batch_size}, {topk}, {layer}, {acc1}, {type1}, {type2}, {type3}\n'.format(
            arch=arch, set=log_info['set'], batch_size=log_info['bs'], topk=log_info['topk'], layer=m, acc1=log_info['acc1'],
            type1=motif.type1, type2=motif.type2, type3=motif.type3
        ))
        logfile.flush()
    print(f'result record into {file_path}')


if __name__ == '__main__':
    func_sim(args)
