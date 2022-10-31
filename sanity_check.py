import os
import tqdm
import numpy as np
import torch
from args import args
from train import get_model, set_gpu, get_pretrained, get_dataset
from metric.CKA import get_features, linear_CKA
from metric.GBS import get_graphs, LSim
from utils.design_for_hook import get_inner_feature_for_cnn, get_inner_feature_for_resnet, get_inner_feature_for_vgg

'''python sanity_check.py --config ./configs/resnet18.yaml --topk 500 --batch-size 500  --data /public/ly/czh/hidden-networks-master/dataset --gpu 4'''
def sanity_check(args):
    if 'vgg' in args.arch.lower():
        my_seeds = [23, 24, 26, 257, 277, 287, 298, 300, 31, 32]
    else:
        my_seeds = np.arange(23, 33)

    data = get_dataset(args)
    model = get_model(args)
    model = set_gpu(args, model)

    for seed in my_seeds:
        args.pretrained = './runs/{}/seed{}/prune_rate=0.0/checkpoints/model_best.pth'.format(args.arch, seed)
        model = get_pretrained(args, model)
        args.seed = seed
        get_edge(model, data, args)

    if args.metric == 'cka':
        features = get_features(args)
    else:
        features = get_graphs(args)
    sanity_check_acc(features)


def sanity_check_acc(model_graphs):
    # inter_feature [model1, model2, model3, ...]
    model_num = len(model_graphs)
    total_acc = []
    for i in range(model_num):
        for j in range(i + 1, model_num):  # use i_th model to find the corresponding layer of the rest j models
            # above is the iteration of models, below is the iteration of layers
            current_layer = 0
            correct_cnt = 0
            for layer_i in model_graphs[i]:
                sim_of_layers = []
                for layer_j in model_graphs[j]:
                    if args.metric == 'cka':
                        sim_of_layers.append(linear_CKA(layer_i, layer_j))
                    else:
                        sim_of_layers.append(LSim(layer_i, layer_j))
                found_layer = sim_of_layers.index(max(sim_of_layers))
                if found_layer == current_layer:
                    correct_cnt += 1
                current_layer += 1
            acc = correct_cnt / current_layer
            total_acc.append(acc)
            print("model {i} and model {j}, acc is {acc}".format(i=i, j=j, acc=acc))
    print("total acc is {ta}".format(ta=np.mean(total_acc)))
    return np.mean(total_acc)


def get_edge(model, data, args):
    if not os.path.exists(args.feature_save):
        os.makedirs(args.feature_save)

    inter_feature = []

    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    for i, (images, target) in tqdm.tqdm(enumerate(data.val_loader), ascii=True, total=len(data.val_loader)):
        model.eval()
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            if 'vgg' in args.arch.lower():
                get_inner_feature_for_vgg(model, hook, args.arch)
            elif 'cnn' in args.arch.lower():
                get_inner_feature_for_cnn(model, hook)
            else:
                get_inner_feature_for_resnet(model, hook, args.arch)
            output = model(images)
            label_check = os.path.join(args.feature_save, "{arch}-N{de}_label.npy".format(arch=args.arch, de=args.batch_size))
            np.save(label_check, target.cpu().detach().numpy())
            for m in range(len(inter_feature)):
                if len(inter_feature[m].shape) != 2:
                    if args.metric == 'cka':
                        inter_feature[m] = np.mean(inter_feature[m].cpu().numpy(), axis=(2, 3))
                    else:
                        inter_feature[m] = inter_feature[m].reshape(args.batch_size, -1)
                  # save features for further experiments
                file_check = os.path.join(args.feature_save, "{arch}-{layer}-sd{seed}-N{de}_{metric}.npy".format(
                     arch=args.arch, layer=m, seed=args.seed, de=args.batch_size, metric=args.metric))

                np.save(file_check, inter_feature[m].cpu().detach().numpy())
                print(file_check + " saved !")
        break  # one batch is enough

    return


if __name__ == "__main__":
    print('sanity check ')
    sanity_check(args)
