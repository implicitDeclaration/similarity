import os
import numpy as np
import torch
import tqdm


cfgs = {
        'cResNet18': [2, 2, 2, 2],
        'cvgg16_bn': ['features.0', 'features.2', 'features.5', 'features.7', 'features.10',
           'features.12', 'features.14', 'features.17', 'features.19',
           'features.21', 'features.24', 'features.26', 'features.28'],
        }

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def get_inner_feature_for_resnet(model, hook, arch):
    cfg = cfgs[arch]
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


def get_inner_feature_for_vgg(model, hook, arch):
    cfg = ['features.0', 'features.2', 'features.5', 'features.7', 'features.10',
           'features.12', 'features.14', 'features.17', 'features.19',
           'features.21', 'features.24', 'features.26', 'features.28']
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

def CKAget_edge_list(model, data, graph_save, arch, args, topk=2, degree=500):  # for one model
    if not os.path.exists(graph_save):
        os.makedirs(graph_save)

    inter_feature = []

    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    for i, (images, target) in tqdm.tqdm(enumerate(data.val_loader), ascii=True, total=len(data.val_loader)):
        model.eval()
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            if 'vgg' in args.arch:
                print("start getting edglist")
                get_inner_feature_for_vgg(model, hook, arch)
            else:
                get_inner_feature_for_resnet(model, hook, arch)
            output = model(images)
            _, pred = output.topk(5, 1, True, True)  # return (value, index)

            for m in range(len(inter_feature)):
                if len(inter_feature[m].shape) != 2:
                    # print("shape")
                    # print(inter_feature[m].shape)
                    inter_feature[m] = np.mean(inter_feature[m].cpu().numpy(), axis=(2, 3))
                    
                file_check = os.path.join(graph_save, "{init}-{arch}-{layer}-sd{seed}-top{k}-N{de}_feature.npy".format(
                    init=args.init, arch=args.arch, layer=m, seed=args.seed if args.seed is not None else "0", k=topk,
                    de=degree))
                if os.path.exists(file_check):
                    print(file_check + " ****exist *** !")
                    continue
                else:
                    np.save(file_check, inter_feature[m])
                    print(file_check + " saved !")
        break  # one batch is enough
        
    return inter_feature


def get_features(graph_save, args, topk=2, degree=500):
    model_graphs = []
    whole_label = torch.rand(500, 1)
    if 'vgg' in args.arch:
        my_seeds = [23, 24, 26, 256, 277, 287, 298, 300, 31, 32]
        model_size = 13
    else:
        my_seeds = range(23, 33)
        model_size = 9  # res18
    feature_models = []
    for s in my_seeds:
        one_model = []
        for m in range(model_size):
            file_check = os.path.join(graph_save, "{init}-{arch}-{layer}-sd{seed}-top{k}-N{de}_feature.npy".format(
                init=args.init, arch=args.arch, layer=m, seed=s, k=topk, de=degree))
            
            features = np.load(file_check)
            one_model.append(features)
        feature_models.append(one_model)
    return feature_models


def CKAsanity_check_cka(inter_feature, args):
    # inter_feature [model1, model2, model3, ...]
    model_num = len(inter_feature)
    layer_num = len(inter_feature[0])
    total_acc = []
    for i in range(model_num):
        for j in range(model_num):
            current_layer = 0
            correct_cnt = 0
            if i < j:
                for k in range(layer_num):
                    sim_of_layers = []
                    for l in range(layer_num):
                        sim_of_layers.append(linear_CKA(inter_feature[i][k], inter_feature[j][l]))
                    found_layer = sim_of_layers.index(max(sim_of_layers))
                    if found_layer == current_layer:
                        correct_cnt +=1
                    current_layer += 1
                acc = correct_cnt/current_layer
                total_acc.append(acc)
                print("model {i} and model {j}, acc is {acc}".format(i=i, j=j, acc=acc))
    print("total acc is {ta}".format(ta=np.mean(total_acc)))
    return total_acc



if __name__=='__main__':
    X = np.random.randn(2, 100, 64, 64)
    xx = np.mean(X, axis=(2, 3))
    print(xx.shape)
    Y = np.random.randn(100, 64)

    # print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    # print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))
