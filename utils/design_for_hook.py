# from model.VGG_cifar import *
# from model.VGG_ImageNet import *

cfgs = {
    'cresnet18': [2, 2, 2, 2],
    'cresnet34': [3, 4, 6, 3],
    'cresnet50': [3, 4, 6, 3],
    'cresnet101': [3, 4, 23, 3],
    'cresnet152': [3, 8, 36, 3],
    'ires18': [2, 2, 2, 2],
    'ires34': [3, 4, 6, 3],
    'ires50': [3, 4, 6, 3],
    'ires101': [3, 4, 23, 3],
    'ires152': [3, 8, 36, 3],
    'cvgg11': ['features.0', 'features.3', 'features.6', 'features.8', 'features.11', 'features.13', 'features.16', 'features.18'],
    'cvgg13': ['features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12', 'features.15', 'features.17', 'features.20', 'features.22'],
    'cvgg16': ['features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12', 'features.14', 'features.17', 'features.19', 'features.21', 'features.24', 'features.26', 'features.28'],
    'cvgg19': ['features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12','features.14', 'features.16', 'features.19', 'features.21', 'features.23', 'features.25', 'features.28', 'features.30', 'features.32', 'features.34'],
    'ivgg11': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22', 'features.25'],
    'ivgg13': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.21', 'features.24', 'features.28', 'features.31'],
    'ivgg16': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20', 'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40'],
    'ivgg19': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17','features.20', 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40', 'features.43', 'features.46', 'features.49'],
    'cvgg11_bn': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22',
               'features.25'],
    'cvgg13_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.21',
               'features.24', 'features.28', 'features.31'],
    'cvgg16_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
               'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40'],
    'cvgg19_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
               'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40', 'features.43',
               'features.46', 'features.49'],
}


def get_inner_feature_for_cnn(model, hook):
    cfg = ['layers.0', 'layers.3', 'layers.6', 'layers.9', 'layers.12', 'layers.15',
           'layers.18', 'layers.21', ]
    print('cfg:', cfg)
    count = 0
    for idx, m in enumerate(model.named_modules()):
        name, module = m[0], m[1]
        if count < len(cfg):
            if name == cfg[count]:
                handle = module.register_forward_hook(hook)
                count += 1
        else:
            break


def get_inner_feature_for_resnet(model, hook, arch):
    cfg = cfgs[arch.lower()]
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
    cfg = cfgs[arch.lower()]
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


if __name__ == "__main__":
    pass
    # demo
    import torch
    # input = torch.randn((2, 3, 32, 32))
    # inter_feature = []
    # model = cvgg19_bn(10)
    # # model = cvgg16_bn(10)
    # # model = cvgg13_bn(10)
    # # model = cvgg11_bn(10)
    # # model = vgg11_bn()
    # # model = vgg13_bn()
    # # model = vgg16_bn()
    # # model = vgg19_bn()
    # print(model)
    #
    #
    # def hook(module, input, output):
    #     inter_feature.append(output.clone().detach())
    # get_inner_feature_for_vgg(model, hook, 'cvgg19')
    # model(input)
