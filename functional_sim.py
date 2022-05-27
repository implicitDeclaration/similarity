


#  python main.py --gpu 6 --pretrained True --batch-size 200 --topk 5 --config configs/sanity_check/cnn_10.yaml
def func_sim(args):
    train, validate, modifier = get_trainer(args)
    data = get_dataset(args)
    criterion = nn.CrossEntropyLoss().cuda()
    archs = ['cresnet34', 'cresnet50', 'cresnet101', 'cresnet152', ] # 'cresnet18', 'cvgg11_bn', 'cvgg13_bn', 'cvgg16_bn', 'cvgg19_bn'
    info = {'set': 'cifar10', 'bs': 200, 'topk': 5, 'acc1': 0}
    for a in archs:
        args.arch = a
        model = get_model(args)
        model = set_gpu(args, model)
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        info['acc1'] = acc1
        save_dir = './graph_save/function'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        device = torch.device("cuda:{}".format(args.gpu))
        get_edge_func(model, data.val_loader, save_dir, device, arch=a, topk=5, degree=200)
        get_motif(save_dir, arch=a, log_dir='./', log_info=info)