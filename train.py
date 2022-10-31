import os
import pathlib
import random
import time

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    get_fl_index,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
)
from utils.schedulers import get_policy
from args import args
import importlib
import models
import data


def main():
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # train model
    main_worker(args)


def main_worker(args):

    train, validate = get_trainer(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.stitch:
        top_model, botm_model, model = get_model(args)
    else:
        model = get_model(args)
    set_gpu(args, model)
    if args.pretrained:
        model = get_pretrained(args, model)
    data = get_dataset(args)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = get_optimizer(args, model)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    if args.evaluate:
        model.eval()
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.epochs
        )
        return

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    acc1, best_acc1 = None, 0

    device = 'cuda:%s' % args.gpu

    for epoch in range(args.epochs):
        lr_policy(epoch, iteration=None)

        cur_lr = get_lr(optimizer)
        # train for one epoch
        start_train = time.time()
        if args.stitch:
            fea_ind, layer_ind = get_fl_index(args.arch, args.stitch_loc)
            train_acc1, train_acc5 = train(botm_model, top_model, model, data.train_loader,
                                           args, fea_ind, layer_ind, epoch, device)
        else:
            train_acc1, train_acc5 = train(
                data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
            )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        if args.stitch:
            fea_ind, layer_ind = get_fl_index(args.arch, args.stitch_loc)
            acc1, acc5 = validate(botm_model, top_model, model, data.val_loader, args, fea_ind, layer_ind, device)
        else:
            acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best {best_acc1}, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                model.state_dict(), is_best, filename=ckpt_base_dir / f"epoch_{epoch}.state", save=save,)

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )
        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")
    if args.stitch:
        return trainer.train_stitch, trainer.validate_stitch
    else:
        return trainer.train, trainer.validate


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"
    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_model(args):
    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=10)

    if args.stitch:
        top_pretrained = torch.load(args.top)['state_dict'] # ***!!!
        botm_pretrained = torch.load(args.botm)
        top_model = models.__dict__[args.arch](num_classes=10)
        botm_model = models.__dict__[args.arch](num_classes=10)
        stitch_model = models.__dict__['ConvStitch'](args.stitch_ch, args.stitch_ch)

        top_model.load_state_dict(top_pretrained)
        botm_model.load_state_dict(botm_pretrained)
        if os.path.isfile(args.stitch_model):
            stitch_model.load_state_dict(args.stitch_model)
        set_gpu(args, botm_model)
        set_gpu(args, top_model)
        # set_gpu(args, stitch_model)
        return top_model, botm_model, stitch_model

    return model


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )
    cudnn.benchmark = True

    return model


def get_pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.gpu)),
        )# ['state_dict']

        load_weight = {}
        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if 'module.' in k:  # in case that model trained on multi-gpu
                k = k.replace('module.', '')

            if v.size() != model_state_dict[k].size():  # k not in model_state_dict or
                print("IGNORE:", k)
                continue
            load_weight[k] = v
        model_state_dict.update(load_weight)
        model.load_state_dict(model_state_dict)
    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))

    return model


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


if __name__ == "__main__":
    main()
