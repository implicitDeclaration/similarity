import time
import torch
import tqdm
import numpy as np

from models.VGG_cifar import ConvStitch
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.design_for_hook import get_inner_feature_for_vgg

__all__ = ["train", "validate", "train_stitch", "validate_stitch"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    if args.stitch:
        botm_model, top_model = model[0], model[1]
        botm_model.eval()
        top_model.train()
    else:
        model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            device = 'cuda:%s' % args.gpu
            images = images.to(device)
            target = target.to(device)

        # compute output
        if args.stitch:
            mid_output = botm_model(images)
            output, _, _ = top_model(mid_output)
        else:
            output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                device = 'cuda:%s' % args.gpu
                images = images.to(device)
                target = target.to(device)
                #
                # images = images.cuda(args.gpu, non_blocking=True)
                # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg


def train_stitch(botm_model, top_model, stitch_model, train_loader, epoch, fea_ind, layer_ind, device):
    botm_model.eval()
    top_model.train()
    bottom_feature = []

    def hook(module, input, output):
        bottom_feature.append(output.clone().detach())
    get_inner_feature_for_vgg(botm_model, hook, arch='cvgg16_bn')  # get inner feature of the bottom model

    # stitch_model = ConvStitch(512, 512)
    stitch_model.to(device)
    optimizer = torch.optim.Adam(stitch_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    acc1_rec, acc5_rec = [], []

    for images, targets in train_loader:

        images, targets = images.to(device), targets.long().to(device)
        botm_output = botm_model(images)
        feature = stitch_model(bottom_feature[fea_ind])
       # print(bottom_feature[2].shape)
        top_output = top_model(feature, mid_input=layer_ind)

        loss = criterion(top_output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(top_output, targets, topk=(1, 5))
        acc1_rec.append(acc1)
        acc5_rec.append(acc5)
        bottom_feature = []  # clear features
    return np.mean(acc1_rec), np.mean(acc5_rec)
        #print('epoch : {}, top1 train acc : {}， top5 train acc : {}'.format(epoch, acc1, acc5))


def validate_stitch(botm_model, top_model, stitch_model, test_loader, fea_ind, layer_ind, device):
    botm_model.eval()
    top_model.eval()
    stitch_model.eval()
    bottom_feature = []
    def hook(module, input, output):
        bottom_feature.append(output.clone())

    get_inner_feature_for_vgg(botm_model, hook, arch='cvgg16_bn')  # get inner feature of the bottom model

    acc1_rec, acc5_rec = [], []
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        botm_output = botm_model(images)
        feature = stitch_model(bottom_feature[fea_ind])
        top_output = top_model(feature, mid_input=layer_ind)
        acc1, acc5 = accuracy(top_output, targets, topk=(1, 5))
        acc1_rec.append(acc1)
        acc5_rec.append(acc5)
        bottom_feature = []  # clear features

    return np.mean(acc1_rec), np.mean(acc5_rec)
    #print("overall accuracy is {}".format(np.mean(acc_all)))

