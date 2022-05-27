import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
from args import args

import tqdm
from utils.logging import AverageMeter, ProgressMeter
import torch
import time


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def imagenet_validate(val_loader, model, criterion, args):
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

        for i, data in tqdm.tqdm(
                enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            images = data[0]["data"].cuda(non_blocking=True)
            target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
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
    return top1.avg, top5.avg


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.readers.File(file_root=data_dir, shard_id=0, num_shards=1, random_shuffle=True)
        self.decode = ops.decoders.Image(device="mixed")
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.random.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)

        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.readers.File(file_root=data_dir, shard_id=0, num_shards=1, random_shuffle=True)  # default: False for test, True for modularity
        self.decode = ops.decoders.Image(device="mixed")
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, crop, val_size=256):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                    data_dir=image_dir + '/train',
                                    crop=crop)
        pip_train.build()
        print(f'pip_train.epoch_size("Reader"):{pip_train.epoch_size("Reader")}')
        dali_iter_train = DALIClassificationIterator(
            pip_train,
            size=pip_train.epoch_size("Reader")
        )
        return dali_iter_train
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                data_dir=image_dir + '/val',
                                crop=crop, size=val_size)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(
            pip_val,
            size=pip_val.epoch_size("Reader")
        )

        return dali_iter_val


class ImageNetDali:
    def __init__(self, args):
        super(ImageNetDali, self).__init__()
        self.train_loader = get_imagenet_iter_dali(
            type='train',
            image_dir='/public/ly/ImageNet50/',
            batch_size=args.batch_size,
            num_threads=16,
            crop=224,
            device_id=args.multigpu[0]
        )
        self.val_loader = get_imagenet_iter_dali(
            type='val',
            image_dir='/public/ly/ImageNet50/',
            batch_size=args.batch_size,
            num_threads=16,
            crop=224,
            device_id=args.multigpu[0]
        )
        # '/public/MountData/dataset/ImageNet50/'
        # /public/ly/dataset/small_ImageNet_1/
        # /public/ly/Interpretable_CNN/dataset/
        # /public/xjy/cv/data'
