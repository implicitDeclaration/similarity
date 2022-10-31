import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from args import args


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.len = images.shape[0]
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.len


class SynCIFAR10():
    def __init__(self, args):
        super(SynCIFAR10, self).__init__()
        #  use the original test set
        cifar_root = os.path.join(args.data, "cifar10")

        kwargs = {"num_workers": args.workers, "pin_memory": True} if torch.cuda.is_available() else {}
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        test_dataset = torchvision.datasets.CIFAR10(
            root=cifar_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )

        data = torch.load(args.syndata)['data']
        images, labels = data[-1]
        syn_data = TensorDataset(images, labels)
        self.train_loader = torch.utils.data.DataLoader(syn_data, batch_size=args.batch_size, shuffle=True, num_workers=0)


class CIFAR10:
    def __init__(self, args):
        super(CIFAR10, self).__init__()

        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )
