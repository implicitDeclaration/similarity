import argparse
import sys
import yaml

from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # General Config
    parser.add_argument(
        "--data", help="path to dataset base directory", default="/mnt/disk1/datasets"
    )
    parser.add_argument(
        "--syndata", help="path to synthetic dataset file"
    )
    parser.add_argument(
        "--save", help="path to save ablation log", default="./"
    )
    parser.add_argument(
        "--feature_save", help="path to save features", default='./feature_save'
    )
    parser.add_argument(
        "--metric", help="use which metric to calculate similarity", default="gbs"
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="cvgg16_bn", help="model architecture"
    )
    parser.add_argument(
        "--stitch", action='store_true', help="if train a stitch layer"
    )
    parser.add_argument(
        "--stitch_ch", type=int, help="channel num of stitch layer, need to match the layer you choose"
    )
    parser.add_argument(
        "--stitch_loc", type=int, help="the layer or block to stitch"
    )
    parser.add_argument(
        "--stitch_model", type=int, help="the weight path of stitch model"
    )
    parser.add_argument(
        "--top", default=None, help="top model weight file"
    )
    parser.add_argument(
        "--botm", default=None, help="botm model weight file"
    )

    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--topk", help="topk in cka and gbs, default -1, i.e., fully connected", type=int, default=-1
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--warmup_length", default=0, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=str,
        help="use pre-trained model",
    )
    
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="Which GPUs to use for multigpu training",
    )
    # Learning Rate Policy Specific
    parser.add_argument(
        "--lr-policy", default="constant_lr", help="Policy for the learning rate."
    )

    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--save_every", default=-1, type=int, help="Save every ___ epochs"
    )
    parser.add_argument(
        "--prune-rate",
        default=0.0,
        help="Amount of pruning to do during sparse training",
        type=float,
    )
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )

    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )

    parser.add_argument(
        "--trainer", type=str, default="default", help="use which trainer file, all is defined in the default.py file"
    )


    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
