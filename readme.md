# Revisit Similarity of Neural Network Representations From Graph Perspective


Paper: https://arxiv.org/pdf/2111.11165.pdf
## Setup

1. Set up a virtualenv with python 3.7.4. 
2. Run ```pip install -r requirements.txt``` to get requirements
3. Create a data directory as a base for all datasets. For example, if your base directory is ```./datasets``` CIFAR10 would be located at ```./datasets/cifar10```. Specify the database with arg ```--data```


## Sanity Check Experiment 
The code of training models of different random seeds refers to **What's hidden in a randomly weighted neural network?**

To train models of different archs and random seeds, run
```bash
sh train_multi_seeds.sh
```
Then, e.g., run sanity check for ResNet models, use
```bash
python sanity_check.py --config ./configs/resnet18.yaml --topk 500 --batch-size 500 
```
## Model Stitching and Motif Experiment
To train stitch models, run
```bash
python train.py --stitch --stitch_ch 256 --stitch_loc 3 --top <top model weight file path> --botm <bottom model weight file path> --config <config file> --gpu 0 --name <the name of save dir> --seed 2023 
```
e.g., --top './runs/resnet18/seed23/prune_rate=0.0/checkpoints/model_best.pth'

To calculate the motif of a layer, run
```bash
python python functional_sim.py --gpu 0 --pretrained <you model file> --batch-size 500 --topk 5 --config configs/resnet18.yaml
```



