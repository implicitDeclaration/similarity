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
python sanity_check.py --config ./configs/resnet18.yaml 
```
## Model Stitching and Motif Experiment
To train stitch models, run
```bash
python train.py --stitch --top 'top model weight file path' --botm 'bottom model weight file path' --config 'config file' --gpu 0 --name 'the name of save dir'
```
e.g., --top './runs/cvgg16_bn/seed23/prune_rate=0.0/checkpoints/model_best.pth'

To run motif experiment, run
```bash
python functional_sim.py 
```



