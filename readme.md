# Revisit Similarity of Neural Network Representations From Graph Perspective


Paper: https://arxiv.org/pdf/2111.11165.pdf
## Setup

1. Set up a virtualenv with python 3.7.4. 
2. Run ```pip install -r requirements.txt``` to get requirements
3. Create a data directory as a base for all datasets. For example, if your base directory is ```./datasets``` CIFAR10 would be located at ```./datasets/cifar10```.


## Sanity Check Experiment 
The code of training models of different random seeds refers to **What's hidden in a randomly weighted neural network?**

To train models of different random seeds, run
```bash
sh train_multi_seeds.sh
```
Then, run 
```bash
python sanity_check.py
```
## Model Stitching and Motif Experiment
Will be updated later




