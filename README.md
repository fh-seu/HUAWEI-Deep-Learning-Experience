# HUAWEI-Deep-Learning-Experience-Hackathon
## HUAWEI Deep Learning Experience Hackathon, December 11-12, Stockholm, Sweden

### Datasets: 
- Dataset A: Unlabeled dataset with 100,000 unlabeled images
- Dataset B: Labeled dataset of 10 classes with 5,000 samples(4,000 training/1,000 testing).
- Dataset C: Labeled dataset of 10 classes with 5,000 samples(4,000 training/1,000 testing).
Tips: Dataset C is similar to Dataset B but with different classes! To be revealed during the evaluation phase.

### Challenge Description: 
How well can you generalize & learn from the unlabeled information you have?
Your goal is to exploit the unlabeled information and build a mechanism that is able to learn using as few labeled training samples as possible!

### Task Description:
You should show how well you can make use of the unlabeled data in order to build a classifier that can generalize well when trained with small labeled datasets. Your ultimate goal is to build and use a useful prior and build a classifier that maximize the evaluation criterion while trained and tested with dataset C. 

### Platfrom:
Hopsworks (Jupyter + Spark).

The repository just contains the codes of training the model with A and B, as C is used for the evaluation.

### Methods: 
1. Traditional Solution: Semi-supervised learning with ladder network
Ladder network is a deep learning algorithm that combines supervised and unsupervised learning. It was introduced in the paper Semi-Supervised Learning with Ladder Network by A Rasmus, H Valpola, M Honkala, M Berglund, and T Raiko (https://arxiv.org/pdf/1507.02672.pdf). The structure is linked by an autocoder and a ladder network / convolutional neural network. The code is modified from the implementation of rinuboney (https://github.com/rinuboney/ladder) and Robileo (https://github.com/Robileo/ladder_network). In this method, the accuracy rate is about 20%, and highest accuracy rate in this hackathon is about 35%, for the datasets are very weak.

2. Brute Force Solution: Resnet
The solution is inspired from the friend of one teammate in our group. It is a powerful tool to implement image classification! In this method, the accuracy rate is about 50%.

**Our team has four very great teammates!** 
