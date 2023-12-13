# CS260D-project Investigation of SAS for imbalanced data

- Linqiao Jiang
- Zixian Li
- Yihao Qin
- Chang Xie

## Description

This is the code of our 260d project. Our project is "Investigation of SAS for imbalanced data".


## Environments

This is my experiment eviroument
- python version: 3.10.13
- pytorch version: 2.1.1+cu118

## Jupyter notebook description
1. Final_Project_balanced.ipynb: 

This jupyter notbook includes loading balanced CIFAR100 data, choosing subset from the balanced data using SAS, and the training result for both fullset and subset data.

2. Final_Project_imbalanced.ipynb:

This jupyter notbook includes constructing imbalanced CIFAR100 data, choosing subset from the imbalanced data using SAS, and the training result for both fullset and subset data.

3. Final_Project_solution.ipynb:

This jupyter notbook includes our proposed solution1(MDP) and solution2(MDA) for upsampling the imbalanced data, choosing subset from the upsampling data using SAS, and the training result for both fullset and subset data.

## Usage

### 1. enter directory and install SAS package
```bash
$ cd SAS-IMBALANCEDDATA
$ pip install sas-pip/
```

### 2. dataset
We use cifar100 dataset from torchvision since it's more convenient to construct an imbalanced dataset and train on it, the sample code for writing own dataset module could be seen in dataset.py, as an example for people don't know how to write it.


### 3. run tensorbard(optional)
Install tensorboard
```bash
$ pip install tensorboard
$ tensorboard --logdir=runs
```

### 4. train the model
The training pipeline could be seen in train_changedata.py. It includes the whole process for training and testing pipeline. In order to apply the training in jupyter notebook, I packaged the training pipeline into function main(), then we could call main() with this format in python. 

```bash
#python
from train_changedata import main
best_acc, confusion_matrix , best_f1, best_recall = main(train_dataset= new_cifar)
```

The function will return best accuracy, confusion matrix, f1 score and recall score for the model.

Besides, I also keep the train.py as reference. 

