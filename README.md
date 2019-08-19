# federated-pytorch-test
We train three CNN models (one is similar to the PyTorch tutorial and another is ResNet18) using CIFAR10 dataset. We use only 1/3 of the data for training each model. We also compare the peformance of federated averaging and consensus optimization in training the three models together, without sharing the data.

The stochastic LBFGS optimizer is provided. Details are given [in this paper](https://ieeexplore.ieee.org/document/8755567). Also see [this introduction](http://sagecal.sourceforge.net/pytorch/index.html).
