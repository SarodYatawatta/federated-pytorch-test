# federated-pytorch-test
We train three CNN models (same as in the PyTorch tutorial) using CIFAR10 dataset. We use only 1/3 of the data for training each model. We also compare the peformance of federated averaging and consensus optimization in training the three models together, without sharing the data.

You might need to clone the [stochastic LBFGS](https://github.com/nlesc-dirac/pytorch) code to get these scripts running.
