# federated-pytorch-test
We train three CNN models using CIFAR10 dataset. In one example, the models are similar to the PyTorch tutorial and in another example, they are ResNet18.  In both cases, we use only 1/3 of the data for training each model. We also compare the performance of federated averaging and consensus optimization in training the three models together, without sharing the data.

The stochastic LBFGS optimizer is provided with the code. Further details are given [in this paper](https://ieeexplore.ieee.org/document/8755567). Also see [this introduction](http://sagecal.sourceforge.net/pytorch/index.html).
