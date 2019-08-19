# federated-pytorch-test
We train three CNN models using CIFAR10 dataset. In one example, the models are similar to the PyTorch tutorial and in another example, they are ResNet18.  In both cases, we use only 1/3 of the data for training each model. We also compare the performance of federated averaging and consensus optimization in training the three models together, without sharing the data.

The stochastic LBFGS optimizer is provided with the code. Further details are given [in this paper](https://ieeexplore.ieee.org/document/8755567). Also see [this introduction](http://sagecal.sourceforge.net/pytorch/index.html).

Files included are:

``` lbfgsnew.py ```: New LBFGS optimizer

``` no_consensus_trio.py ```: Train 3 models using 1/3 of the training data for each model

``` federated_trio.py ```: Train 3 models using 1/3 of the data, but with federated averaging

``` consensus_admm_trio.py ```: Train 3 models using 1/3 of the data, but with consensus optimization

``` federated_trio_resnet.py ```: Train 3 ResNet18 models using 1/3 of the data, but with federated averaging

``` consensus_admm_trio_resnet.py ```: Train 3 ResNet18 models using 1/3 of the data, but with consensus optimization
