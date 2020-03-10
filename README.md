# federated-pytorch-test
We train CNN models __without__ having access to the full dataset. The CIFAR10 dataset is used in all examples. The CNN models can be chosen from simpler models similar to PyTorch or Tensorflow demos and in another example, they are ResNet18.  In all cases, we use only 1/3 of the data for training each CNN model. We also compare the performance of federated averaging and consensus optimization in training the three models, without sharing the training data between models. Note that we only pass __a subset of parameters__ between the models, unlike in normal federated averaging or consensus. This __reduces the bandwidth__ required enormously! 

The stochastic LBFGS optimizer is provided with the code. Further details are given [in this paper](https://ieeexplore.ieee.org/document/8755567). Also see [this introduction](http://sagecal.sourceforge.net/pytorch/index.html).

GPU acceleration is enabled when available, set ```use_cuda=True```.
Files included are:

``` lbfgsnew.py ```: New LBFGS optimizer

``` simple_models.py ```: Relatively simple CNN models for CIFAR10, derived from PyTorch/Tensorflow demos

``` no_consensus_trio.py ```: Train 3 models using 1/3 of the training data for each model

``` federated_trio.py ```: Train 3 models using 1/3 of the data, but with federated averaging

``` federated_multi.py ```: Train K models using 1/K of the data, with federated averaging, K can be varied

``` consensus_admm_trio.py ```: Train 3 models using 1/3 of the data, but with consensus optimization

``` federated_trio_resnet.py ```: Train 3 ResNet18 models using 1/3 of the data, but with federated averaging

``` consensus_admm_trio_resnet.py ```: Train 3 ResNet18 models using 1/3 of the data, but with consensus optimization

