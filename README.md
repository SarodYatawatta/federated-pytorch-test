# federated-pytorch-test
We train CNN models __without__ having access to the full dataset. The CIFAR10 dataset is used in all examples. The CNN models can be chosen from simpler models similar to PyTorch or Tensorflow demos and also ResNet18.  In all cases, we use only 1/K (where K is user defined) of the data for training each CNN model. We also compare the performance of federated averaging and consensus optimization in training the K models, without sharing the training data between models. Note that we only pass __a subset of parameters__ between the models, unlike in normal federated averaging or consensus. This __reduces the bandwidth__ required enormously! 

The stochastic LBFGS optimizer is provided with the code. Further details are given [in this paper](https://ieeexplore.ieee.org/document/8755567). Also see [this introduction](http://sagecal.sourceforge.net/pytorch/index.html).

GPU acceleration is enabled when available, set ```use_cuda=True```.
Files included are:

``` lbfgsnew.py ```: New LBFGS optimizer

``` simple_models.py ```: Relatively simple CNN models for CIFAR10, derived from PyTorch/Tensorflow demos, also ResNet18

``` no_consensus_trio.py ```: Train 3 models using 1/3 of the training data for each model

``` federated_multi.py ```: Train K models using 1/K of the data, with federated averaging, K can be varied

``` consensus_multi.py ```: Train K models using 1/K of the data, with consensus optimization (adaptive) ADMM, K can be varied

``` federated_vae.py ```: Train K variational autoencoders, using federated averaging

``` federated_vae_cl.py ```: Train K variational autoencoders for clustering, using federated averaging, based on [this paper](https://arxiv.org/abs/2005.04613).

``` federated_cpc.py ```: Train K models using contrastive predictive coding, using LOFAR data, based on [this paper](https://arxiv.org/abs/1905.09272).
