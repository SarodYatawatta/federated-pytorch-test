import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


# device for computing: CPU or GPU
global mydevice
########################################################################### helper functions
def init_weights(m):
  if type(m)==nn.Linear or type(m)==nn.Conv2d:
    torch.nn.init.xavier_uniform_(m.weight)
    if hasattr(m, 'bias'):
      if m.bias is not None:
        m.bias.data.fill_(0.01)

def unfreeze_one_layer(net,layer_id):
  ' set all layers to not-trainable except the layer given by layer_id (0,1,..)'
  for ci,param in enumerate(net.parameters(),0):
    if (ci == 2*layer_id) or (ci==2*layer_id+1):
       param.requires_grad=True
    else:
       param.requires_grad=False

def unfreeze_all_layers(net):
  ' reset all layers to trainable'
  for ci,param in enumerate(net.parameters(),0):
    param.requires_grad=True

def unfreeze_one_block(net,layers):
  ''' layers=[llow,lhigh]
    make all layers in 2*llow..2*lhigh-1 trainable
  '''
  llow=2*layers[0]
  lhigh=2*layers[1]-1
  for ci,param in enumerate(net.parameters(),0):
    if (ci >= llow) and (ci<=lhigh):
       param.requires_grad=True
    else:
       param.requires_grad=False

def get_trainable_values(net):
  ' return trainable parameter values as a vector (only the first parameter set)'
  trainable=filter(lambda p: p.requires_grad, net.parameters())
  paramlist=list(trainable) 
  N=0
  for params in paramlist:
    N+=params.numel()
  X=torch.empty(N,dtype=torch.float).to(mydevice)
  X.fill_(0.0)
  offset=0
  for params in paramlist:
    numel=params.numel()
    with torch.no_grad():
      X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
    offset+=numel

  return X

def put_trainable_values(net,X):
  ' replace trainable parameter values by the given vector (only the first parameter set)'
  trainable=filter(lambda p: p.requires_grad, net.parameters())
  paramlist=list(trainable)
  offset=0
  for params in paramlist:
    numel=params.numel()
    with torch.no_grad():
     params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
    offset+=numel

def number_of_layers(net):
  ' get total number of layers (note: each layers has weight and bias , so count as 2) '
  for ci,param in enumerate(net.parameters(),0):
   pass
  return int((ci+1)/2) # because weight+bias belong to one layer

def number_of_blocks(net):
  ' get total number of blocks of layers (for ResNet) '
  return len(net.upidx())

