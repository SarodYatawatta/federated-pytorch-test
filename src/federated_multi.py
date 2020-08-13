import torch
import torchvision
import torchvision.transforms as transforms

import math
import time

# How many models (==slaves)
K=10
# train K models by Federated learning
# each iteration over a subset of parameters: 1) average 2) pass back average to slaves 3) SGD step
# initialize with pre-trained models (better to use common initialization)
# loop order: loop 0: parameters/layers   {
#               loop 1 : {  averaging (part of the model)
#                loop 2: { epochs/databatches  { train; } } } }
# repeat this Nloop times


torch.manual_seed(69)
# minibatch size
default_batch=128 # no. of batches per model is (50000/K)/default_batch
Nloop=12 # how many loops over the whole network
Nepoch=1 # how many epochs?
Nadmm=3 # how many FA iterations

# regularization
lambda1=0.0001 # L1 sweet spot 0.00031
lambda2=0.0001 # L2 sweet spot ?

load_model=False
init_model=True
save_model=True
check_results=True
# if input is biased, each 1/K training data will have
# (slightly) different normalization. Otherwise, same normalization
biased_input=True

# Set this to true for using ResNet instead of simpler models
# In that case, instead of one layer, one block will be trained
use_resnet=False

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


# split 50000 training data into K subsets (last one will be smaller if K is not a divisor)
K_perslave=math.floor((50000+K-1)/K)
subsets_dict={}
for ck in range(K):
 if K_perslave*(ck+1)-1 <= 50000:
  subsets_dict[ck]=range(K_perslave*ck,K_perslave*(ck+1)-1)
 else:
  subsets_dict[ck]=range(K_perslave*ck,50000)

transforms_dict={}
for ck in range(K):
 if biased_input:
  # slightly different normalization for each subset
  transforms_dict[ck]=transforms.Compose(
   [transforms.ToTensor(),
     transforms.Normalize((0.5+ck/100,0.5-ck/100,0.5),(0.5+ck/100,0.5-ck/100,0.5))])
 else:
  # same normalization for all training data
  transforms_dict[ck]=transforms.Compose(
   [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


trainset_dict={}
testset_dict={}
trainloader_dict={}
testloader_dict={}
for ck in range(K):
 trainset_dict[ck]=torchvision.datasets.CIFAR10(root='./torchdata', train=True,
    download=True, transform=transforms_dict[ck])
 testset_dict[ck]=torchvision.datasets.CIFAR10(root='./torchdata', train=False,
    download=True, transform=transforms_dict[ck])
 trainloader_dict[ck] = torch.utils.data.DataLoader(trainset_dict[ck], batch_size=default_batch, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(subsets_dict[ck]),num_workers=1)
 testloader_dict[ck]=torch.utils.data.DataLoader(testset_dict[ck], batch_size=default_batch,
    shuffle=False, num_workers=0)

import numpy as np

# define a cnn
from simple_models import *

net_dict={}

for ck in range(K):
 if not use_resnet:
  net_dict[ck]=Net().to(mydevice)
 else:
  net_dict[ck]=ResNet18().to(mydevice)
 # update from saved models
 if load_model:
   checkpoint=torch.load('./s'+str(ck)+'.model',map_location=mydevice)
   net_dict[ck].load_state_dict(checkpoint['model_state_dict'])
   net_dict[ck].train()

########################################################################### helper functions
from simple_utils import *

def verification_error_check(net_dict):
  for ck in range(K):
   correct=0
   total=0
   net=net_dict[ck]
   for data in testloader_dict[ck]:
     images,labels=data
     outputs=net(Variable(images).to(mydevice))
     _,predicted=torch.max(outputs.data,1)
     correct += (predicted==labels.to(mydevice)).sum()
     total += labels.size(0)

   print('Accuracy of the network %d on the %d test images:%%%f'%
     (ck,total,100*correct/total))
##############################################################################################

if init_model:
  for ck in range(K):
   # note: use same seed for random number generation
   torch.manual_seed(0)
   net_dict[ck].apply(init_weights)

criteria_dict={}
for ck in range(K):
 criteria_dict[ck]=nn.CrossEntropyLoss()

if not use_resnet:
 L=number_of_layers(net_dict[0])
else:
 L=number_of_blocks(net_dict[0])

# get layer ids in given order 0..L-1 for selective training
np.random.seed(0)# get same list
Li=net_dict[0].train_order_layer_ids()
# make sure number of layers match
if L != len(Li):
  print("Warning, expected number of layers and given layer ids do not agree")
else:
  print(Li)

from lbfgsnew import LBFGSNew # custom optimizer
import torch.optim as optim
############### loop 00 (over the full net)
for nloop in range(Nloop):
  ############ loop 0 (over layers of the network)
  for ci in Li:
   for ck in range(K):
     if not use_resnet:
       unfreeze_one_layer(net_dict[ck],ci)
     else:
       unfreeze_one_block(net_dict[ck],ci)
   trainable=filter(lambda p: p.requires_grad, net_dict[0].parameters())
   params_vec1=torch.cat([x.view(-1) for x in list(trainable)])
  
   # number of parameters trained
   N=params_vec1.numel()
   z=torch.empty(N,dtype=torch.float,requires_grad=False)
   z.fill_(0.0)
  
   opt_dict={}
   for ck in range(K):
    opt_dict[ck]=LBFGSNew(filter(lambda p: p.requires_grad, net_dict[ck].parameters()), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)
    #opt_dict[ck]=optim.Adam(filter(lambda p: p.requires_grad, net_dict[ck].parameters()),lr=0.001)
  
   ############# loop 1 (Federated avaraging for subset of model)
   for nadmm in range(Nadmm):
     ##### loop 2 (data) (all network updates are done per epoch, because K is large
     ##### and data per host is assumed to be small)
     for epoch in range(Nepoch):

        #### loop 3 (models)
        for ck in range(K):
          running_loss=0.0
  
          for i,data1 in enumerate(trainloader_dict[ck],0):
            # get the inputs
            inputs1,labels1=data1
            # wrap them in variable
            inputs1,labels1=Variable(inputs1).to(mydevice),Variable(labels1).to(mydevice)
    
            trainable=filter(lambda p: p.requires_grad, net_dict[ck].parameters())
            params_vec1=torch.cat([x.view(-1) for x in list(trainable)])
  
            def closure1():
                 if torch.is_grad_enabled():
                    opt_dict[ck].zero_grad()
                 outputs=net_dict[ck](inputs1)
                 loss=criteria_dict[ck](outputs,labels1)
                 if ci in net_dict[ck].linear_layer_ids():
                    loss+=lambda1*torch.norm(params_vec1,1)+lambda2*(torch.norm(params_vec1,2)**2)
                 if loss.requires_grad:
                    loss.backward()
                 return loss
  
            # ADMM step 1
            opt_dict[ck].step(closure1)
  
            # only for diagnostics
            outputs1=net_dict[ck](inputs1)
            loss1=criteria_dict[ck](outputs1,labels1).data.item()
            running_loss +=loss1
           
            print('model=%d layer=%d %d(%d) minibatch=%d epoch=%d loss %e'%(ck,ci,nloop,N,i,epoch,loss1))
         

        # Federated averaging
        x_dict={}
        for ck in range(K):
          x_dict[ck]=get_trainable_values(net_dict[ck],mydevice)

        znew=torch.zeros(x_dict[0].shape)
        for ck in range(K):
         znew=znew+x_dict[ck]
        znew=znew/K

        dual_residual=torch.norm(z-znew).item()/N # per parameter
        print('dual (epoch=%d,loop=%d,layer=%d,avg=%d)=%e'%(epoch,nloop,ci,nadmm,dual_residual))
        z=znew
        for ck in range(K):
          put_trainable_values(net_dict[ck],z)

        if check_results:
          verification_error_check(net_dict)
  

print('Finished Training')


if save_model:
 for ck in range(K):
   torch.save({
     'model_state_dict':net_dict[ck].state_dict(),
     'epoch':epoch,
     'optimizer_state_dict':opt_dict[ck].state_dict(),
     'running_loss':running_loss,
     },'./s'+str(ck)+'.model')
