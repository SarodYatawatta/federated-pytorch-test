import torch
import torchvision
import torchvision.transforms as transforms

import math
import time

# How many models (==slaves)
K=10
# train K models using 1/K of data each


torch.manual_seed(69)
# minibatch size
default_batch=128 # no. of batches per model is (50000/K)/default_batch
Nepoch=20 # how many epochs?

load_model=False
init_model=True
save_model=True
check_results=True
# if input is biased, each 1/K training data will have
# (slightly) different normalization. Otherwise, same normalization
biased_input=True
be_verbose=False

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
     (ck,total,100*correct//total))
##############################################################################################

if init_model:
  for ck in range(K):
   # note: use same seed for random number generation
   torch.manual_seed(0)
   net_dict[ck].apply(init_weights)

criteria_dict={}
for ck in range(K):
 criteria_dict[ck]=nn.CrossEntropyLoss()

# get layer ids in given order 0..L-1 for selective training
np.random.seed(0)# get same list

from lbfgsnew import LBFGSNew # custom optimizer
import torch.optim as optim
############### loop 00 (epochs)
for epoch in range(Nepoch):
   opt_dict={}
   for ck in range(K):
    #opt_dict[ck]=LBFGSNew(net_dict[ck].parameters(), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)
    opt_dict[ck]=optim.Adam(net_dict[ck].parameters(),lr=0.001)
  
   #### loop 3 (models)
   print('Epoch %d'%epoch)
   for ck in range(K):
          running_loss=0.0
  
          for i,data1 in enumerate(trainloader_dict[ck],0):
            # get the inputs
            inputs1,labels1=data1
            # wrap them in variable
            inputs1,labels1=Variable(inputs1).to(mydevice),Variable(labels1).to(mydevice)
    
            def closure1():
                 if torch.is_grad_enabled():
                    opt_dict[ck].zero_grad()
                 outputs=net_dict[ck](inputs1)
                 loss=criteria_dict[ck](outputs,labels1)
                 if loss.requires_grad:
                    loss.backward()
                 return loss
  
            # ADMM step 1
            opt_dict[ck].step(closure1)
  
            # only for diagnostics
            outputs1=net_dict[ck](inputs1)
            loss1=criteria_dict[ck](outputs1,labels1).data.item()
            running_loss +=loss1
           
            if be_verbose:
              print('model=%d minibatch=%d epoch=%d loss %e'%(ck,i,epoch,loss1))
         
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
