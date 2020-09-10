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
Nadmm=6 # how many ADMM iterations

# regularization
lambda1=0.0001 # L1 sweet spot 0.00031
lambda2=0.0001 # L2 sweet spot ?
admm_rho0=0.0001 # ADMM penalty, default value 
# note that per each slave, and per each layer, there will be a unique rho value

load_model=False
init_model=True
save_model=True
check_results=True
# if input is biased, each 1/K training data will have
# (slightly) different normalization. Otherwise, same normalization
biased_input=True
be_verbose=False

bb_update=False # if true, use adaptive ADMM (Barzilai-Borwein) update
if bb_update:
 #periodicity for the rho update, normally > 1
 bb_period_T=2
 bb_alphacorrmin=0.2 # minimum correlation required before an update is done
 bb_epsilon=1e-3 # threshold to stop updating
 bb_rhomax=0.1 # keep regularization below a safe upper limit


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

# regularization (per layer, per slave)
# Note: need to scale rho down when starting from scratch  
rho=torch.ones(L,3).to(mydevice)*admm_rho0
# this will be updated when using adaptive ADMM (bb_update=True)


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
   z=torch.empty(N,dtype=torch.float,requires_grad=False).to(mydevice)
   z.fill_(0.0)
   y_dict={}
   for ck in range(K):
      y_dict[ck]=torch.empty(N,dtype=torch.float,requires_grad=False).to(mydevice)
      y_dict[ck].fill_(0.0)

   if bb_update: # extra storage for adaptive ADMM
      yhat_dict={}
      yhat0_dict={}
      x0_dict={}
      for ck in range(K):
         yhat_dict[ck]=torch.empty(N,dtype=torch.float,requires_grad=False).to(mydevice)
         yhat_dict[ck].fill_(0.0)
         x0_dict[ck]=torch.empty(N,dtype=torch.float,requires_grad=False).to(mydevice)
         yhat0_dict[ck]=get_trainable_values(net_dict[ck],mydevice)
      
  
   opt_dict={}
   for ck in range(K):
    opt_dict[ck]=LBFGSNew(filter(lambda p: p.requires_grad, net_dict[ck].parameters()), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)
    #opt_dict[ck]=optim.Adam(filter(lambda p: p.requires_grad, net_dict[ck].parameters()),lr=0.001)
  
   ############# loop 1 (ADMM for subset of model)
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
                 # augmented lagrangian terms y^T (x-z) + rho/2 ||x-z||^2
                 augmented_terms=(torch.dot(y_dict[ck],params_vec1-z))+0.5*rho[ci,0]*(torch.norm(params_vec1-z,2)**2)
                 loss=criteria_dict[ck](outputs,labels1)+augmented_terms
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
           
            if be_verbose:
              print('model=%d layer=%d %d(%d) minibatch=%d epoch=%d loss %e'%(ck,ci,nloop,N,i,epoch,loss1))
         
        # ADMM step 2 update global z
        x_dict={}
        for ck in range(K):
          x_dict[ck]=get_trainable_values(net_dict[ck],mydevice)

        # decide and update rho for this ADMM iteration (not the first iteration)
        if bb_update:
          if nadmm==0:
            # store for next use
            for ck in range(K):
              x0_dict[ck]=x_dict[ck]
          elif (nadmm%bb_period_T)==0:
            for ck in range(K):
              yhat_1=y_dict[ck]+rho[ci,0]*(x_dict[ck]-z)
              deltay1=yhat_1-yhat0_dict[ck]
              deltax1=x_dict[ck]-x0_dict[ck]
              # inner products
              d11=torch.dot(deltay1,deltay1)
              d12=torch.dot(deltay1,deltax1) # note: can be negative
              d22=torch.dot(deltax1,deltax1)

              print('admm %d deltas=(%e,%e,%e)'%(nadmm,d11,d12,d22))
              rhonew=rho[ci,0]
              # catch situation where denominator is very small
              if torch.abs(d12).item()>bb_epsilon and d11.item()>bb_epsilon and d22.item()>bb_epsilon:
                 alpha=d12/torch.sqrt(d11*d22)
                 alphaSD=d11/d22
                 alphaMG=d12/d22

                 if 2.0*alphaMG>alphaSD:
                   alphahat=alphaMG
                 else:
                   alphahat=alphaSD-0.5*alphaMG
                 if alpha>=bb_alphacorrmin and alphahat<bb_rhomax: # catches d12 being negative
                   rhonew=alphahat
                 print('admm %d alphas=(%e,%e,%e)'%(nadmm,alpha,alphaSD,alphaMG))

              rho[ci,0]=rhonew
              ###############

              # carry forward current values for the next update
              yhat0_dict[ck]=yhat_1
              x0_dict[ck]=x_dict[ck]


        znew=torch.zeros(x_dict[0].shape).to(mydevice)
        for ck in range(K):
         znew=znew+x_dict[ck]
        znew=znew/K

        dual_residual=torch.norm(z-znew).item()/N # per parameter
        z=znew

        # -> master will send z to all slaves
        # ADMM step 3 update Lagrange multiplier 
        primal_residual=0.0
        for ck in range(K):
          ydelta=rho[ci,0]*(x_dict[ck]-z)
          primal_residual=primal_residual+torch.norm(ydelta)
          y_dict[ck].add_(ydelta)

        print('layer=%d(%d,%f) ADMM=%d/%d primal=%e dual=%e'%(ci,N,torch.mean(rho).item(),nadmm,nloop,primal_residual,dual_residual))

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
