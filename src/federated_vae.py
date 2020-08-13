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

load_model=False
init_model=True
save_model=True
check_results=True
# if input is biased, each 1/K training data will have
# (slightly) different normalization. Otherwise, same normalization
biased_input=True

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

# define variational autoencoder
from simple_models import *

net_dict={}

for ck in range(K):
 net_dict[ck]=AutoEncoderCNN().to(mydevice)
 # update from saved models
 if load_model:
   checkpoint=torch.load('./s'+str(ck)+'.model',map_location=mydevice)
   net_dict[ck].load_state_dict(checkpoint['model_state_dict'])
   net_dict[ck].train()

########################################################################### helper functions
from simple_utils import *

reconstruction_function = nn.MSELoss(reduction='sum')
def loss_function(recon_x, x, mu, logvar):
  """
    recon_x: generated image
    x : original image
    mu : latent z mean
    logvar: latent z log variance : log(sigma^2)
  """
  MSE=reconstruction_function(recon_x,x)
  # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
  #print('%f %f'%(MSE,KLD))
  return MSE+KLD

##############################################################################################

if init_model:
  for ck in range(K):
   # note: use same seed for random number generation
   torch.manual_seed(0)
   net_dict[ck].apply(init_weights)

L=number_of_layers(net_dict[0])

# get layer ids in given order 0..L-1 for selective training
np.random.seed(0)# get same list
Li=net_dict[0].train_order_layer_ids()
# make sure number of layers match
if L != len(Li):
  print("Warning, expected number of layers and given layer ids do not agree")
else:
  print(Li)

import torch.optim as optim
############### loop 00 (over the full net)
for nloop in range(Nloop):
  ############ loop 0 (over layers of the network)
  for ci in Li:
   for ck in range(K):
     unfreeze_one_layer(net_dict[ck],ci)
   trainable=filter(lambda p: p.requires_grad, net_dict[0].parameters())
   params_vec1=torch.cat([x.view(-1) for x in list(trainable)])
  
   # number of parameters trained
   N=params_vec1.numel()
   z=torch.empty(N,dtype=torch.float,requires_grad=False)
   z.fill_(0.0)
  
   opt_dict={}
   for ck in range(K):
    opt_dict[ck]=optim.Adam(filter(lambda p: p.requires_grad, net_dict[ck].parameters()),lr=0.001)
  
   ############# loop 1 (Federated avaraging for subset of model)
   for nadmm in range(Nadmm):
     ##### loop 2 (data) (all network updates are done per epoch, because K is large
     ##### and data per host is assumed to be small)
     for epoch in range(Nepoch):

        #### loop 3 (models)
        for ck in range(K):
          running_loss=0.0
  
          for i,(images, _) in enumerate(trainloader_dict[ck],0): # ignore labels
            # get the inputs
            x=Variable(images).to(mydevice)
    
            def closure1():
                 out, mu, logvar = net_dict[ck](x)
                 if torch.is_grad_enabled():
                    opt_dict[ck].zero_grad()
                 loss=loss_function(out,x,mu,logvar)
                 if loss.requires_grad:
                    loss.backward()
                 return loss
  
            # ADMM step 1
            opt_dict[ck].step(closure1)
  
            # only for diagnostics
            out, mu, logvar= net_dict[ck](x)
            loss1=loss_function(out,x,mu,logvar).data.item()
            running_loss +=float(loss1)

            print('model=%d layer=%d %d(%d) minibatch=%d epoch=%d loss %e'%(ck,ci,nloop,N,i,epoch,loss1))
            del x,loss1,out,mu,logvar
         

        # Federated averaging
        x_dict={}
        for ck in range(K):
          x_dict[ck]=get_trainable_values(net_dict[ck])

        znew=torch.zeros(x_dict[0].shape)
        for ck in range(K):
         znew=znew+x_dict[ck]
        znew=znew/K

        dual_residual=torch.norm(z-znew).item()/N # per parameter
        print('dual (epoch=%d,loop=%d,layer=%d,avg=%d)=%e'%(epoch,nloop,ci,nadmm,dual_residual))
        z=znew
        for ck in range(K):
          put_trainable_values(net_dict[ck],z)


print('Finished Training')


if save_model:
 for ck in range(K):
   torch.save({
     'model_state_dict':net_dict[ck].state_dict(),
     'epoch':epoch,
     'optimizer_state_dict':opt_dict[ck].state_dict(),
     'running_loss':running_loss,
     },'./s'+str(ck)+'.model')
