import torch
import torchvision
import torchvision.transforms as transforms

import math
import time

# train three models, also trying to reach consensus between models
# initialize with pre-trained models (better to use common initialization)
# loop order: loop 0: parameters/layers   {
#               loop 1 : {  admm (part of the model)
#                loop 2: { databatches  { train; } } } }
# repeat this Nloop times


torch.manual_seed(69)
default_batch=512 # no. of batches (50000/3)/default_batch
batches_for_epoch=33#(50000/3)/default_batch
Nloop=12 # how many loops over the whole network
Nepoch=1 # how many epochs?
Nadmm=5 # how many ADMM iterations

admm_rho0=0.001 # ADMM penalty, default value 
# note that per each slave, and per each layer, there will be a unique rho value

# regularization
lambda1=0.0001 # L1 sweet spot 0.00031
lambda2=0.0001 # L2 sweet spot ?

load_model=False
init_model=True
save_model=True
check_results=True
bb_update=True # if true, use adaptive ADMM (Barzilai-Borwein) update

if bb_update:
 #periodicity for the rho update, normally > 1
 bb_period_T=2
 bb_alphacorrmin=0.2 # minimum correlation required before an update is done
 bb_epsilon=1e-3 # threshold to stop updating
 bb_rhomax=0.1 # keep regularization below a safe upper limit

transform=transforms.Compose(
   [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# split 50000 training data into three
subset1=range(0,16666)
subset2=range(16666,33333)
subset3=range(33333,50000)

trainset1=torchvision.datasets.CIFAR10(root='./torchdata', train=True,
    download=True, transform=transform)
trainloader1 = torch.utils.data.DataLoader(trainset1, batch_size=default_batch, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(subset1),num_workers=1)
trainloader2 = torch.utils.data.DataLoader(trainset1, batch_size=default_batch, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(subset2),num_workers=1)
trainloader3 = torch.utils.data.DataLoader(trainset1, batch_size=default_batch, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(subset3),num_workers=1)

testset=torchvision.datasets.CIFAR10(root='./torchdata', train=False,
    download=True, transform=transform)

testloader=torch.utils.data.DataLoader(testset, batch_size=default_batch,
    shuffle=False, num_workers=0)


classes=('plane', 'car', 'bird', 'cat', 
  'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



import numpy as np

# define a cnn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# test relu or elu

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv1=nn.Conv2d(3,6,5) # increase width 6->?
    self.pool=nn.MaxPool2d(2,2)
    self.conv2=nn.Conv2d(6,16,5) # increase width 6->?
    self.fc1=nn.Linear(16*5*5,120)
    self.fc2=nn.Linear(120,84)
    self.fc3=nn.Linear(84,10)
  
  def forward(self,x):
    x=self.pool(F.elu(self.conv1(x)))
    x=self.pool(F.elu(self.conv2(x)))
    x=x.view(-1,16*5*5)
    x=F.elu(self.fc1(x))
    x=F.elu(self.fc2(x))
    x=self.fc3(x)
    return x

net1=Net()
net2=Net()
net3=Net()

# update from saved models
if load_model:
  checkpoint=torch.load('./s1.model')
  net1.load_state_dict(checkpoint['model_state_dict'])
  net1.train()
  checkpoint=torch.load('./s2.model')
  net2.load_state_dict(checkpoint['model_state_dict'])
  net2.train()
  checkpoint=torch.load('./s3.model')
  net3.load_state_dict(checkpoint['model_state_dict'])
  net3.train()

########################################################################### helper functions
def init_weights(m):
  if type(m)==nn.Linear:
    torch.nn.init.xavier_uniform_(m.weight)
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

def get_trainable_values(net):
  ' return trainable parameter values as a vector (only the first parameter set)'
  trainable=filter(lambda p: p.requires_grad, net.parameters())
  paramlist=list(trainable) 
  N=0
  for params in paramlist:
    N+=params.numel()
  X=torch.empty(N,dtype=torch.float)
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

def distance_of_layers(net1,net2,net3):
  'find Eculidean distance of each layer (from the mean) and return this as vector (normalized by size of layer)'
  L=number_of_layers(net1)
  W=np.zeros(L)
  for ci in range(L):
   unfreeze_one_layer(net1,ci)
   unfreeze_one_layer(net2,ci)
   unfreeze_one_layer(net3,ci)
   W1=get_trainable_values(net1)
   W2=get_trainable_values(net2)
   W3=get_trainable_values(net3)
   N=W1.numel()
   Wm=(W1+W2+W3)/3
   W[ci]=(Wm-W1).norm()/N
   W[ci]+=(Wm-W2).norm()/N
   W[ci]+=(Wm-W3).norm()/N
  return W

def sthreshold(z,sval):
  """soft threshold a tensor  
    if element(z) > sval, element(z)=sval
    if element(z) < -sval, element(z)=-sval 
  """
  with torch.no_grad():
    T=nn.Softshrink(sval) # if z_i < -sval, z_i -> z_i +sval , ...
    z=T(z)
  return z

##############################################################################################

if init_model:
  # note: use same seed for random number generation
  torch.manual_seed(0)
  net1.apply(init_weights)
  torch.manual_seed(0)
  net2.apply(init_weights)
  torch.manual_seed(0)
  net3.apply(init_weights)


criterion1=nn.CrossEntropyLoss()
criterion2=nn.CrossEntropyLoss()
criterion3=nn.CrossEntropyLoss()

L=number_of_layers(net1)
# create layer ids in random order 0..L-1 for selective training
np.random.seed(0)# get same list
Li=np.random.permutation(L).tolist()
# prioritize by current difference in weights
#D=distance_of_layers(net1,net2,net3)
#print(D)
#Li=np.argsort(D)
#Li=Li.tolist()
# remove layer numbers with lowest dist
#for ci in range(4):
# Li.pop()
print(Li)

# regularization (per layer, per slave)
# Note: need to scale rho down when starting from scratch  
rho=torch.ones(L,3)*admm_rho0
# this will be updated when using adaptive ADMM

from lbfgsnew import LBFGSNew # custom optimizer
import torch.optim as optim
############### loop 00 (over the full net)
for nloop in range(Nloop):
  ############ loop 0 (over layers of the network)
  for ci in Li:
   unfreeze_one_layer(net1,ci)
   unfreeze_one_layer(net2,ci)
   unfreeze_one_layer(net3,ci)
   trainable=filter(lambda p: p.requires_grad, net1.parameters())
   params_vec1=torch.cat([x.view(-1) for x in list(trainable)])
  
   # number of parameters trained
   N=params_vec1.numel()
   # set up primal,dual variables
   y1=torch.empty(N,dtype=torch.float,requires_grad=False)
   y2=torch.empty(N,dtype=torch.float,requires_grad=False)
   y3=torch.empty(N,dtype=torch.float,requires_grad=False)
   y1.fill_(0.0)
   y2.fill_(0.0)
   y3.fill_(0.0)
   z=torch.empty(N,dtype=torch.float,requires_grad=False)
   z.fill_(0.0)
   if bb_update: # extra storage for adaptive ADMM
     yhat_1=torch.empty(N,dtype=torch.float,requires_grad=False)
     yhat_2=torch.empty(N,dtype=torch.float,requires_grad=False)
     yhat_3=torch.empty(N,dtype=torch.float,requires_grad=False)
     x0_1=torch.empty(N,dtype=torch.float,requires_grad=False)
     x0_2=torch.empty(N,dtype=torch.float,requires_grad=False)
     x0_3=torch.empty(N,dtype=torch.float,requires_grad=False)
     # initialize yhat0 for all slaves
     yhat_1.fill_(0.0)
     yhat_2.fill_(0.0)
     yhat_3.fill_(0.0)
     yhat0_1=get_trainable_values(net1)
     yhat0_2=get_trainable_values(net2)
     yhat0_3=get_trainable_values(net3)
     
  
   #opt1=optim.Adam(filter(lambda p: p.requires_grad, net1.parameters()),lr=0.001)
   #opt2=optim.Adam(filter(lambda p: p.requires_grad, net2.parameters()),lr=0.001)
   #opt3=optim.Adam(filter(lambda p: p.requires_grad, net3.parameters()),lr=0.001)
   opt1 =LBFGSNew(filter(lambda p: p.requires_grad, net1.parameters()), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)
   opt2 =LBFGSNew(filter(lambda p: p.requires_grad, net2.parameters()), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)
   opt3 =LBFGSNew(filter(lambda p: p.requires_grad, net3.parameters()), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)
  
   ############# loop 1 (ADMM for subset of model)
   for nadmm in range(Nadmm):
     ##### loop 2 (data)
     for epoch in range(Nepoch):
        running_loss1=0.0
        running_loss2=0.0
        running_loss3=0.0
  
        for i,(data1,data2,data3) in enumerate(zip(trainloader1,trainloader2,trainloader3),0):
           # get the inputs
           inputs1,labels1=data1
           inputs2,labels2=data2
           inputs3,labels3=data3
           # wrap them in variable
           inputs1,labels1=Variable(inputs1),Variable(labels1)
           inputs2,labels2=Variable(inputs2),Variable(labels2)
           inputs3,labels3=Variable(inputs3),Variable(labels3)
    
           trainable=filter(lambda p: p.requires_grad, net1.parameters())
           params_vec1=torch.cat([x.view(-1) for x in list(trainable)])
           trainable=filter(lambda p: p.requires_grad, net2.parameters())
           params_vec2=torch.cat([x.view(-1) for x in list(trainable)])
           trainable=filter(lambda p: p.requires_grad, net3.parameters())
           params_vec3=torch.cat([x.view(-1) for x in list(trainable)])
  
           # fc1 and fc3 layers have L1 and L2 regularization
           def closure1():
                 if torch.is_grad_enabled():
                    opt1.zero_grad()
                 outputs=net1(inputs1)
                 # augmented lagrangian terms y^T x + rho/2 ||x-z||^2
                 augmented_terms=(torch.dot(y1,params_vec1))+0.5*rho[ci,0]*(torch.norm(params_vec1-z,2)**2)
                 loss=criterion1(outputs,labels1)+augmented_terms
                 if ci==2 or ci==3:
                    loss+=lambda1*torch.norm(params_vec1,1)+lambda2*(torch.norm(params_vec1,2)**2)
                 if loss.requires_grad:
                    loss.backward()
                 return loss
           def closure2():
                 if torch.is_grad_enabled():
                    opt2.zero_grad()
                 outputs=net2(inputs2)
                 # augmented lagrangian terms y^T x + rho/2 ||x-z||^2
                 augmented_terms=(torch.dot(y2,params_vec2))+0.5*rho[ci,1]*(torch.norm(params_vec2-z,2)**2)
                 loss=criterion2(outputs,labels2)+augmented_terms
                 if ci==2 or ci==3:
                    loss+=lambda1*torch.norm(params_vec2,1)+lambda2*(torch.norm(params_vec2,2)**2)
                 if loss.requires_grad:
                    loss.backward()
                 return loss
           def closure3():
                 if torch.is_grad_enabled():
                    opt3.zero_grad()
                 outputs=net3(inputs3)
                 # augmented lagrangian terms y^T x + rho/2 ||x-z||^2
                 augmented_terms=(torch.dot(y3,params_vec3))+0.5*rho[ci,2]*(torch.norm(params_vec3-z,2)**2)
                 loss=criterion3(outputs,labels3)+augmented_terms
                 if ci==2 or ci==3:
                    loss+=lambda1*torch.norm(params_vec3,1)+lambda2*(torch.norm(params_vec3,2)**2)
                 if loss.requires_grad:
                    loss.backward()
                 return loss
  
           # ADMM step 1
           opt1.step(closure1)
           opt2.step(closure2)
           opt3.step(closure3)
  
           # only for diagnostics
           outputs1=net1(inputs1)
           loss1=criterion1(outputs1,labels1).data.item()
           running_loss1 +=loss1
           outputs2=net2(inputs2)
           loss2=criterion2(outputs2,labels2).data.item()
           running_loss2 +=loss2
           outputs3=net3(inputs3)
           loss3=criterion3(outputs3,labels3).data.item()
           running_loss3 +=loss3
  
           
           print('layer=%d %d(%d,%f) minibatch=%d epoch=%d losses %e,%e,%e'%(ci,nloop,N,torch.mean(rho).item(),i,epoch,loss1,loss2,loss3))
  
     # ADMM step 2 update global z
     x1=get_trainable_values(net1)
     x2=get_trainable_values(net2)
     x3=get_trainable_values(net3)
  
     # decide and update rho for this ADMM iteration (not the first iteration)
     if bb_update:
       if nadmm==0:
         # store for next use
         x0_1=x1
         x0_2=x2
         x0_3=x3
       elif (nadmm%bb_period_T)==0:
         yhat_1=y1+rho[ci,0]*(x1-z)
         deltay1=yhat_1-yhat0_1
         deltax1=x1-x0_1

         # inner products
         d11=torch.dot(deltay1,deltay1)
         d12=torch.dot(deltay1,deltax1) # note: can be negative
         d22=torch.dot(deltax1,deltax1)
    
         print('admm %d deltas=(%e,%e,%e)\n'%(nadmm,d11,d12,d22))
         rhonew=rho[ci,0]
         # catch situation where denominator is very small
         if torch.abs(d12).item()>bb_epsilon and d11.item()>bb_epsilon and d22.item()>bb_epsilon:
           alpha=d12/math.sqrt(d11*d22)
           alphaSD=d11/d12
           alphaMG=d12/d22 

           if 2.0*alphaMG>alphaSD:
             alphahat=alphaMG
           else:
             alphahat=alphaSD-0.5*alphaMG
           if alpha>=bb_alphacorrmin and alphahat<bb_rhomax: # catches d12 being negative
             rhonew=alphahat
           print('admm %d alphas=(%e,%e,%e)\n'%(nadmm,alpha,alphaSD,alphaMG))

         rho[ci,0]=rhonew
         ###############

         yhat_2=y2+rho[ci,1]*(x2-z)
         deltay1=yhat_2-yhat0_2
         deltax1=x2-x0_2

         # inner products
         d11=torch.dot(deltay1,deltay1)
         d12=torch.dot(deltay1,deltax1) # note: can be negative
         d22=torch.dot(deltax1,deltax1)
    
         print('admm %d deltas=(%e,%e,%e)\n'%(nadmm,d11,d12,d22))
         rhonew=rho[ci,1]
         # catch situation where denominator is very small
         if torch.abs(d12).item()>bb_epsilon and d11.item()>bb_epsilon and d22.item()>bb_epsilon:
           alpha=d12/math.sqrt(d11*d22)
           alphaSD=d11/d12
           alphaMG=d12/d22 

           if 2.0*alphaMG>alphaSD:
             alphahat=alphaMG
           else:
             alphahat=alphaSD-0.5*alphaMG
           if alpha>=bb_alphacorrmin and alphahat<bb_rhomax: # catches d12 being negative
             rhonew=alphahat
           print('admm %d alphas=(%e,%e,%e)\n'%(nadmm,alpha,alphaSD,alphaMG))

         rho[ci,1]=rhonew
         ###############

         yhat_3=y3+rho[ci,2]*(x3-z)
         deltay1=yhat_3-yhat0_3
         deltax1=x3-x0_3

         # inner products
         d11=torch.dot(deltay1,deltay1)
         d12=torch.dot(deltay1,deltax1) # note: can be negative
         d22=torch.dot(deltax1,deltax1)
    
         print('admm %d deltas=(%e,%e,%e)\n'%(nadmm,d11,d12,d22))
         rhonew=rho[ci,2]
         # catch situation where denominator is very small
         if torch.abs(d12).item()>bb_epsilon and d11.item()>bb_epsilon and d22.item()>bb_epsilon:
           alpha=d12/math.sqrt(d11*d22)
           alphaSD=d11/d12
           alphaMG=d12/d22 

           if 2.0*alphaMG>alphaSD:
             alphahat=alphaMG
           else:
             alphahat=alphaSD-0.5*alphaMG
           if alpha>=bb_alphacorrmin and alphahat<bb_rhomax: # catches d12 being negative
             rhonew=alphahat
           print('admm %d alphas=(%e,%e,%e)\n'%(nadmm,alpha,alphaSD,alphaMG))

         rho[ci,2]=rhonew
         ###############

         print(rho)
         # carry forward current values for the next update
         yhat0_1=yhat_1
         x0_1=x1
         yhat0_2=yhat_2
         x0_2=x2
         yhat0_3=yhat_3
         x0_3=x3

 
     # <- each slave will send (y+rho*x)/rho to master
     znew=((y1+rho[ci,0]*x1) +(y2+rho[ci,1]*x2) +(y3+rho[ci,2]*x3))/(rho[ci,0]+rho[ci,1]+rho[ci,2])
     dual_residual=torch.norm(z-znew).item()/N

     # decide if to stop ADMM if dual residual is too low 
     #if dual_residual<1e-9:
     #  break
     z=znew
     # -> master will send z to all slaves
     # ADMM step 3 update Lagrange multiplier 
     y1.add_(rho[ci,0]*(x1-z)) 
     y2.add_(rho[ci,1]*(x2-z)) 
     y3.add_(rho[ci,2]*(x3-z)) 
     primal_residual=(torch.norm(x1-z).item()+torch.norm(x2-z).item()+torch.norm(x3-z).item())/(3*N)
  
     
     print('layer=%d(%d,%f) ADMM=%d primal=%e dual=%e'%(ci,N,torch.mean(rho).item(),nadmm,primal_residual,dual_residual))
  
     


print('Finished Training')


if check_results:
  correct1=0
  correct2=0
  correct3=0
  total=0

  for data in testloader:
    images,labels=data
    outputs=net1(Variable(images))
    _,predicted=torch.max(outputs.data,1)
    correct1 += (predicted==labels).sum()
    outputs=net2(Variable(images))
    _,predicted=torch.max(outputs.data,1)
    correct2 += (predicted==labels).sum()
    outputs=net3(Variable(images))
    _,predicted=torch.max(outputs.data,1)
    correct3 += (predicted==labels).sum()
    total += labels.size(0)
   
  print('Accuracy of the network on the %d test images:%%%f %%%f %%%f'%
     (total,100*correct1/total,100*correct2/total,100*correct3/total))



if save_model:
 torch.save({
     'model_state_dict':net1.state_dict(),
     'epoch':epoch,
     'optimizer_state_dict':opt1.state_dict(),
     'running_loss':running_loss1,
     },'./s1.model')
 torch.save({
     'model_state_dict':net2.state_dict(),
     'epoch':epoch,
     'optimizer_state_dict':opt2.state_dict(),
     'running_loss':running_loss2,
     },'./s2.model')
 torch.save({
     'model_state_dict':net3.state_dict(),
     'epoch':epoch,
     'optimizer_state_dict':opt3.state_dict(),
     'running_loss':running_loss3,
     },'./s3.model')


#  cat LBFGS.out |grep ^layer |cut -d'=' -f 5|cut -d' ' -f 1 > primal_residual
#  cat LBFGS.out |grep ^layer |cut -d'=' -f 6|cut -d' ' -f 1 > dual_residual
#  cat LBFGS.out |grep ^layer |cut -d'=' -f 6|cut -d' ' -f 3|cut -d',' -f 1 > err1
#  cat LBFGS.out |grep ^layer |cut -d'=' -f 6|cut -d' ' -f 3|cut -d',' -f 2 > err2
#  cat LBFGS.out |grep ^layer |cut -d'=' -f 6|cut -d' ' -f 3|cut -d',' -f 3 > err3
