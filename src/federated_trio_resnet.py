import torch
import torchvision
import torchvision.transforms as transforms

import math
import time

# train three models, Federated learning
# each iteration over a subset of parameters: 1) average 2) pass back average to slaves 3) SGD step
# initialize with pre-trained models (better to use common initialization)
# loop order: loop 0: parameters/layers   {
#               loop 1 : {  averaging (part of the model)
#                loop 2: { epochs/databatches  { train; } } } }
# repeat this Nloop times


torch.manual_seed(69)
default_batch=32 # no. of batches (50000/3)/default_batch
batches_for_epoch=521#(50000/3)/default_batch
Nloop=12 # how many loops over the whole network
Nepoch=1 # how many epochs?
Nadmm=3 # how many FA iterations

# regularization
lambda1=0.0001 # L1 sweet spot 0.00031
lambda2=0.0001 # L2 sweet spot ?

load_model=False
init_model=True
save_model=False
check_results=False

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



import matplotlib.pyplot as plt
import numpy as np

# define a cnn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# test relu or elu
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = F.elu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


net1=ResNet18()
net2=ResNet18()
net3=ResNet18()
net1.train()
net2.train()
net3.train()

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

# possible partition of layers, by index
# if ci<= upperindex[k] and ci>upperindex[k-1], 
# all parameters belong to partition k
# NOTE: this should be specified by hand
upidx=[2,8,14,23,29,38,44,53,59,61]


########################################################################### helper functions
def init_weights(m):
  if type(m)==nn.Linear or type(m)==nn.Conv2d:
    torch.nn.init.xavier_uniform_(m.weight)
    if hasattr(m, 'bias'):
      if m.bias is not None:
       m.bias.data.fill_(0.01)

def unfreeze_one_layer(net,upperindex,layer_id):
  ''' set all layers to not-trainable except the layer given by layer_id (0,1,..)
    parameters ci, if upperindex[layer_id-1] < ci and upperindex[layer_id] >= ci
  '''
  for ci,param in enumerate(net.parameters(),0):
    if layer_id==0:
      if (ci<=upperindex[layer_id]):
       param.requires_grad=True
      else:
       param.requires_grad=False
    else:
      if (ci > upperindex[layer_id-1]) and (ci<=upperindex[layer_id]):
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
  ' get total number of blocks of layers '
  return len(upidx)

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
print(Li)

import torch.optim as optim
############### loop 00 (over the full net)
for nloop in range(Nloop):
  ############ loop 0 (over layers of the network)
  for ci in Li:
   unfreeze_one_layer(net1,upidx,ci)
   unfreeze_one_layer(net2,upidx,ci)
   unfreeze_one_layer(net3,upidx,ci)
   trainable=filter(lambda p: p.requires_grad, net1.parameters())
   params_vec1=torch.cat([x.view(-1) for x in list(trainable)])
  
   # number of parameters trained
   N=params_vec1.numel()
   z=torch.empty(N,dtype=torch.float,requires_grad=False)
   z.fill_(0.0)
  
   #opt1=optim.Adam(filter(lambda p: p.requires_grad, net1.parameters()),lr=0.001)
   #opt2=optim.Adam(filter(lambda p: p.requires_grad, net2.parameters()),lr=0.001)
   #opt3=optim.Adam(filter(lambda p: p.requires_grad, net3.parameters()),lr=0.001)
   opt1 =torch.optim.LBFGS(filter(lambda p: p.requires_grad, net1.parameters()), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)
   opt2 =torch.optim.LBFGS(filter(lambda p: p.requires_grad, net2.parameters()), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)
   opt3 =torch.optim.LBFGS(filter(lambda p: p.requires_grad, net3.parameters()), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)
  
   ############# loop 1 (Federated avaraging for subset of model)
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
                 loss=criterion1(outputs,labels1)
                 if loss.requires_grad:
                    loss.backward()
                 return loss
           def closure2():
                 if torch.is_grad_enabled():
                    opt2.zero_grad()
                 outputs=net2(inputs2)
                 loss=criterion2(outputs,labels2)
                 if loss.requires_grad:
                    loss.backward()
                 return loss
           def closure3():
                 if torch.is_grad_enabled():
                    opt3.zero_grad()
                 outputs=net3(inputs3)
                 loss=criterion3(outputs,labels3)
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
  
           
           print('layer=%d %d(%d) minibatch=%d epoch=%d losses %e,%e,%e'%(ci,nloop,N,i,epoch,loss1,loss2,loss3))
     # Federated averaging
     x1=get_trainable_values(net1)
     x2=get_trainable_values(net2)
     x3=get_trainable_values(net3)
     znew=(x1+x2+x3)/3
     dual_residual=torch.norm(z-znew).item()/N # per parameter
     print('dual (loop=%d,layer=%d,avg=%d)=%f'%(nloop,ci,nadmm,dual_residual))
     z=znew
     put_trainable_values(net1,z)
     put_trainable_values(net2,z)
     put_trainable_values(net3,z)
  

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
