import torch
import torchvision
import torchvision.transforms as transforms

import math
import time

# train three models, find average of weights and see what you get

torch.manual_seed(69)
default_batch=32 # no. of batches 50000/default_batch
batches_for_epoch=200#50000/default_batch

# regularization
lambda1=0.0001 # L1 sweet spot 0.00031
lambda2=0.0001 # L2 sweet spot ?

load_model=False
init_model=False
save_model=True
check_results=True
average_model=False

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
from simple_models import *
net1=Net1()
net2=Net1()
net3=Net1()


def init_weights(m):
  if type(m)==nn.Linear or type(m)==nn.Conv2d:
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

if init_model:
  # note: use same seed for random number generation
  torch.manual_seed(0)
  net1.apply(init_weights)
  torch.manual_seed(0)
  net2.apply(init_weights)
  torch.manual_seed(0)
  net3.apply(init_weights)

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



if average_model:
  X1=torch.cat([x.view(-1) for x in net1.parameters()])
  X2=torch.cat([x.view(-1) for x in net2.parameters()])
  X3=torch.cat([x.view(-1) for x in net3.parameters()])
  X=(X1+X2+X3)/3
  cnt=0
  for ci,(param1,param2,param3) in enumerate(zip(net1.parameters(),net2.parameters(),net3.parameters()),0):
   assert param1.numel() == param2.numel()
   numel=param1.numel()
   with torch.no_grad():
    param1.data.copy_(X[cnt:cnt+numel].view_as(param1.data))
    param2.data.copy_(X[cnt:cnt+numel].view_as(param2.data))
    param3.data.copy_(X[cnt:cnt+numel].view_as(param3.data))
   cnt+=numel
 

from lbfgsnew import LBFGSNew # custom optimizer
import torch.optim as optim
criterion1=nn.CrossEntropyLoss()
criterion2=nn.CrossEntropyLoss()
criterion3=nn.CrossEntropyLoss()
#optimizer1=optim.Adam(net1.parameters(), lr=0.001)
#optimizer2=optim.Adam(net2.parameters(), lr=0.001)
#optimizer3=optim.Adam(net3.parameters(), lr=0.001)
optimizer1 = LBFGSNew(net1.parameters(), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)
optimizer2 = LBFGSNew(net2.parameters(), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)
optimizer3 = LBFGSNew(net3.parameters(), history_size=10, max_iter=4, line_search_fn=True,batch_mode=True)

start_time=time.time()
# train network LBFGS 12, other 60
for epoch in range(12):
  running_loss1=0.0
  running_loss2=0.0
  running_loss3=0.0
  for i,(data1,data2,data3) in enumerate(zip(trainloader1,trainloader2,trainloader3),0):
    # get the inputs
    inputs1,labels1=data1
    # wrap them in variable
    inputs1,labels1=Variable(inputs1),Variable(labels1)

    # parameters in linear layers
    linear1=net1.linear_layer_parameters()

    def closure1():
        if torch.is_grad_enabled():
         optimizer1.zero_grad()
        outputs1=net1(inputs1)
        # regularization terms || ||_1 + || ||_2^2
        l1_penalty=lambda1*(torch.norm(linear1,1))
        l2_penalty=lambda2*(torch.norm(linear1,2)**2)
        loss=criterion1(outputs1,labels1)+l1_penalty+l2_penalty
        #print('1: loss %f'%(loss))
        if loss.requires_grad:
         loss.backward()
        return loss
    optimizer1.step(closure1)

    # only for diagnostics
    outputs1=net1(inputs1)
    loss=criterion1(outputs1,labels1)
    running_loss1 +=loss.data.item()


    # get the inputs
    inputs2,labels2=data2
    # wrap them in variable
    inputs2,labels2=Variable(inputs2),Variable(labels2)

    # parameters in linear layers
    linear1=net2.linear_layer_parameters()

    def closure2():
        if torch.is_grad_enabled():
         optimizer2.zero_grad()
        outputs2=net2(inputs2)
        # regularization terms || ||_1 + || ||_2^2
        l1_penalty=lambda1*(torch.norm(linear1,1))
        l2_penalty=lambda2*(torch.norm(linear1,2)**2)
        loss=criterion2(outputs2,labels2)+l1_penalty+l2_penalty
        #print('2: loss %f'%(loss))
        if loss.requires_grad:
         loss.backward()
        return loss
    optimizer2.step(closure2)

    # only for diagnostics
    outputs2=net2(inputs2)
    loss=criterion2(outputs2,labels2)
    running_loss2 +=loss.data.item()


    # get the inputs
    inputs3,labels3=data3
    # wrap them in variable
    inputs3,labels3=Variable(inputs3),Variable(labels3)

    # parameters in linear layers
    linear1=net3.linear_layer_parameters()

    def closure3():
        if torch.is_grad_enabled():
         optimizer3.zero_grad()
        outputs3=net3(inputs3)
        # regularization terms || ||_1 + || ||_2^2
        l1_penalty=lambda1*(torch.norm(linear1,1))
        l2_penalty=lambda2*(torch.norm(linear1,2)**2)
        loss=criterion3(outputs3,labels3)+l1_penalty+l2_penalty
        #print('3: loss %f'%(loss))
        if loss.requires_grad:
         loss.backward()
        return loss
    optimizer3.step(closure3)

    # only for diagnostics
    outputs3=net3(inputs3)
    loss=criterion3(outputs3,labels3)
    running_loss3 +=loss.data.item()

    # print statistics
    if i%(batches_for_epoch) == (batches_for_epoch-1): # after every epoch
      #print('%f: [%d, %5d] loss: %.3f'%
      #   (time.time()-start_time,epoch+1,i+1,running_loss/batches_for_epoch))
      print('%f %.3f %.3f %.3f'%
         (time.time()-start_time,running_loss1/batches_for_epoch,running_loss2/batches_for_epoch,running_loss3/batches_for_epoch))

      running_loss1=0.0
      running_loss2=0.0
      running_loss3=0.0




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
     'optimizer_state_dict':optimizer1.state_dict(),
     'running_loss':running_loss1,
     },'./s1.model')
 torch.save({
     'model_state_dict':net2.state_dict(),
     'epoch':epoch,
     'optimizer_state_dict':optimizer2.state_dict(),
     'running_loss':running_loss2,
     },'./s2.model')
 torch.save({
     'model_state_dict':net3.state_dict(),
     'epoch':epoch,
     'optimizer_state_dict':optimizer3.state_dict(),
     'running_loss':running_loss3,
     },'./s3.model')
