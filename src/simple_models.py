from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# We define the (relatively simple) CNN models used for training
# replaced relu with elu

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv1=nn.Conv2d(3,6,5)
    self.pool=nn.MaxPool2d(2,2)
    self.conv2=nn.Conv2d(6,16,5)
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

  # return linear layer ids (in 0...4)
  def linear_layer_ids(self):
    return [2,3,4]

  # return layer ids (in 0...4) ordered for training
  def train_order_layer_ids(self):
    return [2,0,1,3,4]



class Net1(nn.Module):
  def __init__(self):
    super(Net1,self).__init__()
    self.conv1=nn.Conv2d(3,32,3) # in 3 chan, out 32 chan, kernel 3x3
    self.conv2=nn.Conv2d(32,32,3) # in 32 chan, out 32 chan, kernel 3x3
    self.conv3=nn.Conv2d(32,64,3) # in 32 chan, out 64 chan, kernel 3x3
    self.conv4=nn.Conv2d(64,64,3) # in 64 chan, out 64 chan, kernel 3x3
    self.pool1=nn.MaxPool2d(2,2)
    self.pool2=nn.MaxPool2d(2,2)
    self.fc1=nn.Linear(64*5*5,512) # 5x5 comes from  image size
    self.fc2=nn.Linear(512,10)

  def forward(self,x):
    x=F.elu(self.conv1(x)) # original image 32x32, out image 30x30
    x=F.elu(self.conv2(x)) # image 28x28
    x=self.pool1(x) # image 14x14
    x=F.elu(self.conv3(x)) # image 12x12
    x=F.elu(self.conv4(x)) # image 10x10
    x=self.pool2(x) # image 5x5
    x=x.view(-1,64*5*5) # 5x5 from above
    x=F.elu(self.fc1(x))
    x=self.fc2(x)
    return x

  # return linear layer ids (in 0...5)
  def linear_layer_ids(self):
    return [4,5]

  # return layer ids (in 0...5) ordered for training
  def train_order_layer_ids(self):
    return [2,5,1,3,0,4]



class Net2(nn.Module):
  def __init__(self):
    super(Net2,self).__init__()
    # note image size do not change because of padding
    self.conv1=nn.Conv2d(3,64,3,padding=1) # in 3 chan, out 64 chan, kernel 3x3
    self.conv2=nn.Conv2d(64,128,3,padding=1) # in 64 chan, out 128 chan, kernel 3x3
    self.conv3=nn.Conv2d(128,256,3,padding=1) # in 128 chan, out 256 chan, kernel 3x3
    self.conv4=nn.Conv2d(256,512,3,padding=1) # in 256 chan, out 512 chan, kernel 3x3
    self.pool1=nn.MaxPool2d(2,2)
    self.pool2=nn.MaxPool2d(2,2)
    self.pool3=nn.MaxPool2d(2,2)
    self.pool4=nn.MaxPool2d(2,2)
    self.fc1=nn.Linear(512*2*2,128) # 2x2 comes from  image size
    self.fc2=nn.Linear(128,256)
    self.fc3=nn.Linear(256,512)
    self.fc4=nn.Linear(512,1024)
    self.fc5=nn.Linear(1024,10)

  def forward(self,x):
    x=F.elu(self.conv1(x)) # image 32x32
    x=self.pool1(x) # image 16x16
    x=F.elu(self.conv2(x)) # image 16x16
    x=self.pool2(x) # image 8x8
    x=F.elu(self.conv3(x)) # image 8x8
    x=self.pool3(x) # image 4x4
    x=F.elu(self.conv4(x)) # image 4x4
    x=self.pool4(x) # image 2x2
    x=x.view(-1,512*2*2) # 2x2 from above
    x=F.elu(self.fc1(x))
    x=F.elu(self.fc2(x))
    x=F.elu(self.fc3(x))
    x=F.elu(self.fc4(x))
    x=self.fc5(x)
    return x

  # return linear layer ids (in 0...8)
  def linear_layer_ids(self):
    return [4,5,6,7,8]

  # return layer ids (in 0...8) ordered for training
  def train_order_layer_ids(self):
    return [7, 2, 1, 4, 8, 6, 3, 0, 5]




