from torch.autograd import Variable
import torch
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
  
  # return linear layer parameters (for regularization)
  def linear_layer_parameters(self):
    linear1=torch.cat([x.view(-1) for x in (self.fc1.parameters() or self.fc2.parameters() or self.fc3.parameters())])
    return linear1

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

  # return linear layer parameters (for regularization)
  def linear_layer_parameters(self):
    linear1=torch.cat([x.view(-1) for x in (self.fc1.parameters() or self.fc2.parameters())])
    return linear1

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

  # return linear layer parameters (for regularization)
  def linear_layer_parameters(self):
    linear1=torch.cat([x.view(-1) for x in (self.fc1.parameters() or self.fc2.parameters() or self.fc3.parameters() or self.fc4.parameters() or self.fc5.parameters())])
    return linear1



  # return layer ids (in 0...8) ordered for training
  def train_order_layer_ids(self):
    return [7, 2, 1, 4, 8, 6, 3, 0, 5]



####################### ResNet related  classes
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
    def __init__(self, block, num_blocks, qualifier, num_classes=10):
        super(ResNet, self).__init__()
        self.qualifier=qualifier # 9 or 18
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

    # possible partition of layers, by index
    # if ci<= upperindex[k] and ci>upperindex[k-1], 
    # all parameters belong to partition k
    # NOTE: this should be specified by hand
    def upidx(self):
      if self.qualifier==18:
       return [[0,2],[3,8],[9,14],[15,23],[24,29],[30,38],[39,44],[45,53],[54,59],[60,61]]
      else:
       return [[0,2],[3,8],[9,14],[15,17],[18,23],[24,29],[30,32],[33,37]]

    # return linear layer ids (empty)
    def linear_layer_ids(self):
      return []
 
    # return block ids (in 0...max_block) ordered for training 
    def train_order_layer_ids(self):
      if self.qualifier==18:
       return [0,1,2,3,4,5,6,7,8,9]
      else:
       return [0,1,2,3,4,5,6,7]


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2], qualifier=18)

def ResNet9():
    return ResNet(BasicBlock, [1,1,1,1], qualifier=9)

####################### End of ResNet related  classes

# define variational autoencoder
########################################################
class AutoEncoderCNN(nn.Module):
    # variational AE CNN  CIFAR10  
    def __init__(self):
        super().__init__()
        self.latent_dim=10
        # 32x32 -> 16x16
        self.conv1=nn.Conv2d(3, 12, 4, stride=2, padding=1)# in 3 chan, out 12 chan, kernel 4x4
        # 16x16 -> 8x8
        self.conv2=nn.Conv2d(12, 24, 4, stride=2,  padding=1)# in 12 chan, out 24 chan, kernel 4x4
        # 8x8 -> 4x4
        self.conv3=nn.Conv2d(24, 48, 4, stride=2,  padding=1)# in 24 chan, out 48 chan, kernel 4x4
        # 4x4 -> 2x2
        self.conv4=nn.Conv2d(48, 96, 4, stride=2,  padding=1)# in 48 chan, out 96 chan, kernel 4x4

        self.fc1=nn.Linear(384,16)
        self.fc21=nn.Linear(16,self.latent_dim)
        self.fc22=nn.Linear(16,self.latent_dim)

        self.fc3=nn.Linear(self.latent_dim,384)
        self.tconv1=nn.ConvTranspose2d(96,48,4,stride=2,padding=1)
        self.tconv2=nn.ConvTranspose2d(48,24,4,stride=2,padding=1)
        self.tconv3=nn.ConvTranspose2d(24,12,4,stride=2,padding=1)
        self.tconv4=nn.ConvTranspose2d(12,3,4,stride=2,padding=1)

    def forward(self, x):
        mu, logvar=self.encode(x)
        z=self.reparametrize(mu,logvar)
        return self.decode(z), mu, logvar

    def encode(self, x):
        #In  1,1,32,32
        x=F.elu(self.conv1(x)) # 1,12,16,16
        x=F.elu(self.conv2(x)) # 1,24,8,8
        x=F.elu(self.conv3(x)) # 1,48,4,4
        x=F.elu(self.conv4(x)) # 1,96,2,2
        x=torch.flatten(x,start_dim=1) # 1,96*2*2
        x=F.elu(self.fc1(x)) # 1,16
        return self.fc21(x), self.fc22(x) # 1,latent_dim

    def decode(self, z):
        # In 1,latent_dim
        x=self.fc3(z) # 1,384
        x=torch.reshape(x,(-1,96,2,2)) # 1,96,2,2
        x=F.elu(self.tconv1(x)) # 1,48,4,4
        x=F.elu(self.tconv2(x)) # 1,24,8,8
        x=F.elu(self.tconv3(x)) # 1,12,16,16
        x=F.elu(self.tconv4(x)) # 1,3,32,32
        return torch.sigmoid(x) # 1,3,32,32

    def reparametrize(self, mu, logvar):
        std=logvar.mul(0.5).exp_()
        # sample eps from N(0,1)
        if torch.cuda.is_available():
           eps= torch.cuda.FloatTensor(std.size()).normal_()
        else:
           eps= torch.FloatTensor(std.size()).normal_()
        eps=Variable(eps)

        return eps.mul(std).add_(mu)

    # return layer ids (in 0...11) ordered for training
    def train_order_layer_ids(self):
      return [0,1,2,3,4,7,8,9,10,11,5,6] 
########################################################

class AutoEncoderCNNCL(nn.Module):
    # variational AE CNN  CIFAR10 Clustering 
    # Variational Clustering: https://arxiv.org/abs/2005.04613
    def __init__(self,K=10,L=32):
        super().__init__()
        self.K=K # clusters
        self.L=L # latent dimension
        self.repr_flag=True # turn on reparametrization by default
        self.conv1=nn.Conv2d(3, 12, 4, stride=2, padding=1)# 
        self.conv2=nn.Conv2d(12, 24, 4, stride=2, padding=1)#
        self.conv3=nn.Conv2d(24, 48, 4, stride=2, padding=1)#
        self.conv4=nn.Conv2d(48, 96, 4, stride=2, padding=1)#

        self.fc11=nn.Linear(384,128)
        self.fc12=nn.Linear(128,64)
        self.fc13=nn.Linear(64,self.K)
        self.fc21=nn.Linear(384+self.K,128)
        self.fc22=nn.Linear(128,128)
        self.fc23=nn.Linear(128,self.L)
        self.fc24=nn.Linear(128,self.L)

        self.fc14=nn.Linear(self.K,64)
        self.fc15=nn.Linear(64,64)
        self.fc16=nn.Linear(64,self.L)
        self.fc17=nn.Linear(64,self.L)

        self.fc25=nn.Linear(self.L,384)
        self.tconv1=nn.ConvTranspose2d(96,48,4,stride=2, padding=1)
        self.tconv2=nn.ConvTranspose2d(48,24,4,stride=2, padding=1)
        self.tconv3=nn.ConvTranspose2d(24,12,4,stride=2, padding=1)
        self.tconv4=nn.ConvTranspose2d(12,3,4,stride=2, padding=1)
        self.tconv5=nn.ConvTranspose2d(12,3,4,stride=2, padding=1)

    def enable_repr(self):
      self.repr_flag=True # turn on reparametrization
    def disable_repr(self):
      self.repr_flag=True # turn off reparametrization

    def forward(self, x):
        ekhat=self.encodeclus(x)
        mu_xi={}
        sig2_xi={}
        mu_b={}
        sig2_b={}
        mu_th={}
        sig2_th={}
        for ci in range(self.K):
          if torch.cuda.is_available():
           ek1=torch.cuda.FloatTensor(ekhat.shape).fill_(0)
          else:
           ek1=torch.FloatTensor(ekhat.shape).fill_(0)

          ek1[:,ci]=1
          mu_xi[ci],sig2_xi[ci]=self.encode(x, ek1)
          z=self.reparametrize(mu_xi[ci],sig2_xi[ci])
          mu_b[ci], sig2_b[ci], mu_th[ci], sig2_th[ci]=self.decode(ek1,z)
          del ek1

        return ekhat,mu_xi,sig2_xi,mu_b,sig2_b,mu_th,sig2_th

    def encodeclus(self, x):
        #In  1,3,32,32
        x=F.elu(self.conv1(x)) # 1,12,16,16
        x=F.elu(self.conv2(x)) # 1,24,8,8
        x=F.elu(self.conv3(x)) # 1,48,4,4
        x=F.elu(self.conv4(x)) # 1,96,2,2
        x1=torch.flatten(x,start_dim=1) # 1,96*2*2=384
        x=F.elu(self.fc11(x1)) # 1,128
        x=F.elu(self.fc12(x)) # 1,64
        ekhat=F.elu(self.fc13(x)) # 1,K
        # ek: parametrize q(k|x)
        return F.softmax(ekhat,dim=1) # vectors of K


    def encode(self, x, ek):
        #In  1,3,32,32 and 1,K
        x=F.elu(self.conv1(x)) # 1,12,16,16
        x=F.elu(self.conv2(x)) # 1,24,8,8
        x=F.elu(self.conv3(x)) # 1,48,4,4
        x=F.elu(self.conv4(x)) # 1,96,2,2
        x1=torch.flatten(x,start_dim=1) # 1,96*2*2=384
        y=F.elu(self.fc21(torch.cat((x1,ek),1))) # in 1,K+384, out 1,128
        y=F.elu(self.fc22(y)) # 1,128
        y1=F.elu(self.fc23(y)) # 1,L
        y2=F.elu(self.fc24(y)) # 1,L
        # mu_xi, sig2_xi: parametrize q(z|x,k)
        return y1,F.softplus(y2) # vectors of L, L

    def decode(self, ek, z):
        # In 1,K and 1,L
        x=F.elu(self.fc14(ek)) # 1,64
        x=F.elu(self.fc15(x)) # 1,64
        x1=(self.fc16(x)) # 1,L
        x2=(self.fc17(x)) # 1,L
        x=F.elu(self.fc25(z)) # 1,384
        x=torch.reshape(x,(-1,96,2,2)) # 1,96,2,2
        x=F.elu(self.tconv1(x)) # 1,48,4,4
        x=F.elu(self.tconv2(x)) # 1,24,8,8
        x=F.elu(self.tconv3(x)) # 1,12,16,16
        y1=F.elu(self.tconv4(x)) # 1,3,32,32
        y2=F.elu(self.tconv5(x)) # 1,3,32,32
        # mu_b, sig2_b, mu_th, sig2_th
        # mu_b, sig2_b: parametrize p(z|k)
        # mu_th, sig2_th: parametrize p(x|z)
        return x1,F.softplus(x2),y1,F.softplus(y2) # 1,L; 1,L; 1,3,32,32; 1,3,32,32

    def reparametrize(self, mu, sig2):
        if not self.repr_flag: # no reparametrization
          return mu

        std=sig2.sqrt()
        # sample eps from N(0,1)
        if torch.cuda.is_available():
           eps= torch.cuda.FloatTensor(std.size()).normal_()
        else:
           eps= torch.FloatTensor(std.size()).normal_()
        eps=Variable(eps)

        return eps.mul(std).add_(mu)

    # return layer ids (in 0...20) ordered for training
    def train_order_layer_ids(self):
      return [ii for ii in range(0,21)]

    # low,high: layers 2*low...2*high-1 are trained
    def train_order_block_ids(self):
      # encoder, decoder, latent space
      return [[0,4],[16,21],[4,16]]

########################################################
# Encoder, Context generator and Predictor for CPC
class EncoderCNN(nn.Module):
   def __init__(self,latent_dim=1024):
     super(EncoderCNN,self).__init__()
     self.latent_dim=latent_dim
     # 32x32 -> 16x16
     self.conv1_1=nn.Conv2d(8, 8, 4, stride=2, dilation=1, padding=1)# in 8 chan, out 8 chan, kernel 4x4
     self.conv1_2=nn.Conv2d(8, 8, 4, stride=2, dilation=2, padding=3)# in 8 chan, out 8 chan, kernel 4x4
     self.conv1_4=nn.Conv2d(8, 8, 4, stride=2, dilation=4, padding=6)# in 8 chan, out 8 chan, kernel 4x4
     self.conv1_8=nn.Conv2d(8, 8, 4, stride=2, dilation=8, padding=12)# in 8 chan, out 8 chan, kernel 4x4
     self.conv1_16=nn.Conv2d(8, 8, 4, stride=2, dilation=16, padding=24)# in 8 chan, out 8 chan, kernel 4x4
     # 16x16 -> 8x8
     self.conv2=nn.Conv2d(8*5, self.latent_dim//4, 4, stride=2,  padding=1)# in 8*5 chan, out 128 chan, kernel 4x4
     # 8x8 -> 4x4
     self.conv3=nn.Conv2d(self.latent_dim//4, self.latent_dim//2, 4, stride=2,  padding=1)# in latent/4 chan, out latent/2 chan, kernel 4x4
     # 4x4 -> 2x2
     self.conv4=nn.Conv2d(self.latent_dim//2, self.latent_dim, 4, stride=2,  padding=1)# in latent/2 chan, out latent_dim chan, kernel 4x4

   def forward(self,x):
     x1=F.elu(self.conv1_1(x))
     x2=F.elu(self.conv1_2(x))
     x4=F.elu(self.conv1_4(x))
     x8=F.elu(self.conv1_8(x))
     x16=F.elu(self.conv1_16(x))
     # concat
     x=torch.cat((x1,x2,x4,x8,x16),dim=1)
     x=F.elu(self.conv2(x))
     x=F.elu(self.conv3(x))
     x=F.elu(self.conv4(x))
     x= F.avg_pool2d(x,2).squeeze()
     return x

   # return layer ids (in 0...15) ordered for training
   def train_order_layer_ids(self):
      return [ii for ii in range(0,16)]

   # low,high: layers 2*low...2*high-1 are trained
   def train_order_block_ids(self):
      # divide to two blocks
      return [[0,5],[5,8]]


# pixelCNN  to create context from latents
class ContextgenCNN(nn.Module):
  def __init__(self,latent_dim=1024):
    super(ContextgenCNN,self).__init__()
    self.latent_dim=latent_dim
    self.conv1=nn.Conv2d(self.latent_dim,self.latent_dim//4,1,stride=1,padding=0,bias=False)
    self.conv2=nn.Conv2d(self.latent_dim//4,self.latent_dim//4,2,stride=1,padding=1,bias=False)
    self.conv3=nn.Conv2d(self.latent_dim//4,self.latent_dim//2,2,stride=1,padding=0,bias=False)
    self.conv4=nn.Conv2d(self.latent_dim//2,self.latent_dim,1,stride=1,padding=0,bias=False)

  def forward(self, x):
    x=F.elu(self.conv1(x))
    x=F.elu(self.conv2(x))
    x=F.elu(self.conv3(x))
    x=F.elu(self.conv4(x))
    return x

  # return layer ids (in 0...3) ordered for training
  def train_order_layer_ids(self):
      return [ii for ii in range(0,4)]

  # low,high: layers 2*low...2*high-1 are trained
  def train_order_block_ids(self):
      # full net
      return [[0,2]]


# prediction network
class PredictorCNN(nn.Module):
  def __init__(self,latent_dim=1024,reduced_dim=64):
    super(PredictorCNN,self).__init__()
    self.latent_dim=latent_dim
    self.reduced_dim=reduced_dim
    self.conv1=nn.Conv2d(self.latent_dim,self.reduced_dim,1,bias=False)
    self.conv2=nn.Conv2d(self.latent_dim,self.reduced_dim,1,bias=False)

  def forward(self, latents, context):
    reduced_latents=self.conv1(latents)
    prediction=self.conv2(context)
    return reduced_latents,prediction

  # return layer ids (in 0...1) ordered for training
  def train_order_layer_ids(self):
      return [ii for ii in range(0,2)]

  # low,high: layers 2*low...2*high-1 are trained
  def train_order_block_ids(self):
      # full net
      return [[0,1]]

########################################################
