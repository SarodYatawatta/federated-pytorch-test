import h5py
import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# based on paper: Data-Efficient Image Recognition with Contrastive Predictive Coding
# https://arxiv.org/abs/1905.09272
# Working with LOFAR HDF5 data files

# How many models (==workers=='number of H5 files' and SAPs)
K=4

# train K models by Federated learning
# each iteration over a subset of parameters: 1) average 2) pass back average to slaves 3) optimization step
# initialize with pre-trained models (better to use common initialization)
# loop order: loop 0: models/parameters/layers   {
#               loop 1 : {  averaging (part of the model)
#                loop 2: { epochs/databatches  { train; } } } }
# repeat this Nloop times


# model parameters
# latent dimension
Lc=128
# reduced latent dimension
Rc=32

# minibatch size (==baselines selected)
batch_size=32
Nloop=2 # how many loops over the whole network
Niter=10 # how many minibatches are considered for an epoch
Nadmm=1 # how many Federated Averaging iterations

load_model=False # enable this to load saved models
init_model=True # enable this to initialize all K models to same (not when loading saved model)
save_model=True # save model
be_verbose=True

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

#import torchvision
#import torchvision.transforms as transforms

def get_data_minibatch(filename,batch_size=2,patch_size=32,SAP='0'):
  # open LOFAR H5 file, read data from a SAP,
  # randomly select number of baselines equal to batch_size
  # and sample patches and return input for training
  f=h5py.File(filename,'r')
  # select a dataset SAP (int8)
  g=f['measurement']['saps'][SAP]['visibilities']
  # scale factors for the dataset (float32)
  h=f['measurement']['saps'][SAP]['visibility_scale_factors']

  (nbase,ntime,nfreq,npol,ncomplex)=g.shape
  # h shape : nbase, nfreq, npol

  x=torch.zeros(batch_size,8,ntime,nfreq).to(mydevice,non_blocking=True)
  # randomly select baseline subset
  baselinelist=np.random.randint(0,nbase,batch_size)

  ck=0
  for mybase in baselinelist:
   # this is 8 channels in torch tensor
   for ci in range(4):
    # get visibility scales
    scalefac=torch.from_numpy(h[mybase,:,ci]).to(mydevice,non_blocking=True)
    # add missing (time) dimension
    scalefac=scalefac[None,:]
    x[ck,2*ci]=torch.from_numpy(g[mybase,:,:,ci,0])
    x[ck,2*ci]=x[ck,2*ci]*scalefac
    x[ck,2*ci+1]=torch.from_numpy(g[mybase,:,:,ci,1])
    x[ck,2*ci+1]=x[ck,2*ci+1]*scalefac
   ck=ck+1

  #torchvision.utils.save_image(x[0,0].data, 'sample.png')
  stride = patch_size//2 # patch stride (with 1/2 overlap)
  num_channels=8 # 4x2 polarizationx(real,imag)
  y = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
  # get new shape
  (nbase1,nchan1,patchx,patchy,nx,ny)=y.shape
  # create a new tensor
  y1=torch.zeros([nbase1*patchx*patchy,nchan1,nx,ny]).to(mydevice,non_blocking=True)

  # copy data ordered according to the patches
  ck=0
  for ci in range(patchx):
   for cj in range(patchy):
     y1[ck*nbase1:(ck+1)*nbase1,:,:,:]=y[:,:,ci,cj,:,:]
     ck=ck+1

  y = y1
  del x,y1 
  # note: nbatch = batch_size x patchx x patchy
  #(nbatch,nchan,nxx,nyy)=y.shape

  # do some rough cleanup of data
  ##y[y!=y]=0 # set NaN,Inf to zero
  torch.clamp(y,-1e6,1e6) # clip high values
  
  return patchx,patchy,y


# import models, utils
from simple_models import *
from simple_utils  import *


encoder_dict={}
contextgen_dict={}
predictor_dict={}

for ck in range(K):
  encoder_dict[ck]=EncoderCNN(latent_dim=Lc).to(mydevice)
  contextgen_dict[ck]=ContextgenCNN(latent_dim=Lc).to(mydevice)
  predictor_dict[ck]=PredictorCNN(latent_dim=Lc,reduced_dim=Rc).to(mydevice)

  if load_model:
    checkpoint=torch.load('./encoder.model',map_location=mydevice)
    encoder_dict[ck].load_state_dict(checkpoint['model_state_dict'])
    encoder_dict[ck].train()
    checkpoint=torch.load('./contextgen.model',map_location=mydevice)
    contextgen_dict[ck].load_state_dict(checkpoint['model_state_dict'])
    contextgen_dict[ck].train()
    checkpoint=torch.load('./predictor.model',map_location=mydevice)
    predictor_dict[ck].load_state_dict(checkpoint['model_state_dict'])
    predictor_dict[ck].train()


### specify data files and subarray pointings (SAPs)
# both should have K items
#file_list=['/home/sarod/L785751.MS_extract.h5','/home/sarod/L785751.MS_extract.h5']
#file_list=file_list+['/home/sarod/L785747.MS_extract.h5','/home/sarod/L785757.MS_extract.h5']
file_list=['../../drive/My Drive/Colab Notebooks/L785751.MS_extract.h5','../../drive/My Drive/Colab Notebooks/L785751.MS_extract.h5']
file_list=file_list+['../../drive/My Drive/Colab Notebooks/L785747.MS_extract.h5','../../drive/My Drive/Colab Notebooks/L785757.MS_extract.h5']
sap_list=['1','2']
sap_list=sap_list+['0','0']
assert len(sap_list)==K and len(file_list)==K

################################################################################ Loss function
# loss function
def InfoNCE(z,zhat):
  '''
   z: latents size: batch x channel x patchx x patchy
   zhat: prediction size : batch x channel x patchx x patchy
   for patch ci,cj, positive sample : exp(zhat(ci,cj)^T z(ci,cj))
    negative sample: exp(zhat(ci,cj)^T z(.,.)) where z(.,.) is not ci,cj (another patch)
    product (patchx*patchy)^2 matrix: negative samples: row ci, all columns except cj
  '''
  # enable this for debugging
  #torch.autograd.set_detect_anomaly(True)
  assert z.shape==zhat.shape
  (nbatch,nchan,patchx,patchy)=z.shape
  Z=z.view([-1,patchx*patchy])
  Zhat=zhat.view([-1,patchx*patchy])

  # find (normalized) inner products of patchx*patchy values
  zz=torch.zeros(patchx*patchy,patchx*patchy).to(mydevice,non_blocking=True)
  for ci in range(patchx*patchy):
    Znrm=torch.norm(Z[:,ci])
    for cj in range(patchx*patchy):
      zz[ci,cj]=torch.dot(Z[:,ci],Zhat[:,cj])/(Znrm*torch.norm(Zhat[:,cj]))

  # positive sample: diagonal, negative samples: off-diagonal of each row
  loss=0
  for ci in range(patchx*patchy):
    numerator=torch.exp(zz[ci,ci])
    denominator=numerator
    for cj in [ii for ii in range(patchx*patchy) if ii != ci]:
        denominator=denominator+torch.exp(zz[ci,cj])
    loss=loss-torch.log(numerator/denominator+1e-6)

  return loss

##############################################################################################

if init_model:
 for ck in range(K):
   torch.manual_seed(0)
   encoder_dict[ck].apply(init_weights)
   contextgen_dict[ck].apply(init_weights)
   predictor_dict[ck].apply(init_weights)

from lbfgsnew import LBFGSNew

############### loop 00 (over the full models)
for nloop in range(Nloop):
 ### loop 0 (over full model, layer wise)
 for mdl in range(3): 
   if mdl==0:
     ### 0 Encoder
     Bi=encoder_dict[0].train_order_block_ids()
     for ck in range(K):
        freeze_all_layers(contextgen_dict[ck])
        freeze_all_layers(predictor_dict[ck])
   elif mdl==1:
     ### 1 Context generator
     Bi=contextgen_dict[0].train_order_block_ids()
     for ck in range(K):
        freeze_all_layers(encoder_dict[ck])
        freeze_all_layers(predictor_dict[ck])
   else:
     ### 2 Predictor
     Bi=predictor_dict[0].train_order_block_ids()
     for ck in range(K):
        freeze_all_layers(encoder_dict[ck])
        freeze_all_layers(contextgen_dict[ck])

   for ci in range(len(Bi)):
     for ck in range(K):
       if mdl==0:
         unfreeze_one_block(encoder_dict[ck],Bi[ci])
       elif mdl==1:
         unfreeze_one_block(contextgen_dict[ck],Bi[ci])
       else:
         unfreeze_one_block(predictor_dict[ck],Bi[ci])

     if mdl==0:
       trainable=filter(lambda p: p.requires_grad, encoder_dict[0].parameters())
     elif mdl==1:
       trainable=filter(lambda p: p.requires_grad, contextgen_dict[0].parameters())
     else:
       trainable=filter(lambda p: p.requires_grad, predictor_dict[0].parameters())
     params_vec1=torch.cat([x.view(-1) for x in list(trainable)])
     N=params_vec1.numel()
     del trainable,params_vec1


     z=torch.zeros(N,dtype=torch.float,requires_grad=False).to(mydevice,non_blocking=True)

     opt_dict={}
     for ck in range(K):
       if mdl==0:
          #opt_dict[ck]=optim.Adam(filter(lambda p: p.requires_grad, encoder_dict[ck].parameters()),lr=0.0001)
          opt_dict[ck]=LBFGSNew(filter(lambda p: p.requires_grad, encoder_dict[ck].parameters()), history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)
       elif mdl==1:
          #opt_dict[ck]=optim.Adam(filter(lambda p: p.requires_grad, contextgen_dict[ck].parameters()),lr=0.0001)
          opt_dict[ck]=LBFGSNew(filter(lambda p: p.requires_grad, contextgen_dict[ck].parameters()), history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)
       else:
          #opt_dict[ck]=optim.Adam(filter(lambda p: p.requires_grad, predictor_dict[ck].parameters()),lr=0.0001)
          opt_dict[ck]=LBFGSNew(filter(lambda p: p.requires_grad, predictor_dict[ck].parameters()), history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)

     for nadmm in range(Nadmm):
        for ck in range(K):
           for niter in range(Niter):
              patchx,patchy,y=get_data_minibatch(filename=file_list[ck],SAP=sap_list[ck],batch_size=batch_size)
              y=Variable(y).to(mydevice,non_blocking=True)
              def closure():
                if torch.is_grad_enabled():
                  opt_dict[ck].zero_grad()
                output=encoder_dict[ck](y)
                # reshape output to patchx*patchy  grid of vectors
                output=output.contiguous().view(batch_size,patchx,patchy,-1)
                # permute to make last dims patchx * patchy
                # num. latents==channels now
                latents=output.permute([0,3,1,2])
                latents=latents.contiguous()
                context=contextgen_dict[ck](latents)
                # reduce latent dim to Rc for CPC calculation
                reduced_latents,prediction=predictor_dict[ck](latents,context)
                # loss function
                loss=InfoNCE(reduced_latents,prediction)
                if loss.requires_grad:
                   loss.backward()
                   # clip gradient values
                   #torch.nn.utils.clip_grad_value_(solvable_parameters,1e3)
                   if be_verbose:
                     print('%d %d %d %f'%(nadmm,ck,niter,loss.data.item()))
                return loss
              opt_dict[ck].step(closure)
              del y

        # Federated averaging
        x_dict={}
        for ck in range(K):
           if mdl==0:
            x_dict[ck]=get_trainable_values(encoder_dict[ck],mydevice)
           elif mdl==1:
            x_dict[ck]=get_trainable_values(contextgen_dict[ck],mydevice)
           else:
            x_dict[ck]=get_trainable_values(predictor_dict[ck],mydevice)
        znew=torch.zeros(x_dict[0].shape).to(mydevice,non_blocking=True)
        for ck in range(K):
            znew=znew+x_dict[ck]
        znew=znew/K
        del x_dict

        dual_residual=torch.norm(z-znew).item()/N # per parameter
        print('dual (N=%d,iter=%d,loop=%d,model=%d,block=%d,avg=%d)=%e'%(N,niter,nloop,mdl,ci,nadmm,dual_residual))
        z=znew
        for ck in range(K):
           if mdl==0:
            put_trainable_values(encoder_dict[ck],z)
           elif mdl==1:
            put_trainable_values(contextgen_dict[ck],z)
           else:
            put_trainable_values(predictor_dict[ck],z)


# save models
if save_model:
  for ck in range(K):
    torch.save({
     'model_state_dict':encoder_dict[ck].state_dict()
     },'encoder'+str(ck)+'.model')
    torch.save({
     'model_state_dict':contextgen_dict[ck].state_dict()
     },'contextgen'+str(ck)+'.model')
    torch.save({
     'model_state_dict':predictor_dict[ck].state_dict()
     },'predictor'+str(ck)+'.model')
