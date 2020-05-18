import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
from dataloader import dataloader
from models import VAE_GAN,Discriminator
from utils import show_and_save

data_loader=dataloader(64)
gen=VAE_GAN().to(device)
discrim=Discriminator().to(device)
real_batch = next(iter(data_loader))
show_and_save("training" ,make_grid((real_batch[0]*0.5+0.5).cpu(),8))

epochs=25
lr=3e-4
alpha=0.1
gamma=15

criterion=nn.BCELoss().to(device)
optim_E=torch.optim.RMSprop(gen.encoder.parameters(), lr=lr)
optim_D=torch.optim.RMSprop(gen.decoder.parameters(), lr=lr)
optim_Dis=torch.optim.RMSprop(discrim.parameters(), lr=lr*alpha)
z_fixed=Variable(torch.randn((64,128))).to(device)
x_fixed=Variable(real_batch[0]).to(device)

for epoch in range(epochs):
  prior_loss_list,gan_loss_list,recon_loss_list=[],[],[]
  dis_real_list,dis_fake_list,dis_prior_list=[],[],[]
  for i, (data,_) in enumerate(data_loader, 0):
    bs=data.size()[0]
    
    ones_label=Variable(torch.ones(bs,1)).to(device)
    zeros_label=Variable(torch.zeros(bs,1)).to(device)
    zeros_label1=Variable(torch.zeros(64,1)).to(device)
    datav = Variable(data).to(device)
    mean, logvar, rec_enc = gen(datav)
    z_p = Variable(torch.randn(64,128)).to(device)
    x_p_tilda = gen.decoder(z_p)
 
    output = discrim(datav)[0]
    errD_real = criterion(output, ones_label)
    dis_real_list.append(errD_real.item())
    output = discrim(rec_enc)[0]
    errD_rec_enc = criterion(output, zeros_label)
    dis_fake_list.append(errD_rec_enc.item())
    output = discrim(x_p_tilda)[0]
    errD_rec_noise = criterion(output, zeros_label1)
    dis_prior_list.append(errD_rec_noise.item())
    gan_loss = errD_real + errD_rec_enc + errD_rec_noise
    gan_loss_list.append(gan_loss.item())
    optim_Dis.zero_grad()
    gan_loss.backward(retain_graph=True)
    optim_Dis.step()


    output = discrim(datav)[0]
    errD_real = criterion(output, ones_label)
    output = discrim(rec_enc)[0]
    errD_rec_enc = criterion(output, zeros_label)
    output = discrim(x_p_tilda)[0]
    errD_rec_noise = criterion(output, zeros_label1)
    gan_loss = errD_real + errD_rec_enc + errD_rec_noise
    

    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    err_dec = gamma * rec_loss - gan_loss 
    recon_loss_list.append(rec_loss.item())
    optim_D.zero_grad()
    err_dec.backward(retain_graph=True)
    optim_D.step()
    
    mean, logvar, rec_enc = gen(datav)
    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
    prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
    prior_loss_list.append(prior_loss.item())
    err_enc = prior_loss + 5*rec_loss

    optim_E.zero_grad()
    err_enc.backward(retain_graph=True)
    optim_E.step()
    
    if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tRec_loss: %.4f\tdis_real_loss: %0.4f\tdis_fake_loss: %.4f\tdis_prior_loss: %.4f'
                  % (epoch,epochs, i, len(data_loader),
                     gan_loss.item(), prior_loss.item(),rec_loss.item(),errD_real.item(),errD_rec_enc.item(),errD_rec_noise.item()))

  
   
   
   
   
    
    
  


    

  b=gen(x_fixed)[2]
  b=b.detach()
  c=gen.decoder(z_fixed)
  c=c.detach()
  show_and_save('MNISTrec_noise_epoch_%d.png' % epoch ,make_grid((c*0.5+0.5).cpu(),8))
  show_and_save('MNISTrec_epoch_%d.png' % epoch ,make_grid((b*0.5+0.5).cpu(),8))



