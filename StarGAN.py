#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from comet_ml import Experiment
#experiment = Experiment()


# In[ ]:


import os
import itertools
import argparse
from tqdm import tqdm
from pickle import load, dump

import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


class FReLU(nn.Module):
    def __init__(self, n_channel, kernel=3, stride=1, padding=1):
        super().__init__()
        self.funnel_condition = nn.Conv2d(n_channel, n_channel, kernel_size=kernel,stride=stride, padding=padding, groups=n_channel)
        self.norm = nn.InstanceNorm2d(n_channel)

    def forward(self, x):
        tx = self.norm(self.funnel_condition(x))
        out = torch.max(x, tx)
        return out


# In[ ]:


class Mish(nn.Module):
    @staticmethod
    @torch.jit.script
    def mish(x):
        return x * torch.tanh(F.softplus(x))
    
    def forward(self, x):
        return Mish.mish(x)


# In[ ]:


class SelfAttention(nn.Module):
    def __init__(self, input_nc, spectral_norm=False):
        super().__init__()
        
        # Pointwise Convolution
        if spectral_norm:
            self.query_conv = nn.utils.spectral_norm(nn.Conv2d(input_nc, input_nc // 8, kernel_size=1))
            self.key_conv = nn.utils.spectral_norm(nn.Conv2d(input_nc, input_nc // 8, kernel_size=1))
            self.value_conv = nn.utils.spectral_norm(nn.Conv2d(input_nc, input_nc, kernel_size=1))
        else:
            self.query_conv = nn.Conv2d(input_nc, input_nc // 8, kernel_size=1)
            self.key_conv = nn.Conv2d(input_nc, input_nc // 8, kernel_size=1)
            self.value_conv = nn.Conv2d(input_nc, input_nc, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        proj_query = self.query_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3]).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])
        s = torch.bmm(proj_query, proj_key) # バッチ毎の行列乗算
        attention_map_T = self.softmax(s)
        
        proj_value = self.value_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])
        o = torch.bmm(proj_value, attention_map_T)
        
        o = o.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        out = x + self.gamma * o
        
        return out#, attention_map_T.permute(0, 2, 1)


# In[ ]:


class ResidualSEBlock(nn.Module):
    def __init__(self, in_features, reduction=16):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.residual = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features),
            FReLU(in_features),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features, in_features // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze).view(residual.size(0), residual.size(1), 1, 1)
        return F.relu(residual * excitation.expand_as(residual) + shortcut)


# In[ ]:


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, class_dim, n_residual_blocks=8):
        super().__init__()
        
        in_features = 64
        model = [
            nn.Conv2d(input_nc + class_dim, in_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            FReLU(in_features)
        ]
        
        # Downsampling
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                FReLU(out_features)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ ResidualSEBlock(in_features) ]
        
        model += [SelfAttention(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                FReLU(out_features)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.Conv2d(in_features, output_nc, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.model(x)


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, input_nc, class_dim, image_size):
        super().__init__()

        n_features = 64
        model = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, n_features, kernel_size=4, stride=2, padding=1)),
            Mish()
        ]

        model += [
            nn.utils.spectral_norm(nn.Conv2d(n_features, n_features * 2, kernel_size=4, stride=2, padding=1)),
            Mish()
        ]
        n_features *= 2
        
        model += [
            nn.utils.spectral_norm(nn.Conv2d(n_features, n_features * 2, kernel_size=4, stride=2, padding=1)),
            Mish()
        ]
        n_features *= 2
        
        model += [
            nn.utils.spectral_norm(nn.Conv2d(n_features, n_features * 2, kernel_size=4, stride=2, padding=1)),
            Mish()
        ]
        n_features *= 2
        
        model += [SelfAttention(n_features, spectral_norm=True)]
        
        self.model = nn.Sequential(*model)

        # For PatchGAN
        self.conv_patch = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(n_features, 1, kernel_size=3, stride=1, padding=1)),
            nn.Sigmoid()
        )
        
        kernel_size = int(image_size / np.power(2, 4))
        self.conv_class = nn.utils.spectral_norm(nn.Conv2d(n_features, class_dim, kernel_size=kernel_size, stride=1, padding=0))

    def forward(self, x):
        x =  self.model(x)
        
        out = self.conv_patch(x)
        
        out_class = self.conv_class(x)
        out_class = out_class.view(out_class.size(0), out_class.size(1))
        
        return out, out_class


# In[ ]:


class Util:
    @staticmethod
    def loadImages(batch_size, folder_path, size):
        imgs = ImageFolder(folder_path, transform=transforms.Compose([
            transforms.Resize(int(size)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(degrees=30),
            #transforms.RandomPerspective(),
            #transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5),
            transforms.ToTensor()
        ]))
        return DataLoader(imgs, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=os.cpu_count(), pin_memory=True)
    
    @staticmethod
    def showImage(image):
        get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        
        PIL = transforms.ToPILImage()
        
        image = PIL(image)
        fig = plt.figure(dpi=16)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(image)
        plt.show()


# In[ ]:


class Solver:
    def __init__(self, args):
        has_cuda = torch.cuda.is_available() if not args.cpu else False
        self.device = torch.device("cuda" if has_cuda else "cpu")
        self.dtype = torch.cuda.FloatTensor if has_cuda else torch.FloatTensor
        self.itype = torch.cuda.LongTensor if has_cuda else torch.LongTensor
        
        self.args = args
        self.num_channel = 3
        
        self.dataloader = Util.loadImages(self.args.batch_size, self.args.image_dir, self.args.image_size)
        self.num_classes = len(os.listdir(self.args.image_dir))
        
        self.netG = Generator(self.num_channel, self.num_channel, class_dim=self.num_classes).to(self.device)
        self.netD = Discriminator(self.num_channel, class_dim=self.num_classes, image_size=self.args.image_size).to(self.device)
        self.state_loaded = False

        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)

        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=self.args.lr, betas=(0, 0.9))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.args.lr * self.args.mul_lr_dis, betas=(0, 0.9))
        
        self.pseudo_aug = 0.0
        self.epoch = 0
        
    def weights_init(self, module):
        if type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d or type(module) == nn.Linear:
            nn.init.kaiming_normal_(module.weight)
            module.bias.data.fill_(0)
            
    def save_state(self, num):
        self.netG.cpu()
        self.netD.cpu()
        torch.save(self.netG.state_dict(), os.path.join(self.args.weight_dir, f'weight_G.{num}.pth'))
        torch.save(self.netD.state_dict(), os.path.join(self.args.weight_dir, f'weight_D.{num}.pth'))
        self.netG.to(self.device)
        self.netD.to(self.device)
            
    def load_state(self):
        if (os.path.exists('weight_G.pth') and os.path.exists('weight_D.pth')):
            self.netG.load_state_dict(torch.load('weight_G.pth', map_location=self.device))
            self.netD.load_state_dict(torch.load('weight_D.pth', map_location=self.device))
            self.state_loaded = True
            print('Loaded network state.')
            
    def save_resume(self):
        with open(os.path.join('.', 'resume.pkl'), 'wb') as f:
            dump(self, f)
    
    @staticmethod
    def load(args, resume=True):
        if resume and os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                print('Load resume.')
                solver = load(f)
                solver.args = args
                return solver
        else:
            return Solver(args)
        
    def trainGAN(self, epoch, iters, max_iters, real_img, real_label):
        ### Train CycleGAN with WGAN-gp.
        
        loss = {}
        L1_loss = nn.L1Loss()
        BCE_loss = nn.BCEWithLogitsLoss(reduction='mean')

        # Generate target domain labels randomly.
        target_label = torch.LongTensor([random.randrange(self.num_classes) for _ in range(real_label.size(0))])
        target_label = self.label2onehot(target_label, self.num_classes).to(self.device)
        
        # ================================================================================ #
        #                             Train the discriminator                              #
        # ================================================================================ #

        # Compute loss with real images.
        real_src_score, real_cls_score = self.netD(real_img)
        real_src_loss = - torch.mean(real_src_score)
        real_cls_loss = BCE_loss(real_cls_score, real_label)
        
        # Compute loss with fake images.
        fake_img = self.netG(real_img, target_label)
        fake_src_score, _ = self.netD(fake_img)
        
        fake_src_loss = torch.mean(fake_src_score)
        #p = random.uniform(0, 1)
        #if 1 - self.pseudo_aug < p:
        #    fake_src_loss = - torch.mean(fake_src_score)
        #else:
        #    fake_src_loss = torch.mean(fake_src_score)
        #
        ## Update Probability Augmentation.
        #lz = (torch.sign(torch.logit(real_src_score)).mean()
        #      - torch.sign(torch.logit(fake_src_score)).mean()) / 2
        #if lz > self.args.aug_threshold:
        #    self.pseudo_aug += self.args.aug_increment
        #else:
        #    self.pseudo_aug -= self.args.aug_increment
        #self.pseudo_aug = min(1, max(0, self.pseudo_aug))
        
        # Total loss.
        d_loss = real_src_loss + fake_src_loss + self.args.lambda_cls * real_cls_loss
        
        # Backward and optimize.
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        
        # ================================================================================ #
        #                               Train the generator                                #
        # ================================================================================ #
        
        # Compute loss with reconstruction loss.
        fake_img = self.netG(real_img, target_label)
        recon_img = self.netG(fake_img, real_label)

        fake_src_score, fake_cls_score = self.netD(fake_img)
        fake_src_loss = - torch.mean(fake_src_score)
        fake_cls_loss = BCE_loss(fake_cls_score, target_label)
        reconst_loss = L1_loss(recon_img, real_img)

        # Total loss.
        g_loss = fake_src_loss + self.args.lambda_cls * fake_cls_loss + self.args.lambda_recon * reconst_loss

        # Backward and optimize.
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        # Logging.
        loss['G/loss'] = g_loss.item()
        loss['G/fake_cls_loss'] = fake_cls_loss.item()
        loss['G/reconst_loss'] = reconst_loss.item()
              
        # Logging.
        loss['D/loss'] = d_loss.item()
        loss['D/real_cls_loss'] = real_cls_loss.item()
        #loss['Augment/prob'] = self.pseudo_aug
    
        # Save
        if iters == max_iters:
            self.save_state(epoch)
            img_name = str(epoch) + '_' + str(iters) + '.png'
            img_path = os.path.join(self.args.result_dir, img_name)
            self.save_sample(real_img, fake_img, img_path)
        
        return loss
            
    def label2onehot(self, labels, dim):
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out
    
    def train(self, resume=True):
        print(f'Use Device: {self.device}')
        torch.backends.cudnn.benchmark = True
        
        self.netG.train()
        self.netD.train()
        
        hyper_params = {}
        hyper_params['Image Dir'] = self.args.image_dir
        hyper_params['Image Size'] = self.args.image_size
        hyper_params['Result Dir'] = self.args.result_dir
        hyper_params['Wieght Dir'] = self.args.weight_dir
        hyper_params['Learning Rate'] = self.args.lr
        hyper_params["Mul Discriminator's LR"] = self.args.mul_lr_dis
        hyper_params['Epochs'] = self.args.num_epochs
        hyper_params['Batch Size'] = self.args.batch_size
        hyper_params['lambda_cls'] = self.args.lambda_cls
        hyper_params['lambda_recon'] = self.args.lambda_recon
        #hyper_params['Probability Aug-Threshold'] = self.args.aug_threshold
        #hyper_params['Probability Aug-Increment'] = self.args.aug_increment

        for key in hyper_params.keys():
            print(f'{key}: {hyper_params[key]}')
        #experiment.log_parameters(hyper_params)
        
        max_iters = len(iter(self.dataloader))
        
        for epoch in range(1, self.args.num_epochs + 1):
            if epoch < self.epoch:
                continue
            self.epoch = epoch + 1
            
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            
            for iters, (data, label) in enumerate(tqdm(self.dataloader)):
                iters += 1
                
                data = data.to(self.device, non_blocking=True)
                label = self.label2onehot(label, self.num_classes).to(self.device, non_blocking=True)
                
                loss = self.trainGAN(epoch, iters, max_iters, data, label)
                epoch_loss_D += loss['D/loss']
                epoch_loss_G += loss['G/loss'] if 'G/loss' in loss else 0
                #experiment.log_metrics(loss)
                    
            print(f'{epoch} / {self.args.num_epochs}: Loss_G {epoch_loss_G}, Loss_D {epoch_loss_D}')
            
            if resume:
                self.save_resume()
              
    def save_sample(self, real_img, fake_img, img_path):
        N = real_img.size(0)
        img = torch.cat((real_img.data, fake_img.data), dim=0)
        save_image(img, img_path, nrow=N)
        
        #Util.showImage(real_img[0])
        #Util.showImage(fake_img[0])
    
    def generate(self, num):
        self.netG.eval()

        for _ in range(num):
            data, label = next(iter(self.dataloader))
            real_img = data[0]
            real_label = label[0]
            real_img = real_img.unsqueeze(0).to(self.device)
            
            for i in range(self.num_classes):
                label = i
                onehot_label = self.label2onehot(torch.full((1,), label, dtype=torch.long), self.num_classes).to(self.device)
                fake_img = self.netG(real_img, onehot_label).data
                self.save_sample(real_img, fake_img, os.path.join(self.args.result_dir, f'generated_{real_label}-to-{i}_{time.time()}.png'))
                
        print('New picture was generated.')


# In[ ]:


def main(args):
    solver = Solver.load(args, resume=not args.noresume)
    solver.load_state()
        
    if args.generate > 0:
        solver.generate(args.generate)
        exit()
    
    solver.train(not args.noresume)
    #experiment.end()


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mul_lr_dis', type=float, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_recon', type=float, default=10)
    #parser.add_argument('--aug_threshold', type=float, default=0.6)
    #parser.add_argument('--aug_increment', type=float, default=0.01)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--generate', type=int, default=0)
    parser.add_argument('--noresume', action='store_true')

    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)

