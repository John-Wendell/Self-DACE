import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np

# def rgb_to_grayscale(s):
#     return (0.2989*s[:,0,:,:]+ 0.5870*s[:,1,:,:] + 0.1140*s[:,2,:,:]).unsqueeze(1)
def rgb_to_grayscale(s):
    return ((s[:,0,:,:]+ s[:,1,:,:] +s[:,2,:,:])/3).unsqueeze(1)

def gradient2(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
    gradient2_h = (img[:,:,4:,:] - img[:,:,:height-4,:]).abs()
    gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
    gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
    gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
    return gradient_h*gradient2_h, gradient_w*gradient2_w

class L_localcolor(nn.Module):
    def __init__(self):
        super(L_localcolor, self).__init__()
        self.pool = nn.AvgPool2d(4)
    def forward(self, x, org):  # def forward(self, image, src,org):
        # org = torch.clamp(org, min=1e-10)
        mean_rgb = self.pool(org)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        
        # x = torch.clamp(x, min=1e-10)
        mean_rgb1 = self.pool(x)
        mr1, mg1, mb1 = torch.split(mean_rgb1, 1, dim=1)
        
        r = mr / (mr + mg + mb + 0.0001)
        g = mg / (mr + mg + mb + 0.0001)
        b = mb / (mr + mg + mb + 0.0001)
        r1 = mr1 / (mr1 + mg1 + mb1 + 0.0001)
        g1 = mg1 / (mr1 + mg1 + mb1 + 0.0001)
        b1 = mb1 / (mr1 + mg1 + mb1 + 0.0001)
        k = (torch.pow(r - r1, 2) + torch.pow(g - g1, 2) + torch.pow(b - b1, 2))#*(torch.pow(mr+mg+mb, 2)+1)#*torch.exp(0.5*(mr+mg+mb))
        return k


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()
        self.pool = nn.AvgPool2d(4)
    def forward(self, x):
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        r = mr / (mr + mg + mb + 1e-10)
        g = mg / (mr + mg + mb + 1e-10)
        b = mb / (mr + mg + mb + 1e-10)
        k = (torch.pow(r - 1/3, 2) + torch.pow(g - 1/3, 2) + torch.pow(b - 1/3, 2))#*(torch.pow(mr+mg+mb, 2)+1)#*torch.exp(0.5*(mr+mg+mb))
        return k


class L_SMO(nn.Module):
    def __init__(self):
        super(L_SMO, self).__init__()


    def forward(self, x1):
        # batch_size = x1.size()[0]
        # h_x = x1.size()[2]
        # w_x = x1.size()[3]
        # count_h = (x1.size()[2] - 1) * x1.size()[3]
        # count_w = x1.size()[2] * (x1.size()[3] - 1)
        h_tv = torch.pow((x1[:, :, 1:, :] - x1[:, :, :-1, :]), 2)
        w_tv = torch.pow((x1[:, :, :, 1:] - x1[:, :, :, :-1]), 2)
        return torch.mean(h_tv) + torch.mean(w_tv)


class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x, org):
        mean0 = self.pool(org)
        mr0, mg0, mb0 = torch.split(mean0, 1, dim=1)
        xs0 = mr0 + mg0 + mb0 + 1e-10
        xv0 = 1 - torch.pow(torch.pow((mr0/xs0)-1/3, 2) + torch.pow((mg0/xs0)-1/3, 2) + torch.pow((mb0/xs0)-1/3, 2), 0.5)
        xv0 = 3*xv0
        
        mean = self.pool(x)
        mr, mg, mb = torch.split(mean, 1, dim=1)
        xv = mr + mg + mb
        dp = xv -  self.mean_val * xv0
        d = torch.pow(dp, 2)# * (1 + torch.pow(200, dp))
        return d


