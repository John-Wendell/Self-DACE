import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import math
from torchvision.models.vgg import vgg16
import numpy as np

# def rgb_to_grayscale(s):
#     return (0.2989*s[:,0,:,:]+ 0.5870*s[:,1,:,:] + 0.1140*s[:,2,:,:]).unsqueeze(1)
def rgb_to_grayscale(s):
    return ((s[:,0,:,:]+ s[:,1,:,:] +s[:,2,:,:])/3).unsqueeze(1)


class L_gradient(nn.Module):
    def __init__(self):
        super(L_gradient, self).__init__()

    def forward(self, x1, x2):  # def forward(self, image, src,org):
        batch_size = x1.size()[0]
        h_x = x1.size()[2]
        w_x = x1.size()[3]
        count_h = (x1.size()[2] - 1) * x1.size()[3]
        count_w = x1.size()[2] * (x1.size()[3] - 1)
        h_g1 = x1[:, :, 1:, :] - x1[:, :, :h_x - 1, :]
        h_g2 = x2[:, :, 1:, :] - x2[:, :, :h_x - 1, :]
        w_g1 = x1[:, :, :, 1:] - x1[:, :, :, :w_x - 1]
        w_g2 = x2[:, :, :, 1:] - x2[:, :, :, :w_x - 1]
        h_tv = torch.pow((h_g1 - h_g2), 2).sum()
        w_tv = torch.pow((w_g1 - w_g2), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class L_localcolor(nn.Module):
    def __init__(self,win=4):
        super(L_localcolor, self).__init__()
        self.pool = nn.AvgPool2d(win)
    def forward(self, x, org):  # def forward(self, image, src,org):
        org = torch.clamp(org, min=1e-10)
        mean_rgb = self.pool(org)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        
        x = torch.clamp(x, min=1e-10)
        mean_rgb1 = self.pool(x)
        mr1, mg1, mb1 = torch.split(mean_rgb1, 1, dim=1)
        r = mr / (mr + mg + mb)
        g = mg / (mr + mg + mb)
        b = mb / (mr + mg + mb)
        r1 = mr1 / (mr1 + mg1 + mb1)
        g1 = mg1 / (mr1 + mg1 + mb1)
        b1 = mb1 / (mr1 + mg1 + mb1)
        k = (torch.pow(r - r1, 2) + torch.pow(g - g1, 2) + torch.pow(b - b1, 2))#*(torch.pow(mr+mg+mb, 2)+1)#*torch.exp(0.5*(mr+mg+mb))
        return k


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()
        self.pool = nn.AvgPool2d(4)
    def forward(self, x):
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        r = mr / (mr + mg + mb + 0.0001)
        g = mg / (mr + mg + mb + 0.0001)
        b = mb / (mr + mg + mb + 0.0001)
        k = (torch.pow(r - 1/3, 2) + torch.pow(g - 1/3, 2) + torch.pow(b - 1/3, 2))#*(torch.pow(mr+mg+mb, 2)+1)#*torch.exp(0.5*(mr+mg+mb))
        return k


class L_SMO(nn.Module):

    def __init__(self):
        super(L_SMO, self).__init__()

    def forward(self, x1):
        batch_size = x1.size()[0]
        h_x = x1.size()[2]
        w_x = x1.size()[3]
        count_h = (x1.size()[2] - 1) * x1.size()[3]
        count_w = x1.size()[2] * (x1.size()[3] - 1)
        h_tv = torch.pow((x1[:, :, 1:, :] - x1[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x1[:, :, :, 1:] - x1[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x, org):
        mean0 = self.pool(org)
        mr0, mg0, mb0 = torch.split(mean0, 1, dim=1)
        xs0 = mr0 + mg0 + mb0 + 0.0001
        xv0 = 1 - torch.pow(torch.pow((mr0/xs0)-1/3, 2) + torch.pow((mg0/xs0)-1/3, 2) + torch.pow((mb0/xs0)-1/3, 2), 0.5)
        xv0 = 3*xv0
        mean = self.pool(x)
        mr, mg, mb = torch.split(mean, 1, dim=1)
        xv = mr + mg + mb
        dp = xv -  self.mean_val * xv0
        d = torch.pow(dp, 2)# * (1 + torch.pow(200, dp))
        return d

########### define a loss function which calculates the cosine similarity between two images
class L_cos(nn.Module):
    def __init__(self):
        super(L_cos, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, x1, x2):
        return torch.mean(self.cos(x1, x2))
