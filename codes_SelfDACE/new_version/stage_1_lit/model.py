import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def reflection(im):
    mr, mg, mb = torch.split(im, 1, dim=1)
    r = mr / (mr + mg + mb + 1e-10)
    g = mg / (mr + mg + mb + 1e-10)
    b = mb / (mr + mg + mb + 1e-10)
    return torch.cat([r, g, b], dim=1)


def luminance(s):
    return ((s[:, 0, :, :] + s[:, 1, :, :] + s[:, 2, :, :])).unsqueeze(1)



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        features = 16
        
        self.dconv_down1 = DoubleConv(1, features)
        self.dconv_down2 = DoubleConv(features, features)
        self.dconv_down3 = DoubleConv(features, features)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up2 = DoubleConv(2*features, features)
        self.dconv_up1 = DoubleConv(2*features, features)
        
        self.conv_last = nn.Conv2d(features, features, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        x = self.dconv_down3(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

class light_net(nn.Module):

    def __init__(self, mode='test', layers=7):
        super(light_net, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        number_f = 32
        self.layers_high = 1
        self.layers_low = layers - self.layers_high

        input_channel = 3
        if mode == 'train':
            self.input = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
            # self.backbone = nn.ModuleList([nn.Conv2d(input_channel, number_f, 3, 1, 1, bias=True) for i in range(self.layers)])
            # self.backbone_1 = nn.ModuleList([nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True) for i in range(self.layers)])
            # self.backbone_2 = nn.ModuleList([nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True) for i in range(self.layers)])
            # # self.atten = nn.Sequential(*[nn.Conv2d(1, number_f, 3, 1, 1, bias=True) for i in range(layers)])
            # self.conv_a = nn.ModuleList([nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True) for i in range(self.layers)])
            # self.conv_b = nn.ModuleList([nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True) for i in range(self.layers)])
        # #########################################
        #     self.backbone = nn.ModuleList([UNet() for i in range(self.layers)])
        #     self.conv_a = nn.ModuleList([nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True) for i in range(self.layers)])
        #     self.conv_b = nn.ModuleList([nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True) for i in range(self.layers)])
        #     # print(self.backbone[0])
        #########################################
            self.backbone = nn.ModuleList([nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True) for i in range(self.layers_low)])
            self.conv_a = nn.ModuleList([nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True) for i in range(self.layers_low)])
            self.conv_b = nn.ModuleList([nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True) for i in range(self.layers_low)])

            self.backbone_1 = nn.ModuleList([nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True) for i in range(self.layers_high)])
            self.conv_a_1 = nn.ModuleList([nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True) for i in range(self.layers_high)])
            self.conv_b_1 = nn.ModuleList([nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True) for i in range(self.layers_high)])
        ###########################################
            # self.backbone = nn.ModuleList([nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True) for i in range(self.layers_low+self.layers_high)])
            # self.conv_a = nn.ModuleList([nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True) for i in range(self.layers_low+self.layers_high)]) 
            # self.conv_b = nn.ModuleList([nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True) for i in range(self.layers_low+self.layers_high)])

        elif mode == 'test':
            # self.backbone = nn.Conv2d(input_channel, number_f, 3, 1, 1, bias=True)
            # self.backbone_1 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
            # self.backbone_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
            # # self.atten = nn.Conv2d(1, number_f, 3, 1, 1, bias=True)
            # self.conv_a = nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True)
            # self.conv_b = nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True)
            ########################################
            self.input = nn.Conv2d(input_channel, number_f, 3, 1, 1, bias=True)
            self.backbone = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
            self.conv_a = nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True)
            self.conv_b = nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True)
            
            self.backbone_1 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
            self.conv_a_1 = nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True)
            self.conv_b_1 = nn.Conv2d(number_f, input_channel, 3, 1, 1, bias=True)
            
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                # print(m)
                # nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xo, mode='test'):
        if mode == 'train':
            # x_grey = torch.sum(1-xo, dim=1, keepdim=True)
            # x_attention = torch.sigmoid(self.e_conv_a(x_grey))
            # x_i = self.relu(self.input(xo))
            x_i = xo + 0.
            # grey =  torch.sum(xo, dim=1, keepdim=True)
            # x_i = grey/3
            # x_re = reflection(xo)
            amplititude = []
            beta = []
            layer_order_random = np.random.permutation(self.layers_low)
            x_i_1 = self.relu(self.input(xo - 1/2))# * atten
            # for i in range(self.layers):
            for i in layer_order_random:
                # print(i)
                # atten = self.sigmoid(self.atten[i](grey))
                # x_i_1 = self.relu(self.backbone[i](x_i - 1/2))# * atten
                # x_i_1 = self.relu(self.backbone_1[i](x_i_1))# * atten
                # x_i_1 = self.relu(self.backbone_2[i](x_i_1))# * atten
                # amp_i = self.tanh(self.conv_a[i](x_i_1)) * 1
                # bet_i = self.sigmoid(self.conv_b[i](x_i_1))*0.9 + 0.1
                # x_i = x_i + amp_i * 1 / (1 + torch.exp(-10 * (-x_i + bet_i - 0.1))) * x_i * (bet_i - x_i) * (1 / bet_i)
                ################################################################
                # x_i_1 = self.relu(self.backbone[i](x_i - 1/2))# * atten
                # amp_i = self.tanh(self.conv_a[i](x_i_1)) * 1
                # bet_i = self.sigmoid(self.conv_b[i](x_i_1))*0.9 + 0.1
                # x_i = x_i + amp_i * 1 / (1 + torch.exp(-10 * (-x_i + bet_i - 0.1))) * x_i * (bet_i - x_i) * (1 / bet_i)
                ################################################################
                
                x_i_1 = self.relu(self.backbone[i](x_i_1))# * atten
                amp_i = self.sigmoid(self.conv_a[i](x_i_1)) * 1
                bet_i = self.sigmoid(self.conv_b[i](x_i_1))*0.7 + 0.3
                x_i = x_i + amp_i * 1 / (1 + torch.exp(-15 * (-x_i + bet_i - 0.1))) * x_i * (bet_i - x_i) * (1 / bet_i)
                
                ################################################################
                
                # x_i_1 = self.relu(self.input[i](x_i - 1/2))# * atten
                # x_i_1 = self.relu(self.backbone[i](x_i_1))# * atten
                # amp_i = self.tanh(self.conv_a[i](x_i_1)) * 1
                # bet_i = self.sigmoid(self.conv_b[i](x_i_1))*0.4 + 0.6
                # x_i = x_i + amp_i * 1 / (1 + torch.exp(-10 * (-x_i + bet_i - 0.1))) * x_i * (bet_i - x_i) * (1 / bet_i)
                
                amplititude.append(amp_i)
                beta.append(bet_i)
                
            layer_order_random = np.random.permutation(self.layers_high) #+ self.layers_low
            for i in layer_order_random:
                x_i_1 = self.relu(self.backbone_1[i](x_i_1))# * atten
                amp_i = -self.sigmoid(self.conv_a_1[i](x_i_1)) * 1
                bet_i = self.sigmoid(self.conv_b_1[i](x_i_1))*0.2 + 0.1
                x_i = x_i + amp_i * 1 / (1 + torch.exp(15 * (-x_i + (1-bet_i) + 0.1))) * (1 - x_i) * (x_i - (1-bet_i)) * (1 / bet_i)
                
                # x_i_1 = self.relu(self.backbone[i](x_i_1))# * atten
                # amp_i = self.tanh(self.conv_a[i](x_i_1)) * 1
                # bet_i = self.sigmoid(self.conv_b[i](x_i_1))*0.7 + 0.3
                # x_i = x_i + amp_i * 1 / (1 + torch.exp(15 * (-x_i + (1-bet_i) + 0.1))) * (1 - x_i) * (x_i - (1-bet_i)) * (1 / bet_i)
                
                amplititude.append(amp_i)
                beta.append(bet_i)
                
            
            
            amplititude = torch.cat(amplititude, dim=1)
            beta = torch.cat(beta, dim=1)
            # xo = x_i * 3 * x_re
            xo = x_i
        elif mode == 'test':
            x_i = xo + 0.
            # grey =  torch.sum(xo, dim=1, keepdim=True)
            # x_i = grey/3
            # x_re = reflection(xo)
            x_i_1 = self.relu(self.input(xo - 1/2))
            amplititude = []
            beta = []
            for i in range(self.layers_low):
                # atten = self.sigmoid(self.atten(grey))
                # x_i_1 = self.relu(self.backbone(x_i - 3/2))# * atten
                
                # x_i_1 = self.relu(self.backbone(x_i - 1/2))# * atten
                # x_i_1 = self.relu(self.backbone_1(x_i_1))# * atten
                # x_i_1 = self.relu(self.backbone_2(x_i_1))
                # amp_i = self.tanh(self.conv_a(x_i_1)) * 1
                # bet_i = self.sigmoid(self.conv_b(x_i_1))*0.9 + 0.1
                # x_i = x_i + amp_i * 1 / (1 + torch.exp(-10 * (-x_i + bet_i - 0.1))) * x_i * (bet_i - x_i) * (1 / bet_i)
                ################################################################
                
                x_i_1 = self.relu(self.backbone(x_i_1))# * atten
                amp_i = self.sigmoid(self.conv_a(x_i_1)) * 1
                bet_i = self.sigmoid(self.conv_b(x_i_1))*0.7 + 0.3
                x_i = x_i + amp_i * 1 / (1 + torch.exp(-15 * (-x_i + bet_i - 0.1))) * x_i * (bet_i - x_i) * (1 / bet_i)
                
                ################################################################
                
                # x_i_1 = self.relu(self.input(x_i - 1/2))
                # x_i_1 = self.relu(self.backbone(x_i_1))# * atten
                # amp_i = self.tanh(self.conv_a(x_i_1)) * 1
                # bet_i = self.sigmoid(self.conv_b(x_i_1))*0.4 + 0.6
                # x_i = x_i + amp_i * 1 / (1 + torch.exp(-10 * (-x_i + bet_i - 0.1))) * x_i * (bet_i - x_i) * (1 / bet_i)
                
                
                amplititude.append(amp_i)
                beta.append(bet_i)
                
            for i in range(self.layers_high):
                x_i_1 = self.relu(self.backbone_1(x_i_1))# * atten
                amp_i = -self.sigmoid(self.conv_a_1(x_i_1)) * 1
                bet_i = self.sigmoid(self.conv_b_1(x_i_1))*0.2 + 0.1
                x_i = x_i + amp_i * 1 / (1 + torch.exp(15 * (-x_i + (1-bet_i) + 0.1))) * (1 - x_i) * (x_i - (1-bet_i)) * (1 / bet_i)
                # x_i = x_i
                # x_i_1 = self.relu(self.backbone(x_i_1))# * atten
                # amp_i = self.tanh(self.conv_a(x_i_1)) * 1
                # bet_i = self.sigmoid(self.conv_b(x_i_1))*0.7 + 0.3
                # x_i = x_i + amp_i * 1 / (1 + torch.exp(15 * (-x_i + (1-bet_i) + 0.1))) * (1 - x_i) * (x_i - (1-bet_i)) * (1 / bet_i)
                
                
                amplititude.append(amp_i)
                beta.append(bet_i)
                
                
            amplititude = torch.cat(amplititude, dim=1)
            beta = torch.cat(beta, dim=1)
            # xo = x_i * 3 * x_re
            xo = x_i
        return xo, amplititude, beta
    
