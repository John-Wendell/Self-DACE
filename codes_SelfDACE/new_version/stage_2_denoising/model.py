import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def reflection(im):
    mr, mg, mb = torch.split(im, 1, dim=1)
    r = mr / (mr + mg + mb + 0.0001)
    g = mg / (mr + mg + mb + 0.0001)
    b = mb / (mr + mg + mb + 0.0001)
    return torch.cat([r, g, b], dim=1)


def luminance(s):
    return ((s[:, 0, :, :] + s[:, 1, :, :] + s[:, 2, :, :])).unsqueeze(1)


class denoise_net(nn.Module):

    def __init__(self):
        super(denoise_net, self).__init__()
        num_layers = 19
        num_features = 64
        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        #layers.append(nn.ReLU(inplace=True))
        #layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=0))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        return y + residual


class light_net(nn.Module):

    def __init__(self):
        super(light_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        number_f = 32

        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv1_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv1_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv2_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv2_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv3_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv4_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv5 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv5_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv6 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv6_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv6_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv7 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv7_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv7_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

    # self.e_conv0r = nn.Conv2d(number_f * 2, 18, 3, 1, 1, bias=True)
    #
    # self.e_conv14 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
    # self.e_conv15 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
    # self.e_conv16 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
    # self.e_conv1r = nn.Conv2d(number_f * 2, 18, 3, 1, 1, bias=True)
    # self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
    # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
    # self.bn = nn.BatchNorm2d(number_f)

    def forward(self, xo):
        x1 = self.relu(self.e_conv1(xo))
        x1_a = self.tanh(self.e_conv1_a(x1))
        x1_b = self.sigmoid(self.e_conv1_a(x1)) * 0.5 + 0.5

        x2 = self.relu(self.e_conv2(x1))
        x2_a = self.tanh(self.e_conv2_a(x2))
        x2_b = self.sigmoid(self.e_conv2_b(x2)) * 0.5 + 0.5

        x3 = self.relu(self.e_conv3(x2))
        x3_a = self.tanh(self.e_conv3_a(x3))
        x3_b = self.sigmoid(self.e_conv3_b(x3)) * 0.5 + 0.5

        x4 = self.relu(self.e_conv4(x3))
        x4_a = self.tanh(self.e_conv4_a(x4))
        x4_b = self.sigmoid(self.e_conv4_b(x4)) * 0.5 + 0.5

        x5 = self.relu(self.e_conv5(x4))
        x5_a = self.tanh(self.e_conv5_a(x5))
        x5_b = self.sigmoid(self.e_conv5_b(x5)) * 0.5 + 0.5

        x6 = self.relu(self.e_conv6(x5))
        x6_a = self.tanh(self.e_conv6_a(x6))
        x6_b = self.sigmoid(self.e_conv6_b(x6)) * 0.5 + 0.5

        x7 = self.relu(self.e_conv7(x6))
        x7_a = self.tanh(self.e_conv7_a(x7))
        x7_b = self.sigmoid(self.e_conv7_b(x7)) * 0.5 + 0.5

        xr = torch.cat([x1_a, x2_a, x3_a, x4_a, x5_a, x6_a, x7_a], dim=1)  # , x6_a, x7_a
        xr1 = torch.cat([x1_b, x2_b, x3_b, x4_b, x5_b, x6_b, x7_b], dim=1)  # , x6_b, x7_b

        # x14 = self.relu(self.e_conv14(x3))
        # x15 = self.relu(self.e_conv15(torch.cat([x14, x3], 1)))
        # x16 = self.relu(self.e_conv16(torch.cat([x15, x2], 1)))
        # xr1 = torch.sigmoid(self.e_conv1r(torch.cat([x16, x1], 1)))*0.5+0.5

        for i in np.arange(7):
            # xo = xo + xr[:, 3 * i:3 * i + 3, :, :] * torch.maximum(xo * (xr1[:, 3 * i:3 * i + 3, :, :] - xo) * (1 / xr1[:, 3 * i:3 * i + 3, :, :]),0*xo)
            xo = xo + xr[:, 3 * i:3 * i + 3, :, :] * 1 / (
                        1 + torch.exp(-10 * (-xo + xr1[:, 3 * i:3 * i + 3, :, :] - 0.1))) * xo * (
                             xr1[:, 3 * i:3 * i + 3, :, :] - xo) * (1 / xr1[:, 3 * i:3 * i + 3, :, :])
        return xo, xr, xr1
