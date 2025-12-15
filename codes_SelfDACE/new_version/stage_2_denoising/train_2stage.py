import torch
import torch.nn as nn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
# from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim


# def weights_init(m):
# 	classname = m.__class__.__name__
# 	if classname.find('Conv') != -1:
# 		m.weight.data.normal_(0.0, 0.02)
# 	elif classname.find('BatchNorm') != -1:
# 		m.weight.data.normal_(1.0, 0.02)
# 		m.bias.data.fill_(0)
def reflection(im):
    # mr, mg, mb = torch.split(im, 1, dim=1)
    # r = mr / (mr + mg + mb + 0.0001)
    # g = mg / (mr + mg + mb + 0.0001)
    # b = mb / (mr + mg + mb + 0.0001)
    img_clip = torch.clamp(im, min=1e-10)
    img_clip_re = img_clip / torch.sum(img_clip, dim=1, keepdim=True)
    return img_clip_re


def illuminance(s):
    return ((s[:, 0, :, :] + s[:, 1, :, :] + s[:, 2, :, :])).unsqueeze(1)


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    denoise_net = model.denoise_net().cuda()
    denoise_net.load_state_dict(torch.load(config.pretrain_dir)['net'])
    # denoise_net.requires_grad_(requires_grad=False)
    
    light_net = model.light_net().cuda()
    light_net.load_state_dict(torch.load(config.stage1_dir))
    
    # light_net.requires_grad_(requires_grad=False)
    optimizer_light = torch.optim.Adam(light_net.parameters(), lr=0.000001, weight_decay=config.weight_decay)
    optimizer_denoise = torch.optim.Adam(denoise_net.parameters(), lr=0.00001, weight_decay=config.weight_decay)
    
    # if config.load_pretrain == True:
    #     denoise_net.load_state_dict(torch.load(config.pretrain_dir)['net'])
        # optimizer_2.load_state_dict(torch.load(config.pretrain_dir)['optimizer'])
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    criterion = nn.MSELoss(reduction='mean')
    L_gcolor = Myloss.L_color()
    L_exp = Myloss.L_exp(4, 0.8)
    L_smo = Myloss.L_SMO()
    L_locl = Myloss.L_localcolor(win=4)
    L_gra = Myloss.L_gradient()
    L_l1 = nn.L1Loss(reduction='mean')


    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.99)


    loss_idx_value = 0
    for epoch in range(config.num_epochs):
        for iteration, dataset in enumerate(train_loader):
            img_lowlight = dataset
            # img_lowlight_distort = dataset['data_lowlight_distort']
            
            img_lowlight = img_lowlight.cuda()
            # img_lowlight_distort = img_lowlight_distort.cuda()
            
            n,c,h,w = img_lowlight.size()
            sigma = 0.009 * np.random.rand(1) + 0.001
            

            # sigma = 0.01
            gaussian_noise = np.zeros((3, h, w), dtype=np.float32)
            noise_r = np.random.normal(0.0, sigma, (1, h, w)).astype(np.float32)
            noise_g = np.random.normal(0.0, sigma/2, (1, h, w)).astype(np.float32)
            noise_b = np.random.normal(0.0, sigma, (1, h, w)).astype(np.float32)
            gaussian_noise += np.concatenate((noise_r, noise_g, noise_b), axis=0)
            gaussian_noise = torch.from_numpy(gaussian_noise)
            gaussian_noise = gaussian_noise.repeat([n, 1, 1, 1])
            # img_re = reflection(img_lowlight)
            # img_lu = illuminance(img_lowlight) 
            img_lowlight_noise = (img_lowlight + (1 - img_lowlight) * gaussian_noise.cuda()).clip(max=1, min=1e-10)
            
            # img_lowlight_noise_re = reflection(img_lowlight_noise)
            
            # img_lowlight_re = reflection(img_lowlight)
            # img_re_no = reflection(img_lowlight_noise)
            # img_lu_no = illuminance(img_lowlight_noise)
            
            if loss_idx_value % 500 >= 0:
                light_net.train()
                light_net.requires_grad_(requires_grad=True)
                denoise_net.eval()
                denoise_net.requires_grad_(requires_grad=False)
                
                img_normlight_noise, xr, xr1 = light_net(img_lowlight_noise)
                img_normlight_deno = denoise_net(img_normlight_noise)
                
                loss_smoo = 1000 * L_smo(xr) + 5000 * L_smo(xr1)  # 1000
                loss_expo = 5 * torch.mean(L_exp(img_normlight_deno, img_lowlight))  # 5
                loss_locl = 1000 * torch.mean(L_locl(img_normlight_deno, img_lowlight))  # 800
                loss_gcol = 1500 * torch.mean(L_gcolor(img_normlight_deno))  # 20
                loss_stage_1 = loss_expo + loss_smoo + loss_locl + loss_gcol
                
                loss = loss_stage_1
                
                optimizer_light.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(light_net.parameters(), config.grad_clip_norm)
                optimizer_light.step()
                
                loss_idx_value += 1
                
                if ((loss_idx_value) % config.display_iter) == 0:
                    print("Loss at iteration", loss_idx_value, ":", loss.item())
                if ((loss_idx_value) % config.snapshot_iter) == 0:
                # torch.save({'net': denoise_net.state_dict(), 'optimizer': optimizer_2.state_dict()},
                #            config.snapshots_folder + "deno" + "Epoch" + str(epoch) + '.pth')
                    torch.save({'net': light_net.state_dict(), 'optimizer': optimizer_light.state_dict()},
                            config.snapshots_folder + "lit" + "Epoch" + str(epoch) + '.pth')
            else:
                light_net.eval()
                light_net.requires_grad_(requires_grad=False)
                denoise_net.train()
                denoise_net.requires_grad_(requires_grad=True)
                
                img_normlight_noise, xr, xr1 = light_net(img_lowlight_noise)
                img_normlight_deno = denoise_net(img_normlight_noise)
                
                xo = img_lowlight
                for i in np.arange(7):
                    # xo = xo + xr[:, 3 * i:3 * i + 3, :, :] * torch.maximum(xo * (xr1[:, 3 * i:3 * i + 3, :, :] - xo) * (1 / xr1[:, 3 * i:3 * i + 3, :, :]),0*xo)
                    xo = xo + xr[:, 3 * i:3 * i + 3, :, :] * 1 / (
                            1 + torch.exp(-10 * (-xo + xr1[:, 3 * i:3 * i + 3, :, :] - 0.1))) * xo * (
                                xr1[:, 3 * i:3 * i + 3, :, :] - xo) * (1 / xr1[:, 3 * i:3 * i + 3, :, :])
                img_normlight = xo
                
                loss_deno =  10 * torch.mean(L_l1(img_normlight_deno, img_normlight))+\
                                10 * torch.mean(1 - ssim(img_normlight_deno, img_normlight))+\
                                40 * torch.mean(L_gra(img_normlight_deno, img_normlight))
                loss_stage_2 = loss_deno
                loss = loss_stage_2

                optimizer_denoise.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(denoise_net.parameters(), config.grad_clip_norm)
                optimizer_denoise.step()

                loss_idx_value += 1
                
                if ((loss_idx_value) % config.display_iter) == 0:
                    print("Loss at iteration", loss_idx_value, ":", loss.item())
                if ((loss_idx_value) % config.snapshot_iter) == 0:
                    torch.save({'net': denoise_net.state_dict(), 'optimizer': optimizer_denoise.state_dict()},
                            config.snapshots_folder + "deno" + "Epoch" + str(epoch) + '.pth')

            # img_normlight_noise, xr, xr1 = light_net(img_lowlight_noise)

            # xo = img_lowlight
            # for i in np.arange(7):
            #     # xo = xo + xr[:, 3 * i:3 * i + 3, :, :] * torch.maximum(xo * (xr1[:, 3 * i:3 * i + 3, :, :] - xo) * (1 / xr1[:, 3 * i:3 * i + 3, :, :]),0*xo)
            #     xo = xo + xr[:, 3 * i:3 * i + 3, :, :] * 1 / (
            #             1 + torch.exp(-10 * (-xo + xr1[:, 3 * i:3 * i + 3, :, :] - 0.1))) * xo * (
            #                  xr1[:, 3 * i:3 * i + 3, :, :] - xo) * (1 / xr1[:, 3 * i:3 * i + 3, :, :])
            # img_normlight = xo

            # img_normlight_deno = denoise_net(img_normlight_noise)
            # xe = img_lowlight.clone()
            # for i in np.arange(6):
            #     xe = xe + xc[:, 3 * i:3 * i + 3, :, :] * 1 / (
            #             1 + torch.exp(-10 * (-xe + xc1[:, 3 * i:3 * i + 3, :, :] - 0.1))) * xe * (
            #                  xc1[:, 3 * i:3 * i + 3, :, :] - xe) * (1 / xc1[:, 3 * i:3 * i + 3, :, :])

            # loss_deno = 10 * torch.mean(1 - ssim(img_final, img_lowlight))+\
            #             40 * torch.mean(L_gra(img_final, img_lowlight))
            # loss_smo = 1 * torch.mean(L_smo(img_final))
            # loss_deno = 50000 * torch.mean(criterion(enhanced_image, xe))
            # loss_deno = 100 * torch.mean((1 - ssim(xd, img_lowlight)))
            # loss_smoo = 1000 * L_smo(xr) + 5000 * L_smo(xr1)  # 1000
            # loss_expo = 5 * torch.mean(L_exp(img_normlight_deno, img_lowlight))  # 5
            # loss_locl = 1000 * torch.mean(L_locl(img_normlight_deno, img_lowlight))  # 800
            # loss_gcol = 1500 * torch.mean(L_gcolor(img_normlight_deno))  # 20
            # loss_stage_1 = loss_expo + loss_smoo + loss_locl + loss_gcol
            
            # loss_deno =  10 * torch.mean(L_l1(img_normlight_deno, img_normlight))
            # loss_stage_2 = loss_deno
            
            # loss = loss_stage_1# + loss_stage_2
            

            # loss_lu = criterion(img_lu_de, img_lu)
            # loss_re = criterion(img_re_de, img_re)
            # loss = 1*loss_lu + 5*loss_re



            # optimizer_light.zero_grad()
            # optimizer_2.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm(light_net.parameters(), config.grad_clip_norm)
            # torch.nn.utils.clip_grad_norm(denoise_net.parameters(), config.grad_clip_norm)
            # optimizer_light.step()
            # optimizer_2.step()

            # if ((iteration + 1) % config.display_iter) == 0:
            #     print("Loss at iteration", iteration + 1, ":", loss.item())
            # if ((iteration + 1) % config.snapshot_iter) == 0:
            #     # torch.save({'net': denoise_net.state_dict(), 'optimizer': optimizer_2.state_dict()},
            #     #            config.snapshots_folder + "deno" + "Epoch" + str(epoch) + '.pth')
            #     torch.save({'net': light_net.state_dict(), 'optimizer': optimizer_1.state_dict()},
            #                config.snapshots_folder + "lit" + "Epoch" + str(epoch) + '.pth')
            # scheduler.step()


if __name__ == "__main__":
    start1 = time.perf_counter()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="../train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)  # 0.00001

    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)

    parser.add_argument('--num_epochs', type=int, default=1000)  # initial value 200
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=100)
    parser.add_argument('--snapshot_iter', type=int, default=50)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_denoise/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots_denoise/pre_train_2.pth")
    parser.add_argument('--stage1_dir', type=str, default="snapshots_light/pre_train_1.pth")
    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
    end1 = time.perf_counter()
    print("final is in : %s Seconds " % (end1 - start1))








