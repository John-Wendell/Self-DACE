import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np




def reflection(im):
    mr, mg, mb = torch.split(im, 1, dim=1)
    r = mr / (mr + mg + mb + 0.0001)
    g = mg / (mr + mg + mb + 0.0001)
    b = mb / (mr + mg + mb + 0.0001)
    return torch.cat([r, g, b], dim=1)


def luminance(s):
    return ((s[:, 0, :, :] + s[:, 1, :, :] + s[:, 2, :, :])).unsqueeze(1)

def save_network(light_net_e, light_net, optimizer, config, epoch):
        decay = 1.
        layers_number_1 = 1
        layers_number = config.layers - layers_number_1
        
        
        light_net_e.input.weight.data = decay*light_net.input.weight.data + (1-decay)*light_net_e.input.weight.data
        light_net_e.input.bias.data = decay*light_net.input.bias.data + (1-decay)*light_net_e.input.bias.data
        
        # copy_weight = 1/layers_number*light_net.input[0].weight.data
        # copy_bias = 1/layers_number*light_net.input[0].bias.data
        # for i in range(layers_number-1):
        #     copy_weight += 1/layers_number*light_net.input[i+1].weight.data
        #     copy_bias += 1/layers_number*light_net.input[i+1].bias.data
        # light_net_e.input.weight.data = decay*copy_weight + (1-decay)*light_net_e.input.weight.data
        # light_net_e.input.bias.data = decay*copy_bias + (1-decay)*light_net_e.input.bias.data
        
        copy_weight = 1/layers_number*light_net.backbone[0].weight.data
        copy_bias = 1/layers_number*light_net.backbone[0].bias.data
        for i in range(layers_number-1):
            copy_weight += 1/layers_number*light_net.backbone[i+1].weight.data
            copy_bias += 1/layers_number*light_net.backbone[i+1].bias.data
        light_net_e.backbone.weight.data = decay*copy_weight + (1-decay)*light_net_e.backbone.weight.data
        light_net_e.backbone.bias.data = decay*copy_bias + (1-decay)*light_net_e.backbone.bias.data
        
        copy_weight = 1/layers_number_1*light_net.backbone_1[0].weight.data
        copy_bias = 1/layers_number_1*light_net.backbone_1[0].bias.data
        for i in range(layers_number_1-1):
            copy_weight += 1/layers_number_1*light_net.backbone_1[i+1].weight.data
            copy_bias += 1/layers_number_1*light_net.backbone_1[i+1].bias.data
        light_net_e.backbone_1.weight.data = decay*copy_weight + (1-decay)*light_net_e.backbone_1.weight.data
        light_net_e.backbone_1.bias.data = decay*copy_bias + (1-decay)*light_net_e.backbone_1.bias.data
        
        # copy_weight = 1/layers_number*light_net.backbone_2[0].weight.data
        # copy_bias = 1/layers_number*light_net.backbone_2[0].bias.data
        # for i in range(layers_number-1):
        #     copy_weight += 1/layers_number*light_net.backbone_2[i+1].weight.data
        #     copy_bias += 1/layers_number*light_net.backbone_2[i+1].bias.data
        # light_net_e.backbone_2.weight.data = decay*copy_weight + (1-decay)*light_net_e.backbone_2.weight.data
        # light_net_e.backbone_2.bias.data = decay*copy_bias + (1-decay)*light_net_e.backbone_2.bias.data
        
        copy_weight = 1/layers_number*light_net.conv_a[0].weight.data
        copy_bias = 1/layers_number*light_net.conv_a[0].bias.data
        for i in range(layers_number-1):
            copy_weight += 1/layers_number*light_net.conv_a[i+1].weight.data
            copy_bias += 1/layers_number*light_net.conv_a[i+1].bias.data
        light_net_e.conv_a.weight.data = decay*copy_weight + (1-decay)*light_net_e.conv_a.weight.data
        light_net_e.conv_a.bias.data = decay*copy_bias + (1-decay)*light_net_e.conv_a.bias.data
        
        copy_weight = 1/layers_number*light_net.conv_b[0].weight.data
        copy_bias = 1/layers_number*light_net.conv_b[0].bias.data
        for i in range(layers_number-1):
            copy_weight += 1/layers_number*light_net.conv_b[i+1].weight.data
            copy_bias += 1/layers_number*light_net.conv_b[i+1].bias.data
        light_net_e.conv_b.weight.data = decay*copy_weight + (1-decay)*light_net_e.conv_b.weight.data
        light_net_e.conv_b.bias.data = decay*copy_bias + (1-decay)*light_net_e.conv_b.bias.data
        
        
        copy_weight = 1/layers_number_1*light_net.conv_a_1[0].weight.data
        copy_bias = 1/layers_number_1*light_net.conv_a_1[0].bias.data
        for i in range(layers_number_1-1):
            copy_weight += 1/layers_number_1*light_net.conv_a_1[i+1].weight.data
            copy_bias += 1/layers_number_1*light_net.conv_a_1[i+1].bias.data
        light_net_e.conv_a_1.weight.data = decay*copy_weight + (1-decay)*light_net_e.conv_a_1.weight.data
        light_net_e.conv_a_1.bias.data = decay*copy_bias + (1-decay)*light_net_e.conv_a_1.bias.data
        
        copy_weight = 1/layers_number_1*light_net.conv_b_1[0].weight.data
        copy_bias = 1/layers_number_1*light_net.conv_b_1[0].bias.data
        for i in range(layers_number_1-1):
            copy_weight += 1/layers_number_1*light_net.conv_b_1[i+1].weight.data
            copy_bias += 1/layers_number_1*light_net.conv_b_1[i+1].bias.data
        light_net_e.conv_b_1.weight.data = decay*copy_weight + (1-decay)*light_net_e.conv_b_1.weight.data
        light_net_e.conv_b_1.bias.data = decay*copy_bias + (1-decay)*light_net_e.conv_b_1.bias.data
        
        # copy_weight = 1/7*light_net.atten[0].weight.data
        # copy_bias = 1/7*light_net.atten[0].bias.data
        # for i in range(6):
        #     copy_weight += 1/7*light_net.atten[i+1].weight.data
        #     copy_bias += 1/7*light_net.atten[i+1].bias.data
        # light_net_e.atten.weight.data = decay*copy_weight + (1-decay)*light_net_e.atten.weight.data
        # light_net_e.atten.bias.data = decay*copy_bias + (1-decay)*light_net_e.atten.bias.data
        
        torch.save({'net_decay': light_net_e.state_dict(),
                    'net_intact': light_net.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    # mask_net = model.mask_net().cuda()
    # mask_net.load_state_dict(torch.load(config.mask_net_dir)['net'])
    # mask_net.requires_grad_(requires_grad=False)
    # mask_net.eval()

    light_net_train = model.light_net(mode='train', layers=config.layers).cuda()
    light_net_test = model.light_net(mode='test', layers=config.layers).cuda()
    optimizer = torch.optim.Adam(light_net_train.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.load_pretrain == True:
        light_net_train.load_state_dict(torch.load(config.pretrain_dir)['net'])
        # light_net.load_state_dict(torch.load(config.pretrain_dir))
        # optimizer.load_state_dict(torch.load(config.pretrain_dir)['optimizer'])
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    L_gcolor = Myloss.L_color()
    L_exp = Myloss.L_exp(4, 0.8)
    L_smo = Myloss.L_SMO()
    L_locl = Myloss.L_localcolor()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=1)

    light_net_train.train()
    loss_idx_value = 0
    for epoch in range(config.num_epochs):
        print("Epoch:", epoch)
        for iteration, dataset in enumerate(train_loader):
            img_lowlight = dataset['data_lowlight']
            
            # img_lowlight_masked = dataset['data_lowlight_masked']
            # img_lowlight_distort = dataset['data_lowlight_distorted']
            # img_lowlight_distort = (img_lowlight * img_lowlight + (1 - img_lowlight) * img_lowlight_distort).clip(max=1, min=1e-10)
            
            img_lowlight = img_lowlight.cuda()
            # img_lowlight_masked = img_lowlight_masked.cuda()
            # img_lowlight_distort = img_lowlight_distort.cuda()
            # n, c, h, w = img_lowlight.size()
            # sigma = 0.005 * np.random.rand(1)
            # # sigma = 0.01
            # gaussian_noise = np.zeros((3, h, w), dtype=np.float32)
            # noise_r = np.random.normal(0.0, sigma, (1, h, w)).astype(np.float32)
            # noise_g = np.random.normal(0.0, sigma/2, (1, h, w)).astype(np.float32)
            # noise_b = np.random.normal(0.0, sigma, (1, h, w)).astype(np.float32)
            # gaussian_noise += np.concatenate((noise_r, noise_g, noise_b), axis=0)
            # gaussian_noise = torch.from_numpy(gaussian_noise)
            # gaussian_noise = gaussian_noise.repeat([n, 1, 1, 1])
            #
            # # img_re = reflection(img_lowlight)
            # img_lu = luminance(img_lowlight)
            # #
            # img_lowlight_noise = (img_lowlight + (1 - img_lu / 3) * gaussian_noise.cuda()).clip(max=1, min=0)
            #
            # img_re_no = reflection(img_lowlight_noise)
            # img_lu_no = luminance(img_lowlight_noise)
            # masked_features = mask_net(img_lowlight_masked)
            enhanced_image,  rr1, rr2  = light_net_train(img_lowlight, mode='train')

            loss_smo = 5000 * L_smo(rr1) + 5000 * L_smo(rr2)  # 1000
            loss_exp = 5 * torch.mean(L_exp(enhanced_image, img_lowlight))  # 5
            loss_locl = 20000 * torch.mean(L_locl(enhanced_image, img_lowlight))  # 800
            loss_gcol = 50 * torch.mean(L_gcolor(enhanced_image))  # 20
            # loss_regu = 1 * torch.mean(torch.pow(rr1, 2)) +\
            #             1 * torch.mean(torch.pow(rr2-0.4, 2))  # 1
            # loss_regu = 1 * torch.mean(torch.pow(rr2-0.1, 2))  # 1

            loss = loss_exp + loss_smo + loss_gcol + loss_locl #+ loss_regu#+ loss_col# + Loss_TV#+ loss_spa# + loss_smooth #+ loss_noise + loss_exp+ loss_col+ loss_spa  Loss_TV +  loss_locl
            loss_idx_value += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(light_net_train.parameters(),config.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % config.snapshot_iter) == 0:
                # torch.save({'net_intact': light_net_train.state_dict(), 'optimizer': optimizer.state_dict()},
                #            config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
                save_network(light_net_test, light_net_train, optimizer, config, epoch)
            




if __name__ == "__main__":
    start1 = time.perf_counter()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="../train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)#0.00001

    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)

    parser.add_argument('--num_epochs', type=int, default=1000)  # initial value 200
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=100)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_light/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default= "snapshots_light/Epoch90.pth")
    parser.add_argument('--mask_net_dir', type=str, default= "snapshots_mask/Epoch499.pth")
    
    parser.add_argument('--layers', type=int, default=12)

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)


    train(config)
    end1 = time.perf_counter()
    print("final is in : %s Seconds " % (end1 - start1))








