import os

# import pyiqa
import tensorflow as tf
import numpy as np
import torch
from PIL import Image
import glob
from math import sqrt
from pytorch_msssim import ssim
import lpips
def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1



if __name__ == '__main__':
    # enh_filePath = 'data/LSRW_SCI'
    enh_filePath = 'data/result/low_eval'
    # enh_filePath = 'data/result/low'
    # dce_filePath = 'data/result_dce/result_loleval0'#eval0
    # dce_filePath = 'data/lol_sci'  # eval0
    dce_filePath = 'data/result/low'
    # dce_filePath = 'data/LSRW_ru'
    org_filePath = 'data/high_eval'
    # org_filePath = 'data/high_LSRW'
    # enh_file_list = os.listdir(enh_filePath)
    # dce_file_list = os.listdir(dce_filePath)
    # org_file_list = os.listdir(org_filePath)
    test_list = glob.glob(org_filePath+ "/*")
    enh_list = glob.glob(enh_filePath + "/*")
    dce_list = glob.glob(dce_filePath + "/*")

    for i in range(len(test_list)):
        test_list[i] = test_list[i].replace("\\", "/")

    for i in range(len(enh_list)):
        enh_list[i] = enh_list[i].replace("\\", "/")

    for i in range(len(dce_list)):
        dce_list[i] = dce_list[i].replace("\\", "/")
    # iqa_metric = pyiqa.create_metric('niqe', device=torch.device('cuda'))
    yep = []
    ydp = []
    yes = []
    yds = []
    yems = []
    ydms = []
    yema = []
    ydma = []
    yeli = []
    ydli = []
    criterion = torch.nn.MSELoss(reduction='mean')
    # spatial = True
    # loss_fn = lpips.LPIPS(net='alex')  # best forward scores
    # loss_fn = lpips.LPIPS(net='vgg')  # best forward scores
    # loss_fn = lpips.LPIPS(net='squeeze')
    # loss_fn.cuda()
    for i in range(len(test_list)):
        # image = image
        print(test_list[i])


        test = Image.open(test_list[i])
        enh = (test_list[i]).replace('high_eval', 'result/low_eval')
        print(enh)
        # enh = (test_list[i]).replace('high_LSRW', 'result/low')
        enh = Image.open(enh)
        # dce = (test_list[i]).replace('high_eval', 'result/low_eval')
        # print(dce)
        # dce = (test_list[i]).replace('high_LSRW', 'result/low')
        # dce = Image.open(dce)
        test = (np.asarray(test))
        enh = (np.asarray(enh))
        # dce = (np.asarray(dce))

        yep.append(tf.image.psnr(test, enh, max_val=255))
        print('the psnr of ',test_list[i]  , ':', tf.image.psnr(test, enh, max_val=255))
        # ydp.append(tf.image.psnr(test, dce, max_val=255))
############################
        # test = torch.from_numpy(np.rollaxis(test, 2)).float().unsqueeze(0)
        # enh = torch.from_numpy(np.rollaxis(enh, 2)).float().unsqueeze(0)
        # dce = torch.from_numpy(np.rollaxis(dce, 2)).float().unsqueeze(0)
        # test = torch.FloatTensor(test).unsqueeze(0)
        # enh = torch.FloatTensor(enh).unsqueeze(0)
        # dce = torch.FloatTensor(dce).unsqueeze(0)

        # enh_ssim = ssim(test, enh, data_range=255, size_average=False)
        # yes.append(enh_ssim.numpy())
        # dce_ssim = ssim(test, dce, data_range=255, size_average=False)
        # yds.append(dce_ssim.numpy())
#################
        yes.append(tf.image.ssim(test, enh, 255))
        print('the ssim of ',test_list[i] , ':', tf.image.ssim(test, enh, 255))
        # yds.append(tf.image.ssim(test, dce, 255))

        # b,c,w = dce.shape
        # # mse = np.sum((test*1.0 - dce*1.0) ** 2) / (b*c*w)
        # mae = np.sum(np.absolute(test*1.0 - dce*1.0)) / (b*c*w)
        # # ydms.append(mse)
        # ydma.append(mae)
        # #
        # mse = np.sum((test * 1.0 - enh * 1.0) ** 2) / (b * c * w)
        # mae = np.sum(np.absolute(test * 1.0 - enh * 1.0)) / (b * c * w)
        # # yems.append(mse)
        # yema.append(mae)

        # test = lpips.im2tensor(lpips.load_image(test_list[i]))
        # enh = lpips.im2tensor(lpips.load_image(enh_list[i]))
        # dce = lpips.im2tensor(lpips.load_image(dce_list[i]))
        # # test = torch.from_numpy(np.rollaxis(test/255., 2)).float().unsqueeze(0)
        # # enh = torch.from_numpy(np.rollaxis(enh/255., 2)).float().unsqueeze(0)
        # # dce = torch.from_numpy(np.rollaxis(dce/255., 2)).float().unsqueeze(0)

        # yeli.append((loss_fn(test.cuda(), enh.cuda())).mean().item())
        # ydli.append((loss_fn(test.cuda(), dce.cuda())).mean().item())
        # # yeli.append((loss_fn(test, enh)).detach().numpy())
        # # ydli.append((loss_fn(test, dce)).detach().numpy())



    yem = np.mean(yep)
    print('the psnr of new-net:', yem)
    # ydm = np.mean(ydp)
    # print('the psnr of dce-net:', ydm)

    yem = np.mean(yes)
    print('the ssim of new-net:', yem)
    # ydm = np.mean(yds)
    # print('the ssim of dce-net:', ydm)

    # yem = np.mean(yems)
    # print('the MSE of new-net:', yem/1000)
    # ydm = np.mean(ydms)
    # print('the MSE of dce-net:', ydm/1000)

    # yem = np.mean(yema)
    # print('the MAE of new-net:', yem)
    # ydm = np.mean(ydma)
    # print('the MAE of dce-net:', ydm)

    # yem = np.mean(yeli)
    # print('the lpips of new-net:', yem)
    # ydm = np.mean(ydli)
    # print('the lpips of dce-net:', ydm)
