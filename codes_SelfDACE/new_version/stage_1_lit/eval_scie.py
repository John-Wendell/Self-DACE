import os
import tensorflow as tf
import numpy as np
import torch
from PIL import Image
import glob
from math import sqrt
import lpips
from torchvision.transforms import Resize

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


class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)
    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

if __name__ == '__main__':
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()

    enh_filePath = './data/resize_scie'
    # enh_filePath = 'data/result_LOL/low'
    # enh_filePath = 'data/result_scie'
    # enh_filePath = 'data/result_dce/result_loleval0'#eval0
    # dce_filePath = 'data/result_LOL0'
    # dce_filePath = 'data/LSRW_dce/low'
    # org_filePath = 'data/high_eval'
    # org_filePath = 'data/high_lol'
    org_filePath = '../data/high_scie/Label'
    # enh_file_list = os.listdir(enh_filePath)
    # dce_file_list = os.listdir(dce_filePath)
    # org_file_list = os.listdir(org_filePath)
    test_list = glob.glob(org_filePath+ "/*")

    # for i in range(len(test_list)):
    #     test_list[i] = test_list[i].replace("\\", "/")

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
    for i in range(125):#len(test_list)

        enh_list = glob.glob(enh_filePath + "/" + str(1+i) + "/*")
        # for j in range(len(enh_list)):
        #     enh_list[j] = enh_list[j].replace("\\", "/")
        for k in range(2):#len(enh_list)
            if i<=124:
                test_path = glob.glob(org_filePath + "/" + str(1+i)+'.JPG')
            # else:
            #     test_path = glob.glob(org_filePath + "/" + str(1 + i) + '.PNG')

            test = Image.open(test_path[0])
            print(test_path[0])
            
            test = test.resize((512,512),Image.LANCZOS)
            enh_path = glob.glob(enh_filePath + "/" + str(1+i)+ "/" + str(1+k)+'.JPG')
            # print(enh_path)
            enh = Image.open(enh_path[0])
            print(enh_path[0])

            # dce = Image.open(dce_list[k])
            test = (np.asarray(test))
            enh = (np.asarray(enh))
            # dce = (np.asarray(dce))
            #
            #
            yep.append(tf.image.psnr(test, enh, max_val=255))
            # ydp.append(tf.image.psnr(test, dce, max_val=255))
            #
            yes.append(tf.image.ssim(test, enh, 255))
            # print('the ssim of ', enh_list[k],'and',test_path[0], ':', tf.image.ssim(test, enh, 255))
            # yds.append(tf.image.ssim(test, dce, 255))
            #
            # b, c, w = enh.shape
            # # # mse = np.sum((test * 1.0 - dce * 1.0) ** 2) / (b * c * w)
            # # mae = np.sum(np.absolute(test * 1.0 - enh * 1.0)) / (b * c * w)
            # # # ydms.append(mse)
            # # ydma.append(mae)
            #
            # # mse = np.sum((test * 1.0 - enh * 1.0) ** 2) / (b * c * w)
            # mae = np.sum(np.absolute(test * 1.0 - enh * 1.0)) / (b * c * w)
            # # yems.append(mse)
            # yema.append(mae)
            #

            # enh = lpips.im2tensor(lpips.load_image(enh_list[k]))
            # test = lpips.im2tensor(lpips.load_image(test_path[0]))
            # torch_resize = Resize([512,512])  # 定义Resize类对象
            # test = torch_resize(test)


            # yeli.append((loss_fn(test.cuda(), enh.cuda())).mean().item())







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
    #
    # yem = np.mean(yema)
    # print('the MAE of new-net:', yem)
    # ydm = np.mean(ydma)
    # print('the MAE of dce-net:', ydm)

    yem = np.mean(yeli)
    print('the lpips of new-net:', yem)
    # ydm = np.mean(ydli)
    # print('the lpips of dce-net:', ydm)
