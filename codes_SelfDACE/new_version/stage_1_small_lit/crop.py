########### this python file is used to crop the images from a folder and save them in another folder

import os
import sys
import cv2
import numpy as np

def crop_images(input_folder, output_folder):

    files = os.listdir(input_folder)
    print(files)
        # print(file)
        # img = cv2.imread(os.path.join(file))
        # h, w, _ = img.shape
        # crop_img = img[0:int(1/4*h), int(3/4*w):int(4/4*w)]
        # h_1, w_1, _ = crop_img.shape
        # img[-h_1:, -w_1:] = crop_img
        # cv2.imwrite(os.path.join(output_folder, file), img)

if __name__ == '__main__':
    input_folder = '.\org'
    output_folder = '.\cropped'
    print('Cropping images from', input_folder, 'to', output_folder)
    crop_images(input_folder, output_folder)