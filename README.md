# Self-DACE LLIE Method
Official pytorch version for Self-Reference Deep Adaptive Curve Estimation for Low-Light Image Enhancement

- Paper is avalible at:
- Old version: [arXiv version - Self-Reference Deep Adaptive Curve Estimation for Low-Light Image Enhancement](https://arxiv.org/pdf/2308.08197.pdf)
- New version: code is uploaded.

# Hope AI would illuminate our unknown and invisible path to the future as it illuminates low-light images!

# Demo
## Demo on Low-light Image Enhancement
![demo_1](demo4git/demo1.png) | ![demo_2](demo4git/demo2.png) 
---|---
![demo_3](demo4git/demo3.png) | ![demo_4](demo4git/demo4.png) 

Visual comparison with original low-light image
on [LOL](https://daooshee.github.io/BMVC2018website/) and [SCIE](https://github.com/csjcai/SICE) dataset. The enhanced images of our
method are on the top-right corners, and the input low-light
images are on the bottom-left corners.

## Demo on the Improvement of Low-light Face detection (New Version)
Demostration of improvement for Dark Face Detection task ([CVPR UG2+ Challenge 2021](http://cvpr2022.ug2challenge.org/program21/track1.html)) on [DarkFace Dataset](https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset) using [Retinaface](https://github.com/deepinsight/insightface).
![demo_1](demo4git/face_1.png)
![demo_2](demo4git/face_2.png)
![demo_2](demo4git/face_3.png)
The number on the top of box is the confidence score given by Retinaface with a confidence threshold of 0.5.

![demo_1](demo4git/0.25.png) | ![demo_2](demo4git/0.50.png) | ![demo_2](demo4git/0.75.png)
---|---|---
IoU=0.25|IoU=0.50|IoU=0.75

Test on the first 200 images.

## Demo on the Improvement of Low-light Image Interactive Segmentation (Old Version)
![demo_2_1](visualization/vis1.jpg)
<!--[demo_2_2](visualization/vis2.jpg)-->
Demostration of improvement for segmentation task on [DarkFace Dataset](https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset) using [PiClick](https://github.com/cilinyan/PiClick).
The green stars are the objects of interactive segmentation what we want to segment.
GT is annotated on the enhanced images manually by us.

# New version Framework
![frame](demo4git/framework.png) 

# Quantitative Comparison
Old Version:
![metrics](demo4git/com1.png) 
Ours* is the result only from Stage-I.

New Version:
![metrics](demo4git/metrics.png) 

### Note:
- Our metric values are in line with [Low-Light Image and Video Enhancement Using Deep Learning: A Survey](https://github.com/Li-Chongyi/Lighting-the-Darkness-in-the-Deep-Learning-Era-Open).


## Visual Comparison on LIME (Old version)
<div align=center>
<img src="demo4git/visual.png">
</div>

## Visual Comparison on SCIE (New version)
<div align=center>
<img src="demo4git/com_scie.png">
</div>

# How to use it
## Prerequisite
```
cd ./codes_SelfDACE
pip install -r ./requirements.txt
```

## Test Stage-I (only enhancing luminance)
```
cd ./stage1
python test_1stage.py
```
Test dates should be placed in `codes_SelfDACE/stage1/data/test_data/low_eval`,
And then results would be found in `codes_SelfDACE/stage1/data/result/low_eval`.

## Test both Stage-I and Stage-II (enhancing luminance and denoising)
```
cd ./stage2
python test_1stage.py
```
Test dates should be placed in `codes_SelfDACE/stage2/data/test_data/low_eval`,
And then results would be found in `codes_SelfDACE/stage2/data/result/low_eval`.

# How to train it
## Prerequisite
```
cd ./codes_SelfDACE
pip install -r ./requirements.txt
```

## Train Stage-I (only enhancing luminance)
1.
      You should download the training dataset from [SCIE_part1](https://github.com/csjcai/SICE) and resize all images to 256x256.
      Or you could download it directly from [SCIE_part1_ZeroDCE_version](https://github.com/Developer-Zer0/ZeroDCE), of which iamges have been cropped to 512x512 already. If you want to use it in your work, please cite [SCIE_part1](https://github.com/csjcai/SICE).

2.
      ```
      cd ./stage1
      python train_1stage.py
      ```

## Train Stage-II (only denoising)

1.
      Copy the `pre-trained model` and `training dataset` from stage1, and put `pre-trained model` of Stage-I in `./stage2/snapshots_light`

2.
      ```
      cd ./stage2
      python train_2stage.py
      ```
# Acknowledgment
This paper gets a big inspiration from [ZeroDCE](https://github.com/Li-Chongyi/Zero-DCE).

# Citation
If you find our work useful for your research, please cite our paper
```
@article{wen2023self,
  title={Self-Reference Deep Adaptive Curve Estimation for Low-Light Image Enhancement},
  author={Wen, Jianyu and Wu, Chenhao and Zhang, Tong and Yu, Yixuan and Swierczynski, Piotr},
  journal={arXiv preprint arXiv:2308.08197},
  year={2023}
}
```
- Thanks for all related work and workers.
