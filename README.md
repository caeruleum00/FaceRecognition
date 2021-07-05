# All Face Recontion Project
## Mask On/Off, Age, Gender, Emotion
Framework: MXNet
Mask, age, gender model is based on Retinaface Anti Cov.  
https://github.com/deepinsight/insightface  
https://github.com/yeyupiaoling/Age-Gender-MXNET  
https://github.com/deepinx/age-gender-estimation  
  
## Introduction
This project is an analysis project about face such as face recognition, whether to wearing a mask, age, gender and emotion.:laughing: 

## Environment
Python3  
Mxnet  
GTX-1080Ti  
Ubuntu 18.04  
MXNet-cu101  

## Installation
Prepare the environment.

Clone the repository.

  pip install -r requirments.txt

## Codes
### Training
Downlad IMDB-WIKI dataset (face only from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/.  (+ AgeDB)

Unzip them under ./data or others path.  
Pre-process each images in dataset and package it in a rec file.  
You can first train on imdb, then fine tune on wiki dataset. Train MobileNet 0.25X on a GPU such as GTX-1080Ti according to the following command  

  CUDA_VISIBLE_DEVICES='0' python -u train.py --data-dir $DATA_DIR --prefix './models/model' --network m1 --multiplier 0.25 --per-batch-size 128 --lr 0.01 --lr-steps '10000' --ckpt 2  
Instead, you can edit train.sh and run sh ./train.sh to train your models.

### Testing
Download the ESSH model from BaiduCloud or GoogleDrive and place it in ./ssh-model/.  
Here: https://drive.google.com/drive/folders/1eX_i0iZxZTMyJ4QccYd2F4x60GbZqQQJ?usp=drive_open  
You can use python test.py to test the pre-trained models or your own models.  


## References

```
  
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}
```


