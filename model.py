# coding:utf-8
##添加文本方向 检测模型，自动检测文字方向，0、90、180、270
from math import *

import cv2
import numpy as np
from PIL import Image
# import sys
import time
import torch
from crnn1.models import crnn as crnn
from angle.predict import predict as angle_detect  ##文字方向检测
from torch_crnn.crnn import crnnOcr
from crnn1 import test
from crnn1 import utils
from ctpn.text_detect import text_detect
from crnn1 import alphabets


def crnnRec(im, text_recs, ocrMode, adjust=False):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im:Array
    @@text_recs:text box

    """
    index = 0
    results = {}
    xDim, yDim = im.shape[1], im.shape[0]

    # crnn network
    crnn_model_path = 'crnn1/trained_models/mixed_second_finetune_acc97p7.pth'
    tmodel = crnn.CRNN(imgH=32, nc=1, nclass=len(alphabets.alphabet)+1, nh=256)
    if torch.cuda.is_available():
        tmodel = tmodel.cuda()
    print('loading pre trained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    tmodel.load_state_dict(torch.load(crnn_model_path))
    converter = utils.strLabelConverter(alphabets.alphabet)

    for index, rec in enumerate(text_recs):
        results[index] = [
            rec,
        ]
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2),
                   min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  ##图像倾斜角度

        partImg = dumpRotateImage(im, degree, pt1, pt2, pt3, pt4)
        # 根据ctpn进行识别出的文字区域，进行不同文字区域的crnn识别
        image = Image.fromarray(partImg).convert('L')
        # 进行识别出的文字识别
        # if ocrMode == 'keras':
        #    sim_pred = ocr(image)
        if ocrMode == 'pytorch':
            sim_pred = test.crnn_recognition(image, converter, tmodel) # crnnOcr(image)

        results[index].append(sim_pred)  # 识别文字

    return results


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(
        img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation,
                                  np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation,
                                  np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])):min(ydim - 1, int(pt3[1])),
                         max(1, int(pt1[0])):min(xdim - 1, int(pt3[0]))]
    # height,width=imgOut.shape[:2]
    return imgOut


def model(img, model, adjust=False, detectAngle=False):
    """
    @@param:img,输入的图像数组
    @@param:model,选择的ocr模型，支持pytorch版本
    @@param:adjust 调整文字识别结果
    @@param:detectAngle,是否检测文字朝向
    
    """
    angle = 0
    if detectAngle:
        # 进行文字旋转方向检测，分为[0, 90, 180, 270]四种情况,下面所用的是逆时针旋转，因此需要改正
        angle = angle_detect(img=np.copy(img))  ##文字朝向检测
        print('The angel of this character is:', angle)
        im = Image.fromarray(img)
        print('Rotate the array of this img!')
        if angle == 90:
            im = im.transpose(Image.ROTATE_270)
        elif angle == 180:
            im = im.transpose(Image.ROTATE_180)
        elif angle == 270:
            im = im.transpose(Image.ROTATE_90)
        img = np.array(im)
    print(img)
    # 进行图像中的文字区域的识别
    t = time.time()
    text_recs, tmp, img = text_detect(img)
    print('image area recognition finished!')
    print("It takes time:{}s".format(time.time() - t))
    # 识别区域排列
    text_recs = sort_box(text_recs)
    # 文本识别
    t = time.time()
    result = crnnRec(img, text_recs, ocrMode=model, adjust=adjust)
    print('end-to-end text recognition finished!')
    print("It takes time:{}s".format(time.time() - t))
    return result, tmp, angle


def sort_box(box):
    """
    对box排序,及页面进行排版
    text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
    """

    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box
