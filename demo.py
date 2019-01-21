# coding:utf-8
import time
import numpy as np
from PIL import Image
import model

if __name__ == '__main__':

    im = Image.open("test/ttttt.png")
    img = np.array(im.convert('RGB'))
    t = time.time()

    # result,img,angel分别对应-识别结果，图像的数组，文字旋转角度

    result, img, angle = model.model(
        img=img, model='pytorch', adjust=False, detectAngle=False)
    # use VGG to detect the angle of image, pytorch to get crnn working.
    print("Totally it takes time:{}s".format(time.time() - t))
    print("---------------------------------------")
    for key in result:
        print(result[key][1])
