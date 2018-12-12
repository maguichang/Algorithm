# -*- coding: utf-8 -*-
# author:maguichang time:2018/12/12

# 访问像素
import cv2
import numpy as np

"""
其中j，i分别表示图像的行和列。
opencv中矩阵是以BGR而不是RGB显示像素的大小，包括 Gamegear，不过其他的都 大部分采用rgb格式的
"""
def salt(img, n):
    for k in range(n):
        i = int(np.random.random() * img.shape[1]);
        j = int(np.random.random() * img.shape[0]);
        if img.ndim == 2:
            img[j, i] = 1
        elif img.ndim == 3:
            img[j, i, 0] = 1
            img[j, i, 1] = 1
            img[j, i, 2] = 1
    return img


if __name__ == '__main__':
    img = cv2.imread("C:\\Users\\dell\\Desktop\\cv1.jpg")
    saltImage = salt(img, 500)
    cv2.imshow("Salt", saltImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
