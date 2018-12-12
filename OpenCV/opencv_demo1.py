# -*- coding: utf-8 -*-
# author:maguichang time:2018/12/12

# opencv 入门
import cv2
import numpy as np

img = cv2.imread("C:\\Users\\dell\\Desktop\\cv1.jpg")
print(img.shape) # 图片的像素大小

emptyImage = img.copy()
cv2.namedWindow("Image")
# cv2.imshow("Image",img)
cv2.imshow("Image",emptyImage)

"""
创建/复制图像
对于JPG图片，对于JPEG，其表示的是图像的质量，用0-100的整数表示，默认为95。 
注意，cv2.IMWRITE_JPEG_QUALITY类型为Long，必须转换成int。

对于PNG图片，cv2.IMWRITE_PNG_COMPRESSION，从0到9,压缩级别越高，图像尺寸越小。默认级别为3
"""

cv2.imwrite("C:\\Users\\dell\\Desktop\\cv12.png",img,[int(cv2.IMWRITE_PNG_COMPRESSION),1])

"""
cv2.waitKey(parameter)
parameter = NONE & 0表示一直显示
除此之外表示显示的毫秒数
"""
cv2.waitKey(3000)
cv2.destroyAllWindows()
