# -*- coding: utf-8 -*-
# author:maguichang time:2018/12/12

# 分离、合并通道
import cv2
import numpy as np

img = cv2.imread("C:\\Users\\dell\\Desktop\\cv1.jpg")
b, g, r = cv2.split(img)
cv2.imshow("Blue", r)
cv2.imshow("Red", g)
cv2.imshow("Green", b)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果只想返回其中的一个通道，可以使用以下的写法，最后的索引指出所需要的通道。opencv是bgr
b = cv2.split(img)[0]
g = cv2.split(img)[1]
r = cv2.split(img)[2]

# 通道合并，cv2自带merge函数
merged = cv2.merge([b,g,r]) #前面分离出来的三个通道


