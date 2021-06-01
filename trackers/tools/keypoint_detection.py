#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  video_project
FILE_NAME    :  keypoint_detection
AUTHOR       :  DAHAI LU
TIME         :  2019/8/8 上午11:08
PRODUCT_NAME :  PyCharm
"""

from __future__ import print_function
import numpy as np
import cv2
import imutils

'''
原理：必须有至少Ñ沿着连续像素圆形周边具有半径- R是所有或者亮或更暗比中心像素由阈值t
疑问：是否可以修改参数，半径-R和N值？
运用：特征匹配
'''

image = cv2.imread("test/full.jpeg")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if imutils.is_cv2():
    detector = cv2.FeatureDetector_create("FAST")
    kps = detector.detect(gray)

else:
    detector = cv2.FastFeatureDetector_create()
    kps = detector.detect(gray, None)

print("# of keypoints: {}".format(len(kps)))

for kp in kps:
    r = int(0.5 * kp.size)
    (x, y) = np.int0(kp.pt)
    cv2.circle(image, (x, y), r, (0, 255, 255), 1)

cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)
