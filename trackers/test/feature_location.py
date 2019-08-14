#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  video_project
FILE_NAME    :  feature_locattion
AUTHOR       :  DAHAI LU
TIME         :  2019/8/7 下午3:37
PRODUCT_NAME :  PyCharm
"""

'''
基于FLANN的匹配器(FLANN based Matcher)
1.FLANN代表近似最近邻居的快速库。它代表一组经过优化的算法，用于大数据集中的快速最近邻搜索以及高维特征。
2.对于大型数据集，它的工作速度比BFMatcher快。
3.需要传递两个字典来指定要使用的算法及其相关参数等
对于SIFT或SURF等算法，可以用以下方法：
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
对于ORB，可以使用以下参数：
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12   这个参数是searchParam,指定了索引中的树应该递归遍历的次数。值越高精度越高
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
'''

import cv2
import numpy as np

# 基于FLANN的匹配器(FLANN based Matcher)定位图片
MIN_MATCH_COUNT = 10  # 设置最低特征点匹配数量为10
template = cv2.imread('../test/full.jpeg')  # queryImage
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
target = cv2.imread('../test/target.png')  # trainImage
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector创建sift检测器
sift = cv2.xfeatures2d.SIFT_create(10000)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(template_gray, None)
kp2, des2 = sift.detectAndCompute(target_gray, None)

# 创建设置FLANN匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
# 舍弃大于0.7的匹配
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    # 获取关键点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # 计算变换矩阵和MASK
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()
    h, w = template.shape[:2][::-1]
    # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
    # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # scene_points = cv2.perspectiveTransform(pts, M)

    # rect = cv2.minAreaRect(scene_points)
    # box = cv2.boxPoints(rect)
    # cv2.drawContours(template, [np.int0(box)], 0, (0, 255, 0), 2)

    warpedImage = cv2.warpPerspective(target_gray, M,
                                      (h, w), None,
                                      cv2.WARP_INVERSE_MAP,
                                      cv2.BORDER_CONSTANT, (0, 0, 0))
    ret, thresh = cv2.threshold(warpedImage, 1, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    image, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    x1, y1, w1, h1 = cv2.boundingRect(contours[0])
    cv2.rectangle(template, (x1, y1), (x1+w1, y1+h1), color=(0, 255, 0), thickness=2)
    # cv2.imshow("temp.png", template)

    # cv2.polylines(template, [np.int32(box)], True, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.imwrite("temp.png", warpedImage)
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
cv2.imwrite("result.png", result)
cv2.imshow('Warped image', warpedImage)
cv2.imshow('result image', result)

key = cv2.waitKey(0)
