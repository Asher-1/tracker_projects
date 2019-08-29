#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  video_project
FILE_NAME    :  flann_matcher_test
AUTHOR       :  DAHAI LU
TIME         :  2019/8/9 上午11:52
PRODUCT_NAME :  PyCharm
"""

import os
import cv2
import sys
from tools import tools
from core.flann_matcher import Method
from core.flann_matcher import FeatureExtraction

if __name__ == '__main__':
    # Parametry konstruktora:
    ROOT_PATH = os.path.dirname(__file__)
    # 设置加载的视频文件
    # videoPath = os.path.join(ROOT_PATH, "test/video_3.flv")
    videoPath = 0
    scale = 0.5
    distance = 70
    mini_match_count = 6
    pattern_n_features = 100000
    tracker = FeatureExtraction(method=Method.SIFT_FLANN, n_features=pattern_n_features,
                                pattern_image_scale=scale, target_image_scale=scale,
                                distance=distance, mini_match_count=mini_match_count, verbose=False)
    # tracker.change_target_scale(0.5)
    # tracker.change_pattern_scale(0.5)

    # 创建video capture 来读取视频文件
    cap = cv2.VideoCapture(videoPath)

    # read first frame and set patter images
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = cap.read()
    cv2.imwrite("./tmp.png", frame)

    # num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)
    tracker.set_pattern_images(frame)

    # 如果无法读取视频文件就退出
    if not ret:
        print('Failed to read video')
        sys.exit(1)

    bboxes, colors = tools.select_roi(frame)
    target_image_list = tools.get_roi(frame, bboxes)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 获取后续帧中对象的更新位置
        rect_list = tracker.track(target_image_list, frame)
        if len(rect_list) < 1:
            print("cannot find target location")
            continue

        # 绘制跟踪的对象
        tools.draw_targets(frame, rect_list, scale=scale, color=colors)

        # show frame
        cv2.imshow('MultiTracker', frame)

        # quit on ESC button
        if cv2.waitKey(1) == 27:  # Esc pressed
            break

    del tracker
