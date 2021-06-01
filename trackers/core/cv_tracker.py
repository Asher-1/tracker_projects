#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  video_project
FILE_NAME    :  tracker
AUTHOR       :  DAHAI LU
TIME         :  2019/8/12 下午5:34
PRODUCT_NAME :  PyCharm
"""

import sys
import cv2
from enum import Enum
from tools import tools


class TrackerType(Enum):
    BOOSTING = 1
    MIL = 2
    KCF = 3
    TLD = 4
    CSRT = 5
    MOSSE = 6
    GOTURN = 7
    MEDIANFLOW = 8


class CvTracker(object):
    def __init__(self, method=TrackerType.MEDIANFLOW, verbose=False):
        print('Default tracking algorithm is MEDIANFLOW \n')
        self._trackerTypes = method
        self._verbose = verbose

    def _create_tracker(self):
        # 通过跟踪器的名字创建跟踪器
        if self._trackerTypes == TrackerType.BOOSTING:
            tracker = cv2.TrackerBoosting_create()
        elif self._trackerTypes == TrackerType.MIL:
            tracker = cv2.TrackerMIL_create()
        elif self._trackerTypes == TrackerType.KCF:
            tracker = cv2.TrackerKCF_create()
        elif self._trackerTypes == TrackerType.TLD:
            tracker = cv2.TrackerTLD_create()
        elif self._trackerTypes == TrackerType.MEDIANFLOW:
            tracker = cv2.TrackerMedianFlow_create()
        elif self._trackerTypes == TrackerType.GOTURN:
            tracker = cv2.TrackerGOTURN_create()
        elif self._trackerTypes == TrackerType.MOSSE:
            tracker = cv2.TrackerMOSSE_create()
        elif self._trackerTypes == TrackerType.CSRT:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available tracker name')
        return tracker

    def set_targets(self, frame, bboxes=None):
        # 创建多跟踪器对象
        self._multi_tracker = cv2.MultiTracker_create()
        # 初始化多跟踪器
        for bbox in bboxes:
            self._multi_tracker.add(self._create_tracker(), frame, tuple(bbox))

    def update(self, frame):
        ret, boxes = self._multi_tracker.update(frame)
        return ret, boxes


if __name__ == '__main__':

    # 设置加载的视频文件
    videoPath = '../test/video_3.flv'

    # 创建video capture 来读取视频文件
    cap = cv2.VideoCapture(videoPath)

    # 读取第一帧
    ret, frame = cap.read()
    cv_tracker = CvTracker(method=TrackerType.MEDIANFLOW, verbose=False)

    # 如果无法读取视频文件就退出
    if not ret:
        print('Failed to read video')
        sys.exit(1)

    # 选择框
    bboxes, colors = tools.select_roi(frame=frame)

    cv_tracker.set_targets(frame=frame, bboxes=bboxes)

    # 处理视频并跟踪对象
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 获取后续帧中对象的更新位置
        ret, bboxes = cv_tracker.update(frame=frame)
        # cv_tracker.set_targets(frame=frame, bboxes=bboxes)

        # 绘制跟踪的对象
        tools.draw_targets(img=frame, rect_list=bboxes)
        # for i, newbox in enumerate(bboxes):
        #     p1 = (int(newbox[0]), int(newbox[1]))
        #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #     cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        # show frame
        cv2.imshow('MultiTracker', frame)

        # quit on ESC button
        if cv2.waitKey(1) == 27:  # Esc pressed
            break
