#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  video_project
FILE_NAME    :  cv_tracker_test
AUTHOR       :  DAHAI LU
TIME         :  2019/8/13 下午5:25
PRODUCT_NAME :  PyCharm
"""

import sys
import cv2
from core.cv_tracker import CvTracker
from core.cv_tracker import TrackerType
from tools import tools

if __name__ == '__main__':

    # 设置加载的视频文件
    videoPath = 'test/video_3.flv'

    # 创建video capture 来读取视频文件
    cap = cv2.VideoCapture(videoPath)

    # 读取第一帧
    ret, frame = cap.read()
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)

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

        # show frame
        cv2.imshow('MultiTracker', frame)

        # quit on ESC button
        if cv2.waitKey(1) == 27:  # Esc pressed
            break
