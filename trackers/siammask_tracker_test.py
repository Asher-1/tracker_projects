#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  video_project
FILE_NAME    :  siammask_tracker_test
AUTHOR       :  DAHAI LU
TIME         :  2019/8/13 下午4:42
PRODUCT_NAME :  PyCharm
"""

import sys
import cv2
from tools import tools
from core.SiamMaskTracker.siam_mask_tracker import SiamMaskTracker

if __name__ == '__main__':
    tracker = SiamMaskTracker()

    # Parse Image file
    cap = cv2.VideoCapture("test/video_3.flv")
    ret, frame = cap.read()

    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)

    # 如果无法读取视频文件就退出
    if not ret:
        print('Failed to read video')
        sys.exit(1)

    # Select ROI
    cv2.namedWindow("camera1", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('camera1', frame, False, False)
    except Exception as e:
        print(e)
        sys.exit()

    toc = 0
    f = 0
    while cap.isOpened():
        ret, im = cap.read()
        if not ret:
            break

        tic = cv2.getTickCount()
        if f == 0:  # init
            tracker.ini_tracker(im, init_rect)
        elif f > 0:  # tracking
            im = tracker.update(im, display=True)  # track
            bboxes = tracker.get_bounding_boxes()
            tools.draw_targets(img=im, rect_list=bboxes)
            cv2.imshow('camera1', im)
            key = cv2.waitKey(1)
        f += 1
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
