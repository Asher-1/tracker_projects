#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  video_project
FILE_NAME    :  siam_mask
AUTHOR       :  DAHAI LU
TIME         :  2019/8/12 下午7:09
PRODUCT_NAME :  PyCharm
"""

import os
import sys
import torch
import numpy as np
import argparse
import cv2
from .custom import Custom
from .utils import tools
from .utils.config_helper import load_config
from .utils.load_helper import load_pretrain


class SiamMaskTracker(object):
    def __init__(self, cpu=True):
        self._root_path = os.path.dirname(__file__)
        self._args = self._config(cpu=cpu)

        self._cfg = load_config(self._args)
        self._siammask = Custom(anchors=self._cfg['anchors'])
        if self._args.resume:
            assert os.path.isfile(self._args.resume), 'Please download {} first.'.format(self._args.resume)
            self._siammask = load_pretrain(self._siammask, self._args.resume)
        self._siammask.eval().to(self._device)

    def _config(self, cpu):
        config_file = os.path.join(self._root_path, "base/config/config_davis.json")
        model_file = os.path.join(self._root_path, "base/model/SiamMask_DAVIS.pth")

        if cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        assert os.path.exists(config_file) and os.path.exists(model_file)
        parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
        parser.add_argument('--resume', default=model_file, type=str, required=False,
                            metavar='PATH', help='path to latest checkpoint (default: none)')
        parser.add_argument('--config', dest='config', default=config_file,
                            help='hyper-parameter of SiamMask in json format')
        parser.add_argument('--cpu', action='store_true', help='cpu mode', default=cpu)
        return parser.parse_args()

    def ini_tracker(self, frame, box):
        x, y, w, h = box
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self._state = tools.siamese_init(frame, target_pos, target_sz, self._siammask, self._cfg['hp'],
                                         device=self._device)  # init tracker

    @staticmethod
    def _get_roi_img(img, rect):
        box = np.int0(cv2.boxPoints(rect))
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        cropImg = np.copy(img[y1:y1 + hight, x1:x1 + width])
        return cropImg

    def get_bounding_boxes(self):
        try:
            rect = cv2.minAreaRect(self._state['ploygon'])
        except Exception as e:
            print(e)
            return None
        box = np.int0(cv2.boxPoints(rect))
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        rect = (x1, y1, width, hight)
        return [rect]

    def update(self, frame, display=False):
        self._state = tools.siamese_track(self._state, frame, mask_enable=True, refine_enable=True,
                                          device=self._device)  # track
        if display:
            location = self._state['ploygon'].flatten()

            mask = self._state['mask'] > self._state['p'].seg_thr

            frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
            cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        return frame


if __name__ == '__main__':
    tracker = SiamMaskTracker()

    # Parse Image file
    cap = cv2.VideoCapture("../../test/video_3.flv")
    ret, frame = cap.read()
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
            im = tracker.update(im)  # track
            cv2.imshow('camera1', im)
            key = cv2.waitKey(1)
        f += 1
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
