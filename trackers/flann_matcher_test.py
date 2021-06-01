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
from tools import tools
from core.flann_matcher import Method
from core.flann_matcher import FeatureExtraction

if __name__ == '__main__':
    # Parametry konstruktora:
    ROOT_PATH = os.path.dirname(__file__)
    patterns_path = os.path.join(ROOT_PATH, "images")
    targets_path = os.path.join(ROOT_PATH, "targets")
    scale = 1.0
    distance = 70
    mini_match_count = 10
    pattern_n_features = 10000

    featureExtraction = FeatureExtraction(method=Method.SIFT_FLANN, n_features=pattern_n_features,
                                          pattern_image_scale=scale, target_image_scale=scale,
                                          distance=distance, mini_match_count=mini_match_count, verbose=False)
    featureExtraction.set_pattern_images(patterns_path)
    # featureExtraction.change_pattern_scale(0.8)  # 也就是上面的0.5
    featureExtraction.change_target_scale(2.0)  # 也就是上面的0.5

    print('Loading target images...')
    targetPaths = tools.get_filepaths(targets_path)
    print(list(map(os.path.basename, targetPaths)))
    targetImagesSet = map(cv2.imread, targetPaths)
    print('Analyze...')

    mainCounter = 0
    waitingMs = 0

    for target_image in targetImagesSet:
        pattern_outputs, rect_list = featureExtraction.compare_with_pattern(target_image,
                                                                            mainCounter,
                                                                            mask_enable=True,
                                                                            save_flag=True)
        if len(rect_list) < 1:
            print("cannot find target location")
        # featureExtraction.show_results(pattern_output, str(mainCounter))
        cv2.imshow('Warped image - Pattern', pattern_outputs[0])

        mainCounter = mainCounter + 1

        key = cv2.waitKey(waitingMs) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            if waitingMs is 0:
                waitingMs = 1
            else:
                waitingMs = 0

    del featureExtraction
