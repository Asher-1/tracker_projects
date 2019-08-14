#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  video_project
FILE_NAME    :  tools
AUTHOR       :  DAHAI LU
TIME         :  2019/8/9 上午10:35
PRODUCT_NAME :  PyCharm
"""
import os
import cv2
import numpy as np
from random import randint
from .timer_wrapper import timer_wrapper


def preprocess(pattern_data, target_data, pattern_paths):
    # pre-processing for pattern data
    if pattern_data is not None:
        del pattern_paths[:]
    pattern_image_list = []
    if isinstance(pattern_data, str) and os.path.exists(pattern_data):
        if os.path.isdir(pattern_data):
            pattern_paths = get_filepaths(os.path.expanduser(pattern_data))
        elif os.path.isfile(pattern_data):
            pattern_paths = [pattern_data]
        else:
            pass
        for f in pattern_paths:
            if f is not None:
                pattern_image_list.append(cv2.imread(f))
    elif isinstance(pattern_data, np.ndarray):
        pattern_paths.append(None)
        pattern_image_list.append(pattern_data)
    else:
        pattern_image_list = None

    # pre-processing for target data
    target_image_list = []
    if isinstance(target_data, str) and os.path.exists(target_data):
        target_image_list.append(cv2.imread(os.path.expanduser(target_data)))
    elif isinstance(target_data, np.ndarray):
        target_image_list.append(target_data)
    elif isinstance(target_data, list):
        for data in target_data:
            if isinstance(data, str) and os.path.exists(data):
                target_image_list.append(cv2.imread(os.path.expanduser(data)))
            elif isinstance(data, np.ndarray):
                target_image_list.append(data)
            else:
                assert ValueError("cannot recognize type {}".format(type(data)))
    else:
        assert ValueError("cannot recognize type {}".format(type(target_data)))

    return pattern_image_list, target_image_list


def save_results(destination_path, output_images, name, additional_name):
    new_catalogue_path = os.path.join(destination_path, str(additional_name), name)
    os.makedirs(new_catalogue_path, exist_ok=True)
    cv2.imwrite(os.path.join(new_catalogue_path, '1_Matches_' + name + '.png'), output_images[0])
    cv2.imwrite(os.path.join(new_catalogue_path, '2_Warped image_' + name + '.png'), output_images[1])
    cv2.imwrite(os.path.join(new_catalogue_path, '3_Pattern image_' + name + '.png'), output_images[2])
    cv2.imwrite(os.path.join(new_catalogue_path, '4_Keypoints - pattern_' + name + '.png'), output_images[3])
    cv2.imwrite(os.path.join(new_catalogue_path, '5_Keypoints - image_' + name + '.png'), output_images[4])
    cv2.imwrite(os.path.join(new_catalogue_path, '6_Original image_' + name + '.png'), output_images[5])


def show_results(pattern_outputs, name):
    cv2.imshow('1_Matches_' + name, pattern_outputs[0])
    cv2.imshow('2_Warped image_' + name, pattern_outputs[1])
    cv2.imshow('3_Pattern image_' + name, pattern_outputs[2])
    cv2.imshow('4_Keypoints-pattern_' + name, pattern_outputs[3])
    cv2.imshow('5_Keypoints - image_' + name, pattern_outputs[4])
    cv2.imshow('6_Original image_' + name, pattern_outputs[5])


def get_bounding_box(warped_image):
    if len(warped_image.shape) == 3:
        warped_image_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    else:
        warped_image_gray = warped_image
    ret, thresh = cv2.threshold(warped_image_gray, 1, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None or len(contours) == 0:
        return []
    return list(map(cv2.boundingRect, contours))


def draw_targets(img, rect_list, scale=1.0, color=(0, 255, 0), thickness=2):
    if len(rect_list) == 0:
        return
    if isinstance(color, tuple):
        colors = [color] * len(rect_list)
    else:
        colors = color
    assert scale != 0
    for rect, clr in zip(rect_list, colors):
        x, y, w, h = rect
        coord = np.array([x, y, x + w, y + h]) / scale
        coord = coord.astype(int)
        cv2.rectangle(img,
                      (coord[0], coord[1]),
                      (coord[2], coord[3]),
                      color=clr, thickness=thickness)


def get_filepaths(directory):
    file_paths = []  # 将存储所有的全文件路径的列表
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # 加入两个字符串以形成完整的文件路径
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.
    return file_paths  # 所有文件的路径


def resize_function(x):
    step = 200
    x = x - 500.0
    if x < 0:
        scale = 1 / ((1.005) ** abs(x))
    else:
        scale = (x + step) / step
    return scale


def get_roi_mannual(frame):
    bboxes = []
    while True:
        # 在对象上绘制边界框selectROI的默认行为是从fromCenter设置为false时从中心开始绘制框，可以从左上角开始绘制框
        bbox = cv2.selectROI('MultiTracker', frame)
        bboxes.append(bbox)
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0)
        if k == 113:  # q is pressed
            break
    frame_list = []
    for bbox in bboxes:
        frame_list.append(frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]])
    return frame_list


def select_roi(frame):
    # OpenCV的selectROI函数不适用于在Python中选择多个对象
    # 所以循环调用此函数，直到完成选择所有对象
    # 选择框
    bboxes = []
    colors = []
    while True:
        # 在对象上绘制边界框selectROI的默认行为是从fromCenter设置为false时从中心开始绘制框，可以从左上角开始绘制框
        bbox = cv2.selectROI('MultiTracker', frame, showCrosshair=True)
        bboxes.append(bbox)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0)
        if k == 113:  # q is pressed
            break
    print('Selected bounding boxes {}'.format(bboxes))
    return bboxes, colors


def get_roi(frame, bboxes):
    frame_list = []
    for bbox in bboxes:
        frame_list.append(frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]])
    return frame_list


if __name__ == '__main__':
    img = cv2.imread("../images/full2.jpg")
    image_list = get_roi_mannual(img)
    for ind, roi_img in enumerate(image_list):
        new_name = "{}.jpg".format(ind)
        cv2.imwrite(os.path.join("../targets", new_name), roi_img)
    print("done...")
