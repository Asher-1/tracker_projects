#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  video_project
FILE_NAME    :  flann_matcher
AUTHOR       :  DAHAI LU
TIME         :  2019/8/8 上午11:04
PRODUCT_NAME :  PyCharm
"""

from __future__ import print_function
import os
import cv2
import numpy as np
from enum import Enum
from tools import tools
from tools.timer_wrapper import timer_wrapper

'''
在复杂的环境中，FLANN算法不容易将对象混淆，而像素级算法则容易混淆--（查看Matches输出图可知）
单应性估计：
    由于我们的对象是平面且固定的，所以我们就可以找到两幅图片特征点的单应性变换。得到单应性变换的矩阵后就可以计算对应的目标角点
'''

__all__ = ['FeatureExtraction', 'Method']


class Method(Enum):
    ORB = 1
    SIFT_BRUTE_FORCE = 2
    SIFT_FLANN = 3


class FeatureExtraction(object):
    def __init__(self,
                 method=Method.SIFT_FLANN,
                 n_features=10000,
                 pattern_image_scale=1.0,
                 target_image_scale=0.5,
                 distance=70,
                 mini_match_count=10,
                 verbose=False):
        print('Preprocessing...')

        # 1 - ORB, 2 - SIFT BRUTE-FORCE, 3 - SIFT FLANN
        self.METHOD = method
        self.verbose = verbose
        self.PATTERNIMAGESCALE = pattern_image_scale
        self.TARGETIMAGESCALE = target_image_scale
        self.DISTANCE = distance

        self.ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
        self.DESTINATIONPATH = os.path.join(self.ROOT_PATH, "RESULTS")  # 目的地路径

        self.INITSHAPE = (480, 640, 3)
        self.INITDTYPE = np.uint8

        self.mini_match_count = mini_match_count
        self.mini_side_image = 160

        self.red = (20, 140, 255)
        self.blue = (220, 102, 20)

        # IMPORTANT VARIABLES
        self.mask_enable = False
        if self.verbose:
            print('Loading pattern images...')
        self.patternPaths = tools.get_filepaths(os.path.join(self.ROOT_PATH, "patterns"))
        if self.verbose:
            print(self.patternPaths)

        self.patternImagesSetOriginal = []
        self.patternImagesSet = []

        for f in self.patternPaths:
            self.patternImagesSetOriginal.append(cv2.imread(f))

        self.keyPointsLogoDetector = None
        self.keyPointsPatternDetector = None
        self.keyPointsMatcher = None
        self.FLANN_INDEX_KDTREE = 1

        if self.METHOD == Method.ORB:
            self.keyPointsPatternDetector = cv2.ORB_create(nfeatures=n_features)
            self.keyPointsMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.METHOD == Method.SIFT_BRUTE_FORCE:
            self.keyPointsPatternDetector = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
            self.keyPointsMatcher = cv2.BFMatcher()
        elif self.METHOD == Method.SIFT_FLANN:
            self.keyPointsPatternDetector = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)

            index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=3)
            search_params = dict(checks=30)
            self.keyPointsMatcher = cv2.FlannBasedMatcher(index_params, search_params)

        self._resize_set(draw_points=True)

    def __del__(self):
        cv2.destroyAllWindows()
        print('Finished!')

    def set_pattern_images(self, src_path):
        self.patternPaths = []
        self.patternImagesSetOriginal = []
        print('Loading pattern images...')
        if isinstance(src_path, str) and os.path.exists(src_path):
            if os.path.isdir(src_path):
                self.patternPaths = tools.get_filepaths(os.path.expanduser(src_path))
            elif os.path.isfile(src_path):
                self.patternPaths = [src_path]
            else:
                pass
        elif isinstance(src_path, np.ndarray):
            self.patternPaths.append(None)
            self.patternImagesSetOriginal.append(src_path)
        else:
            assert ValueError("cannot recognize type {}".format(type(src_path)))
        if None not in self.patternPaths:
            print(list(map(os.path.basename, self.patternPaths)))

        for f in self.patternPaths:
            if f is not None:
                self.patternImagesSetOriginal.append(cv2.imread(f))

        self._resize_set(draw_points=True)

    def _resize_set(self, draw_points=False):
        if self.verbose:
            print('Resizing entire sets...')

        scale = self.PATTERNIMAGESCALE
        if self.METHOD == Method.ORB:
            self.scales_list = np.linspace(scale / 20, scale, 10)
        else:
            self.scales_list = np.linspace(scale, scale, 1)
        if self.verbose:
            print('Scale linspace: ' + str(self.scales_list))

        self.patternImagesSet = []

        for image in self.patternImagesSetOriginal:
            self.patternImagesSet.append(self._resize_with_keypoints_and_descriptor(
                image, self.keyPointsPatternDetector,
                scale, draw_points=draw_points))

    @timer_wrapper
    def _resize_with_keypoints_and_descriptor(self, image, detector, scale=None, draw_points=False):
        mini_side = min(image.shape[:2])
        if scale is not None and scale != 1.0 and mini_side > self.mini_side_image:
            resized_image = cv2.resize(image, None,
                                       fx=scale,
                                       fy=scale,
                                       interpolation=cv2.INTER_CUBIC)
            mini_side = min(image.shape[:2])
            if mini_side > self.mini_side_image:
                image = resized_image

        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('GRAY NORMAL', imageGray)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # imageGray = clahe.apply(imageGray)
        # imageGray = cv2.equalizeHist(imageGray, None)
        # cv2.imshow('GRAY - HISTOGRAM OPERATIONS', imageGray)
        keypoints, descriptor = detector.detectAndCompute(imageGray, None)
        imageWithKeypoints = np.zeros(image.shape, image.dtype)
        if draw_points:
            cv2.drawKeypoints(image, keypoints, imageWithKeypoints, self.red,
                              cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        else:
            imageWithKeypoints = image
        return keypoints, descriptor, image, imageWithKeypoints

    def compare_with_pattern(self, target_image, name, mask_enable=False, save_flag=False):
        self.mask_enable = mask_enable

        img_keypoints, img_descriptor, resized_target_image, target_image_with_keypoints = \
            self._resize_with_keypoints_and_descriptor(target_image,
                                                       self.keyPointsPatternDetector,
                                                       self.TARGETIMAGESCALE,
                                                       draw_points=True)

        self.totalResults = self._get_set_results(img_keypoints, img_descriptor)

        index = self._get_best_index()

        output_images, rect_list = self._generate_output_images('PATTERN', index, name,
                                                                resized_target_image,
                                                                target_image_with_keypoints,
                                                                img_keypoints)

        if save_flag:
            tools.save_results(self.DESTINATIONPATH, output_images, 'PATTERN', name)

        output_images.append(self.patternPaths[index])
        return output_images, rect_list

    @timer_wrapper
    def track(self, target_data, pattern_data=None):
        """
        Note: if pattern_data is None then use self.patternImagesSetOriginal generated by calling set_pattern_images
        :param target_data: the image data to be targeted on pattern data
        :param pattern_data: the image data to be referenced and matched
        :return: rect_list represents the list of targets bounding boxes on pattern image data
        """
        assert not (pattern_data is None and len(self.patternImagesSetOriginal) == 0), "no pattern images can be used!"
        pattern_image_list, target_image_list = tools.preprocess(pattern_data, target_data, self.patternPaths)
        if pattern_image_list is not None and len(pattern_image_list) > 0:
            self.patternImagesSetOriginal = pattern_image_list
            assert len(self.patternImagesSetOriginal) == len(self.patternPaths)
            self._resize_set(draw_points=False)
        rect_list = []
        for target_image in target_image_list:
            img_keypoints, img_descriptor, resized_target_image, target_image_with_keypoints = \
                self._resize_with_keypoints_and_descriptor(target_image,
                                                           self.keyPointsPatternDetector,
                                                           self.TARGETIMAGESCALE)

            self.totalResults = self._get_set_results(img_keypoints, img_descriptor)

            index = self._get_best_index()

            tmp_rect_list = self._generate_output_images('PATTERN', index, "",
                                                         resized_target_image,
                                                         target_image_with_keypoints,
                                                         img_keypoints,
                                                         visualization=False)
            rect_list.extend(tmp_rect_list)
        return rect_list

    @timer_wrapper
    def _get_set_results(self, imgKeyPoints, imgDescriptor):
        results = []
        for ind, imageFromSet in enumerate(self.patternImagesSet):
            # Match descriptors.
            image_descriptor_num = 0
            good_matches_num = 0
            inliers_num = 0
            statistics = (image_descriptor_num, good_matches_num, inliers_num)
            dst_points = None
            M = None
            good = None
            matches_mask = None
            w = None
            h = None

            if imgDescriptor is not None and imageFromSet[1] is not None:
                image_descriptor_num = len(imgDescriptor)
                good = []
                if self.METHOD == Method.ORB or \
                        ((self.METHOD == Method.SIFT_BRUTE_FORCE or self.METHOD == Method.SIFT_FLANN) and
                         len(imgDescriptor) > 1 and len(imageFromSet[1]) > 1):
                    if self.METHOD == Method.ORB:
                        matches = self.keyPointsMatcher.match(imageFromSet[1], imgDescriptor)
                        # Sort them in the order of their distance.
                        matches = sorted(matches, key=lambda x: x.distance)
                        for m in matches:
                            if m.distance < self.DISTANCE:
                                good.append(m)
                    # 利用近似k近邻算法去寻找一致性，FLANN方法比BF（Brute-Force）方法快的多：
                    elif self.METHOD == Method.SIFT_BRUTE_FORCE or self.METHOD == Method.SIFT_FLANN:
                        # 函数返回一个训练集和询问集的一致性列表
                        matches = self.keyPointsMatcher.knnMatch(imageFromSet[1], imgDescriptor, k=2)
                        for match in matches:
                            # 用比值判别法（ratio test）删除离群点
                            if len(match) == 2:
                                m, n = match
                                # 这里使用的kNN匹配的k值为2（在训练集中找两个点），第一个匹配的是最近邻，第二个匹配的是次近邻。直觉上，一个正确的匹配会更接近第一个邻居。
                                # 换句话说，一个[不]正确的匹配，[两个邻居]的距离是相似的。因此，我们可以通过查看二者距离的不同来评判距匹配程度的好坏。
                                # 比值检测认为第一个匹配和第二个匹配的比值小于一个给定的值（一般是0.5），这里是0.7：
                                if m.distance < self.DISTANCE * n.distance / 100:
                                    good.append(m)

                    good_matches_num = len(good)

                    matches_mask = []
                    M = None
                    if len(good) >= self.mini_match_count:
                        '''
                        # 有了H（M）单应性矩阵，我们可以查看源点被映射到query image中的位置

                        计算单应性矩阵(homography)，在这个函数参数中，输入的src_pts和dst_pts是两个对应的序列，这两组序列的每一对数据一一匹配，其中既有正确的匹配，也有错误的匹配
                        ，正确的可以称为内点，错误的称为外点，RANSAC方法就是从这些包含错误匹配的数据中，分离出正确的匹配，并且求得单应矩阵。

                        返回值中M为变换矩阵。mask是掩模，online的点。

                        mask：标记矩阵，标记内点和外点.他和m1，m2的长度一样，当一个m1和m2中的点为内点时，mask相应的标记为1，反之为0，说白了，通过mask我们最终可以知道序列中哪些是内点，哪些是外点。
                        M(model)：就是我们需要求解的单应矩阵.
                        ransacReprojThreshold=0.0：为阈值，当某一个匹配与估计的假设小于阈值时，则被认为是一个内点，这个阈值，openCV默认给的是3，后期使用的时候自己也可以修改。
                        confidence：为置信度，其实也就是人为的规定了一个数值，这个数值可以大致表示RANSAC结果的准确性，这个值初始时被设置为0.995
                        maxIters：为初始迭代次数，RANSAC算法核心就是不断的迭代，这个值就是迭代的次数，默认设为了2000

                        这个函数的前期，主要是设置了一些变量然后赋初值，然后转换相应的格式等等。

                        //后面，由变换矩阵，求得变换后的物体边界四个点  
                        plt.imshow(inliers_num), plt.show()
                        '''
                        # # 获取关键点的坐标
                        # 1. # 将所有好的匹配的对应点的坐标存储下来，就是为了从序列中随机选取4组，以便下一步计算单应矩阵
                        # [[x,y],]
                        src_pts = np.float32([imageFromSet[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([imgKeyPoints[n.trainIdx].pt for n in good]).reshape(-1, 1, 2)
                        # 2.# 单应性估计
                        #    由于我们的对象是平面且固定的，所以我们就可以找到两幅图片特征点的单应性变换。得到单应性变换的矩阵后就可以计算对应的目标角点：
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, None, 2000,
                                                     0.995, )  # 单应矩阵估计,利用RANSAC方法计算单应矩阵，,置信度设为0.99 循环次数设置为2000
                        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if self.mask_enable:
                            # 把内点转换为drawMatches可以使用的格式
                            # 通过cv2.drawMatchesKnn画出匹配的特征点，再将好的匹配返回
                            matches_mask = mask.ravel().tolist()

                        # 3.# 有了H单应性矩阵，我们可以查看源点被映射到query image中的位置
                        h, w = imageFromSet[2].shape[:2]
                        # print(h, w)
                        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                        # 4.# perspectiveTransform返回点的列表
                        # if M is not None:
                        #     dst = cv2.perspectiveTransform(pts, M)
                        #     dst_points = [np.int32(dst)]  # //由变换矩阵，求得变换后的物体边界四个点.反复计算，求得4个点坐标
                        # print(dst_points)

                        # 5.# 计算非野点个数
                        # // 状态为0表示野点(离群点)
                        inliers_num = len([x for x in mask if x != 0])
                    else:
                        if self.verbose:
                            print('{} matches for {}\tNot enough matches!'.format(len(good), self.patternPaths[ind]))

                    statistics = (image_descriptor_num, good_matches_num, inliers_num)

            results.append(
                (statistics, dst_points, (M, (w, h)), good, matches_mask))
        return results  # 这个结果传入下面函数：

    @timer_wrapper
    def _generate_output_images(self, name, index, additional_name,
                                target_image, target_with_keypoints, img_keypoints, visualization=True):
        #
        matches_image = np.zeros(self.INITSHAPE, self.INITDTYPE)
        warped_image = np.zeros(self.INITSHAPE, self.INITDTYPE)
        # 单应性矩阵图（就是上面都是圈圈的图）
        rect_list = []
        if self.totalResults[index][2][0] is not None and self.totalResults[index][2][1] is not None:
            # cv2.polylines(homograpyImage, self.totalResults[index][1], True, [0, 255, 0], 5, cv2.LINE_AA)
            # 截出图上的logo图
            warped_image = cv2.warpPerspective(target_image, self.totalResults[index][2][0],
                                               self.totalResults[index][2][1], None,
                                               cv2.WARP_INVERSE_MAP,
                                               cv2.BORDER_CONSTANT, (0, 0, 0))
            # cv2.imshow("warped image", warped_image)
            rect_list = tools.get_bounding_box(warped_image)

            if visualization:
                pattern_images = self.patternImagesSet[index][2].copy()
                tools.draw_targets(pattern_images, rect_list)
                if self.totalResults[index][3] is not None and self.totalResults[index][4] is not None:
                    # 通过cv2.drawMatchesKnn画出匹配的特征点，再将好的匹配返回
                    matches_image = cv2.drawMatches(pattern_images,
                                                    self.patternImagesSet[index][0],
                                                    target_with_keypoints,
                                                    img_keypoints,
                                                    self.totalResults[index][3],
                                                    None,
                                                    self.blue, self.red,
                                                    self.totalResults[index][4],
                                                    cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                if self.verbose:
                    print('Number [ ' + name + ' ]: ' + str(additional_name))
                    print('Length of best image descriptor [ ' + name + ' ]: ' + str(self.totalResults[index][0][0]))
                    print('Good matches [ ' + name + ' ]: ' + str(self.totalResults[index][0][1]))
                    print('Inliers [ ' + name + ' ]: ' + str(self.totalResults[index][0][2]))

                output_images = [matches_image, warped_image, self.patternImagesSet[index][2],
                                 self.patternImagesSet[index][3], target_with_keypoints, target_image]
                return output_images, rect_list
        return rect_list

    def change_pattern_scale(self, pattern_scale):
        self.PATTERNIMAGESCALE = pattern_scale
        self._resize_set(draw_points=True)

    def change_target_scale(self, target_scale):
        self.TARGETIMAGESCALE = target_scale

    def _get_best_index(self):
        counter = 0
        best_result = 0
        index = 0
        for result in self.totalResults:
            if result[0][2] >= best_result:
                best_result = result[0][2]
                index = counter
            counter = counter + 1
        return index


# ###################################################################################################################################################

if __name__ == '__main__':
    # Parametry konstruktora:
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    patterns_path = os.path.join(ROOT_PATH, "images")
    targets_path = os.path.join(ROOT_PATH, "targets")
    scale = 1.0

    # patternNfeatures=10000，logo是人工选出来的，所以整个图都可是有用特征，
    featureExtraction = FeatureExtraction(method=Method.SIFT_FLANN, n_features=10000,
                                          pattern_image_scale=scale, target_image_scale=scale,
                                          distance=70, mini_match_count=10, verbose=False)
    featureExtraction.set_pattern_images(patterns_path)
    # featureExtraction.change_pattern_scale(0.8)  # 也就是上面的0.5
    featureExtraction.change_target_scale(0.5)  # 也就是上面的0.5

    print('Loading target images...')
    targetPaths = tools.get_filepaths(targets_path)
    print(list(map(os.path.basename, targetPaths)))
    targetImagesSet = map(cv2.imread, targetPaths)
    print('Analyze...')

    mainCounter = 0
    waitingMs = 0

    for image in targetImagesSet:
        pattern_outputs, rects = featureExtraction.compare_with_pattern(image,
                                                                        mainCounter,
                                                                        mask_enable=True,
                                                                        save_flag=True)
        if len(rects) < 1:
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
