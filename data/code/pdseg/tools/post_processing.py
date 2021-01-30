# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
from PIL import Image
import time
import pickle
import shutil
import random
import copy
import sys
from skimage import morphology


def check_dir(target_dir):
    if not os.path.isdir(target_dir):
        print('Create file directory => %s' % target_dir)
        os.makedirs(target_dir)


def remove_class3And4_noise(img, post_area_threshold=36, post_length_threshold=35):
    # 去除孤立团，用左边像素替换
    # 先去除类3
    binary_mask_class3 = img.copy()
    binary_mask_class3[binary_mask_class3 != 3] = 0
    binary_mask_class3[binary_mask_class3 == 3] = 1
    contours, hierarch = cv2.findContours(binary_mask_class3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        if area <= post_area_threshold or length <= post_length_threshold:
            cnt = contours[i]
            # 左顶点
            location = tuple(cnt[cnt[:,:,0].argmin()][0])
            class_num = int(img[location[1], location[0] - 1])
            cv2.drawContours(img, [cnt], 0, class_num, -1) 
    # 再去除类4
    binary_mask_class4 = img.copy()
    binary_mask_class4[binary_mask_class4 != 4] = 0
    binary_mask_class4[binary_mask_class4 == 4] = 1
    contours, hierarch = cv2.findContours(binary_mask_class4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        if area <= post_area_threshold or length <= post_length_threshold:
            cnt = contours[i]
            # 左顶点
            location = tuple(cnt[cnt[:,:,0].argmin()][0])
            class_num = int(img[location[1], location[0] - 1])
            cv2.drawContours(img, [cnt], 0, class_num, -1) 
            
    return img


def remove_class0_noise(img, post_area_threshold=36, post_length_threshold=35):
    # 去除孤立团，用左边像素替换
    # 先去除类3
    binary_mask_class0 = img.copy()
    binary_mask_class0[binary_mask_class0 == 0] = 1
    binary_mask_class0[binary_mask_class0 != 1] = 0
    contours, hierarch = cv2.findContours(binary_mask_class0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        if area <= post_area_threshold or length <= post_length_threshold:
            cnt = contours[i]
            # 左顶点
            location = tuple(cnt[cnt[:,:,0].argmin()][0])
            class_num = int(img[location[1], location[0] - 1])
            cv2.drawContours(img, [cnt], 0, class_num, -1) 
            
    return img


# 投票
def vote_img_byOringinAndMultiBinary_addPostprocess(img_name):
    pre_area_threshold = 16
    pre_length_threshold = 25
    
    binary4_voted_img = np.array(Image.open(os.path.join(temp_results_path, binary4_path[0] + '_' + 'binary_test%s_votedBy20tta' % suffix, img_name)))
    binary4_voted_img += np.array(Image.open(os.path.join(temp_results_path, binary4_path[1] + '_' + 'binary_test%s_votedBy30tta' % suffix, img_name)))
    binary4_voted_img += np.array(Image.open(os.path.join(temp_results_path, binary4_path[2] + '_' +  'binary_test%s_votedBy30tta' % suffix, img_name)))
    binary4_voted_img += np.array(Image.open(os.path.join(temp_results_path, binary4_path[3] + '_' +  'binary_test%s_votedBy30tta' % suffix, img_name)))
    binary4_voted_img += np.array(Image.open(os.path.join(temp_results_path, binary4_path[4] + '_' +  'binary_test%s_votedBy30tta' % suffix, img_name)))
    # 票数多于0票的设置为4
    bound = 3
    binary4_voted_img[binary4_voted_img <= bound] = 0
    binary4_voted_img[binary4_voted_img > bound] = 4
    # 处理binary4_voted_img
    kernel_close_class4 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    binary4_voted_img = cv2.morphologyEx(binary4_voted_img, cv2.MORPH_CLOSE, kernel_close_class4, iterations=1)
    binary4_voted_img = cv2.medianBlur(binary4_voted_img, 5)
    # 去孤立噪点
    contours, hierarch = cv2.findContours(binary4_voted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        if area < pre_area_threshold and length < pre_length_threshold:
            cv2.drawContours(binary4_voted_img, [contours[i]], 0, 0, -1) 
    # 提取骨架增强连通性
    skeleton_class4 = binary4_voted_img.copy()
    skeleton_class4[skeleton_class4 == 4] = 1
    kernel_skeleton_class4_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))
    skeleton_class4 = cv2.morphologyEx(skeleton_class4, cv2.MORPH_CLOSE, kernel_skeleton_class4_1, iterations=1)
    skeleton_class4 = morphology.skeletonize(skeleton_class4)
    kernel_skeleton_class4_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))
    skeleton_class4 = cv2.dilate(skeleton_class4.astype(np.uint8), kernel_skeleton_class4_2, iterations=1)
    kernel_skeleton_class4_3 = cv2.getStructuringElement(cv2.MORPH_RECT,(11, 11))
    skeleton_class4 = cv2.erode(skeleton_class4.astype(np.uint8), kernel_skeleton_class4_3, iterations=1)
    binary4_voted_img[skeleton_class4 == 1] = 4

    
    binary3_voted_img = np.array(Image.open(os.path.join(temp_results_path, binary3_path[0] + '_' +  'binary_test%s_votedBy20tta' % suffix, img_name)))
    binary3_voted_img += np.array(Image.open(os.path.join(temp_results_path, binary3_path[1] + '_' +  'binary_test%s_votedBy30tta' % suffix, img_name)))
    binary3_voted_img += np.array(Image.open(os.path.join(temp_results_path, binary3_path[2] + '_' +  'binary_test%s_votedBy30tta' % suffix, img_name)))
    binary3_voted_img += np.array(Image.open(os.path.join(temp_results_path, binary3_path[3] + '_' +  'binary_test%s_votedBy30tta' % suffix, img_name)))
    binary3_voted_img += np.array(Image.open(os.path.join(temp_results_path, binary3_path[4] + '_' +  'binary_test%s_votedBy30tta' % suffix, img_name)))
    # 票数多于0票的设置为3
    bound = 3
    binary3_voted_img[binary3_voted_img <= bound] = 0
    binary3_voted_img[binary3_voted_img > bound] = 3
    # 处理binary3_voted_img
    kernel_close_class3 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    binary3_voted_img = cv2.morphologyEx(binary3_voted_img, cv2.MORPH_CLOSE, kernel_close_class3, iterations=1)
    binary3_voted_img = cv2.medianBlur(binary3_voted_img, 5)
    # 去孤立噪点
    contours, hierarch = cv2.findContours(binary3_voted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        if area < pre_area_threshold and length < pre_length_threshold:
            cv2.drawContours(binary3_voted_img, [contours[i]], 0, 0, -1) 
    # 提取骨架增强连通性
    skeleton_class3 = binary3_voted_img.copy()
    skeleton_class3[skeleton_class3 == 3] = 1
    kernel_skeleton_class3_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    skeleton_class3 = cv2.morphologyEx(skeleton_class3, cv2.MORPH_CLOSE, kernel_skeleton_class3_1, iterations=1)
    skeleton_class3 = morphology.skeletonize(skeleton_class3)
    kernel_skeleton_class3_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))
    skeleton_class3 = cv2.dilate(skeleton_class3.astype(np.uint8), kernel_skeleton_class3_2, iterations=1)
    kernel_skeleton_class3_3 = cv2.getStructuringElement(cv2.MORPH_RECT,(11, 11))
    skeleton_class3 = cv2.erode(skeleton_class3.astype(np.uint8), kernel_skeleton_class3_3, iterations=1)
    binary3_voted_img[skeleton_class3 == 1] = 3

    
    binary0_voted_img = np.array(Image.open(os.path.join(temp_results_path, binary0_path + '_' +  'binary_test%s_votedBy4tta' % suffix, img_name))) + 127
    # 票数多于3票的设置为0
    bound = 3
    binary0_voted_img[binary0_voted_img > bound + 127] = 0
    
    
    # 原始预测
    origin_mask = np.array(Image.open(
        os.path.join(temp_results_path, 'voted_by_multi_class_results' + suffix, img_name)))
    # 去除原图噪点
    kernel_open_allClass = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    origin_mask = cv2.morphologyEx(origin_mask, cv2.MORPH_OPEN, kernel_open_allClass, iterations=1)
    origin_mask[binary0_voted_img == 0] = 0
    # 检测是否包含
    check_replace_img = origin_mask.copy()
    check_replace_img = remove_class3And4_noise(check_replace_img, post_area_threshold=25, post_length_threshold=25)
    binary3_voted_img = remove_class0_noise(binary3_voted_img, post_area_threshold=25, post_length_threshold=25)
    binary4_voted_img = remove_class0_noise(binary4_voted_img, post_area_threshold=25, post_length_threshold=25)
    if np.all(binary3_voted_img[np.where(check_replace_img == 4)] == 3):
        binary3_voted_img[binary3_voted_img + check_replace_img == 7] = 0
    if np.all(binary4_voted_img[np.where(check_replace_img == 3)] == 4):
        binary4_voted_img[binary4_voted_img + check_replace_img == 7] = 0
    origin_mask[binary3_voted_img == 3] = 3
    origin_mask[binary4_voted_img == 4] = 4
    
    
    # 后处理，中值滤波去除孤立点
    origin_mask = cv2.medianBlur(origin_mask, 5)
    # 去除类3类4孤立团
    origin_mask = remove_class3And4_noise(origin_mask, post_area_threshold=25, post_length_threshold=25)
    
    return origin_mask


# 产生投票结果
def generate_voted_list_byOringinAndMultiBinary_addPostprocess(target_dir):
    # 计时
    start = time.time()
    file_num = len(test_list)
    
    for i, label_file in enumerate(test_list):
        label_name = label_file.split('/')[-1]
        label_name = label_name.split('.')[0] + '.png'
        
        target_pic_path = os.path.join(target_dir, label_name)
        voted_img = vote_img_byOringinAndMultiBinary_addPostprocess(label_name)
        cv2.imwrite(target_pic_path, voted_img)
            
        if (i + 1) % 100 == 0 or i + 1 == file_num:
            end = time.time() - start
            print('%d images have been done, cost %.2fs, left %.2fs' % (i + 1, end, file_num  / (i + 1) * end - end))
            
    return


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    temp_results_path = sys.argv[2]

    test_list = []
    with open(os.path.join(dataset_path, 'testB_list.txt')) as f:
        lines = f.readlines()
    for line in lines:
        test = line.rstrip()
        test_list.append(test)
        
    suffix = 'B'

    # class 4 binary_model_path
    binary4_path = [
        'sehrnet_cq_1122_s3',
        'seocrnet_lkc_1205_binary4_s1',
        'seocrnet_lkc_1205_binary4_s2',
        'seocrnet_lkc_1211_binary4_s1',
        'sehrnet_cq_1214_binary4_s1'
    ]

    # class 0 binary_model_path
    binary0_path = 'sehrnet_lkc_1122_binary0_s3'

    # class 3 binary_model_path
    binary3_path = [
        'sehrnet_lkc_1124_binary3_s2',
        'seocrnet_lkc_1205_binary3_s1',
        'seocrnet_lkc_1205_binary3_s2',
        'seocrnet_lkc_1211_binary3_s1',
        'sehrnet_cq_1214_binary3_s1'
    ]

    target_dir = os.path.join(temp_results_path, 'results' + suffix)

    check_dir(target_dir)
    generate_voted_list_byOringinAndMultiBinary_addPostprocess(target_dir)