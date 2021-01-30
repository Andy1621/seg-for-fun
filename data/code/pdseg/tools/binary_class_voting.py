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

# 检查目录是否存在
def check_dir(target_dir):
    if not os.path.isdir(target_dir):
        print('Create file directory => %s' % target_dir)
        os.makedirs(target_dir)


# 生成二分类投票数据，直接投票也可，不过多次投票每次遍历速度较慢
def generate_voted_binary_img(binary_path, module_num=1):
    # 不同的TTA模式，需要先测试生成相应结果
    tta_modules = [
        [
            'origin_results', 'invert_h_flip_results', 'invert_v_flip_results', 'invert_r_180_results'
        ],
        [
            'origin_results', 'invert_h_flip_results', 'invert_v_flip_results', 'invert_r_180_results',
            'invert_resize_288_results', 'invert_resize_288_h_flip_results', 
            'invert_resize_288_v_flip_results', 'invert_resize_288_r_180_results',
            'invert_resize_320_results', 'invert_resize_320_h_flip_results', 
            'invert_resize_320_v_flip_results', 'invert_resize_320_r_180_results',
            'invert_resize_352_results', 'invert_resize_352_h_flip_results', 
            'invert_resize_352_v_flip_results', 'invert_resize_352_r_180_results',
            'invert_resize_384_results', 'invert_resize_384_h_flip_results', 
            'invert_resize_384_v_flip_results', 'invert_resize_384_r_180_results',
        ],
        [
            'origin_results', 'invert_h_flip_results', 'invert_v_flip_results', 'invert_r_180_results',
            'invert_resize_288_results', 'invert_resize_288_h_flip_results', 
            'invert_resize_288_v_flip_results', 'invert_resize_288_r_180_results',
            'invert_resize_320_results', 'invert_resize_320_h_flip_results', 
            'invert_resize_320_v_flip_results', 'invert_resize_320_r_180_results',
            'invert_resize_352_results', 'invert_resize_352_h_flip_results', 
            'invert_resize_352_v_flip_results', 'invert_resize_352_r_180_results',
            'invert_resize_384_results', 'invert_resize_384_h_flip_results', 
            'invert_resize_384_v_flip_results', 'invert_resize_384_r_180_results',
            'invert_r_90_results', 'invert_r_270_results',
            'invert_resize_288_r_90_results', 'invert_resize_288_r_270_results',
            'invert_resize_320_r_90_results', 'invert_resize_320_r_270_results',
            'invert_resize_352_r_90_results', 'invert_resize_352_r_270_results',
            'invert_resize_384_r_90_results', 'invert_resize_384_r_270_results',  
        ]
    ]
   
    # 计时
    start = time.time()
    file_num = len(test_list)
    binary_path_prefix = binary_path.split('/')[-1]
    target_dir = os.path.join(temp_results_path, binary_path_prefix + '_binary_test%s_votedBy%dtta' % (suffix, len(tta_modules[module_num])))
    check_dir(target_dir)
    
    # 读入每张图片，计算每个位置出现为1的图片数
    for i, label_file in enumerate(test_list):
        label_name = label_file.split('/')[-1]
        label_name = label_name.split('.')[0] + '.png'
        
        binary_mask = []
        for name in tta_modules[module_num]:
            binary_mask.append(np.array(Image.open(os.path.join(binary_path, name + suffix, label_name))))
        
        binary_voted_img = np.zeros(binary_mask[0].shape)
        for mask in binary_mask:
            binary_voted_img[mask == 1] += 1
        
        target_pic_path = os.path.join(target_dir, label_name)
        cv2.imwrite(target_pic_path, binary_voted_img)
            
        if (i + 1) % 2000 == 0 or i + 1 == file_num:
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
        '../user_data/saved_log/sehrnet_cq_1122_s3',
        '../user_data/saved_log/seocrnet_lkc_1205_binary4_s1',
        '../user_data/saved_log/seocrnet_lkc_1205_binary4_s2',
        '../user_data/saved_log/seocrnet_lkc_1211_binary4_s1',
        '../user_data/saved_log/sehrnet_cq_1214_binary4_s1'
    ]

    # class 0 binary_model_path
    binary0_path = '../user_data/saved_log/sehrnet_lkc_1122_binary0_s3'

    # class 3 binary_model_path
    binary3_path = [
        '../user_data/saved_log/sehrnet_lkc_1124_binary3_s2',
        '../user_data/saved_log/seocrnet_lkc_1205_binary3_s1',
        '../user_data/saved_log/seocrnet_lkc_1205_binary3_s2',
        '../user_data/saved_log/seocrnet_lkc_1211_binary3_s1',
        '../user_data/saved_log/sehrnet_cq_1214_binary3_s1'
    ]

    # 类0,4TTA
    generate_voted_binary_img(binary0_path, module_num=0)
    # 旧类3类4, 20TTA
    generate_voted_binary_img(binary3_path[0], module_num=1)
    generate_voted_binary_img(binary4_path[0], module_num=1)
    # 新类3类4, 30TTA
    generate_voted_binary_img(binary3_path[1], module_num=2)
    generate_voted_binary_img(binary4_path[1], module_num=2)

    generate_voted_binary_img(binary3_path[2], module_num=2)
    generate_voted_binary_img(binary4_path[2], module_num=2)

    generate_voted_binary_img(binary3_path[3], module_num=2)
    generate_voted_binary_img(binary4_path[3], module_num=2)

    generate_voted_binary_img(binary3_path[4], module_num=2)
    generate_voted_binary_img(binary4_path[4], module_num=2)