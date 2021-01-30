# -*- coding: UTF-8 -*-
import os
import cv2
import sys
import numpy as np
from PIL import Image
import time
import pickle
import shutil
import random

from p_tqdm import p_imap
from functools import partial


# 检查目录是否存在
def check_dir(target_dir):
    if not os.path.isdir(target_dir):
        print('Create file directory => %s' % target_dir)
        os.makedirs(target_dir)


# 多线程加权
def mp_vote_img_weight(img_name, target_dir, weight=None):
    img_name = img_name.split('/')[-1]
    img_name = img_name.split('.')[0] + '.png'
    img_list = []
    
    for path in model_path:
        for name in tta_name:
            img_list.append(np.array(Image.open(os.path.join(path, name, img_name))))
    origin_img = img_list[0]
    row, column = origin_img.shape
    voted_img = np.zeros((row, column))
    if not weight:
        weight = [1] * len(img_list)
    # 前两个model增大类4/5的权重
    weight2 = np.array([[1] * 7] * (len(tta_name) * len(model_path)))
    weight2[:len(tta_name) * 5, 4] = 3
    weight2[:len(tta_name) * 5, 5] = 2
    weight2[len(tta_name) * (-2): len(tta_name) * (-1), :] = 0
    weight2[len(tta_name) * (-2): len(tta_name) * (-1), 4] = 3
    weight2[len(tta_name) * (-1): , :] = 0
    weight2[len(tta_name) * (-1): , 5] = 2
    for r in range(row):
        for c in range(column):
            vote = []
            for i, img in enumerate(img_list):
                class_num = img[r, c]
                vote = vote + [class_num] * (weight[i] * weight2[i][class_num])
            voted_img[r, c] =  max(vote, key=vote.count)  
            
    target_pic_path = os.path.join(target_dir, img_name)
    cv2.imwrite(target_pic_path, voted_img)        
    
    return img_name


# 多进程加权
def mp_mugenerate_voted_list(target_dir, weight=None):
    # 计时
    start = time.time()
    file_num = len(test_list)
    count = 0
    
    iterator = p_imap(partial(mp_vote_img_weight, target_dir=target_dir, weight=weight), test_list, position=0, leave=True, ncols=100, dynamic_ncols=False)
    
    for img_name in iterator:
        pass
            
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

    model_path = [
    '../user_data/saved_log/se_ocrnet_cq_1204_1',
    '../user_data/saved_log/se_ocrnet_lkc_1130_class4And5',
    '../user_data/saved_log/se_ocrnet_lkc_1126_class4And5',
    '../user_data/saved_log/hrnet_cq_1120_1',
    '../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2',
    '../user_data/saved_log/hrnet_lkc_1112_1',
    '../user_data/saved_log/hrnet_cq_1112_1',
    '../user_data/saved_log/seocrnet_cq_1126_class4_s2',
    '../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2'
    ]
        
    tta_name = [
        'origin_resultsB', 'invert_h_flip_resultsB', 'invert_v_flip_resultsB', 'invert_r_180_resultsB',
        'invert_resize_288_resultsB', 'invert_resize_288_h_flip_resultsB', 
        'invert_resize_288_v_flip_resultsB', 'invert_resize_288_r_180_resultsB',
        'invert_resize_320_resultsB', 'invert_resize_320_h_flip_resultsB', 
        'invert_resize_320_v_flip_resultsB', 'invert_resize_320_r_180_resultsB',
    ]

    target_dir = os.path.join(temp_results_path, 'voted_by_multi_class_resultsB')
    check_dir(target_dir)
    mp_mugenerate_voted_list(target_dir, weight=None)