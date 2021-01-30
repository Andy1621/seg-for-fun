# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import os
from PIL import Image
import sys
import time

from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate,
     Compose, Resize
)


def check_dir(target_dir):
    if not os.path.isdir(target_dir):
        print('Create file directory => %s' % target_dir)
        os.makedirs(target_dir)


def generate_invert_aug_list(target_dir, aug_name):
    # 计时
    start = time.time()
    file_num = len(test_list)
    
    source_dir = os.path.join(path2, aug_name + results_file)
    img_list = os.listdir(source_dir)
    
    for i, img_file in enumerate(img_list):
        origin_train_img = np.array(Image.open(os.path.join(source_dir, img_file)))
        target_pic_path = os.path.join(target_dir, img_file)
        aug_img = invert_aug_list[aug_name](image=origin_train_img)['image']
        # cv2默认写入为BGR
        cv2.imwrite(target_pic_path, aug_img)
            
        if (i + 1) % 500 == 0 or i + 1 == file_num:
            end = time.time() - start
            print('%d images have been done, cost %.2fs, left %.2fs' % (i + 1, end, file_num  / (i + 1) * end - end))
            
    return


def generate_invert_resize_list(target_dir, aug_name):
    # 计时
    start = time.time()
    file_num = len(test_list)
    
    source_dir = os.path.join(path2, aug_name + results_file)
    img_list = os.listdir(source_dir)
    
    for i, img_file in enumerate(img_list):
        origin_train_img = np.array(Image.open(os.path.join(source_dir, img_file)))
        target_pic_path = os.path.join(target_dir, img_file)
        aug_img = invert_resize_aug_list[aug_name](image=origin_train_img)['image']
        # cv2默认写入为BGR
        cv2.imwrite(target_pic_path, aug_img)
            
        if (i + 1) % 500 == 0 or i + 1 == file_num:
            end = time.time() - start
            print('%d images have been done, cost %.2fs, left %.2fs' % (i + 1, end, file_num  / (i + 1) * end - end))
            
    return


def generate_invert_resize_aug_list(target_dir, resize_aug_name, aug_name):
    # 计时
    start = time.time()
    file_num = len(test_list)
    
    source_dir = os.path.join(path2, aug_name + results_file)
    img_list = os.listdir(source_dir)
    
    for i, img_file in enumerate(img_list):
        origin_train_img = np.array(Image.open(os.path.join(source_dir, img_file)))
        target_pic_path = os.path.join(target_dir, img_file)
        aug = Compose([invert_resize_aug_list[resize_aug_name], invert_aug_list[aug_name]])
        aug_img = aug(image=origin_train_img)['image']
        # cv2默认写入为BGR
        cv2.imwrite(target_pic_path, aug_img)
            
        if (i + 1) % 500 == 0 or i + 1 == file_num:
            end = time.time() - start
            print('%d images have been done, cost %.2fs, left %.2fs' % (i + 1, end, file_num  / (i + 1) * end - end))
            
    return


if __name__ == '__main__':
    dataset_path = sys.argv[1]

    test_list = []
    with open(os.path.join(dataset_path, 'testB_list.txt')) as f:
        lines = f.readlines()
    for line in lines:
        test = line.rstrip()
        test_list.append(test)

    h_flip = HorizontalFlip(p=1.0)
    v_flip = VerticalFlip(p=1.0)
    r_90 = Rotate(limit=(90, 90), p=1.0)
    r_180 = Rotate(limit=(180, 180), p=1.0)
    resize_256 = Resize(height=256, width=256, p=1.0)
    resize_288 = Resize(height=288, width=288, p=1.0)
    resize_320 = Resize(height=320, width=320, p=1.0)

    results_file = '_resultsB'

    # 4TTA
    invert_aug_list = {
        'h_flip': h_flip,
        'v_flip': v_flip,
        'r_180': r_180,
    }
    path2 = '../user_data/saved_log/sehrnet_lkc_1122_binary0_s3'
    print('Now:', path2)
        
    for k, v in invert_aug_list.items():
        target_dir = os.path.join(path2, 'invert_' + k + results_file)
        print('Target:', target_dir)
        check_dir(target_dir)
        generate_invert_aug_list(target_dir, aug_name=k)

    # 20TTA
    invert_aug_list = {
        'h_flip': h_flip,
        'v_flip': v_flip,
        'r_180': r_180,
    }

    invert_resize_aug_list = {
        'resize_288': resize_256,
        'resize_320': resize_256,
        'resize_352': resize_256,
        'resize_384': resize_256,
    }

    path2_list = [
        '../user_data/saved_log/sehrnet_cq_1122_s3',
        '../user_data/saved_log/sehrnet_lkc_1124_binary3_s2',
    ]

    for p in path2_list:
        path2 = p
        
        print('Now:', path2)
        
        for k, v in invert_aug_list.items():
            target_dir = os.path.join(path2, 'invert_' + k + results_file)
            print('Target:', target_dir)
            check_dir(target_dir)
            generate_invert_aug_list(target_dir, aug_name=k)

        for k, v in invert_resize_aug_list.items():
            target_dir = os.path.join(path2, 'invert_' + k + results_file)
            print('Target:', target_dir)
            check_dir(target_dir)
            generate_invert_resize_list(target_dir, aug_name=k)

        for k1, v1 in invert_resize_aug_list.items():
            for k2, v2 in invert_aug_list.items():
                v = Compose([v1, v2])
                target_dir = os.path.join(path2, 'invert_' + k1 + '_' + k2 + results_file)
                print('Target:', target_dir)
                check_dir(target_dir)
                generate_invert_resize_aug_list(target_dir, resize_aug_name=k1, aug_name=k2)

    # 30TTA

    invert_aug_list = {
        'h_flip': h_flip,
        'v_flip': v_flip,
        'r_180': r_180,
        'r_90': r_270,
        'r_270': r_90,
    }

    invert_resize_aug_list = {
        'resize_288': resize_256,
        'resize_320': resize_256,
        'resize_352': resize_256,
        'resize_384': resize_256,
    }

    path2_list = [
        '../user_data/saved_log/seocrnet_lkc_1205_binary4_s1',
        '../user_data/saved_log/seocrnet_lkc_1205_binary4_s2',
        '../user_data/saved_log/seocrnet_lkc_1211_binary4_s1',
        '../user_data/saved_log/sehrnet_cq_1214_binary4_s1',
        '../user_data/saved_log/seocrnet_lkc_1205_binary3_s1',
        '../user_data/saved_log/seocrnet_lkc_1205_binary3_s2',
        '../user_data/saved_log/seocrnet_lkc_1211_binary3_s1',
        '../user_data/saved_log/sehrnet_cq_1214_binary3_s1',
    ]

    for p in path2_list:
        path2 = p
        
        print('Now:', path2)
        
        for k, v in invert_aug_list.items():
            target_dir = os.path.join(path2, 'invert_' + k + results_file)
            print('Target:', target_dir)
            check_dir(target_dir)
            generate_invert_aug_list(target_dir, aug_name=k)

        for k, v in invert_resize_aug_list.items():
            target_dir = os.path.join(path2, 'invert_' + k + results_file)
            print('Target:', target_dir)
            check_dir(target_dir)
            generate_invert_resize_list(target_dir, aug_name=k)

        for k1, v1 in invert_resize_aug_list.items():
            for k2, v2 in invert_aug_list.items():
                v = Compose([v1, v2])
                target_dir = os.path.join(path2, 'invert_' + k1 + '_' + k2 + results_file)
                print('Target:', target_dir)
                check_dir(target_dir)
                generate_invert_resize_aug_list(target_dir, resize_aug_name=k1, aug_name=k2)