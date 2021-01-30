# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import os
import pickle
from PIL import Image
import sys
import time
from tqdm import tqdm

from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90,
    Compose, OneOf, Rotate, Resize
)


# 水平翻转+随机90/180/270
h_flip_aug = Compose([
    HorizontalFlip(p=1.0),
    RandomRotate90(p=0.3)
])

# 垂直翻转+随机90/180/270
v_flip_aug = Compose([
    VerticalFlip(p=1.0),
    RandomRotate90(p=0.3)
])

# 放缩0.75+水平翻转+随机90/180/270
resize192_aug = Compose([
    Resize(height=192, width=192, p=1.0),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.3)
])

# 放缩0.875+水平翻转+随机90/180/270
resize224_aug = Compose([
    Resize(height=224, width=224, p=1.0),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.3)
])
    
# 放缩1.125+水平翻转+随机90/180/270
resize288_aug = Compose([
    Resize(height=288, width=288, p=1.0),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.3)
])  
    
# 放缩1.25+水平翻转+随机90/180/270
resize320_aug = Compose([
    Resize(height=320, width=320, p=1.0),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.3)
])

resize_aug_list = {
    'h_flip_aug': h_flip_aug,
    'v_flip_aug': v_flip_aug,
    'resize192_aug': resize192_aug,
    'resize224_aug': resize224_aug,
    'resize288_aug': resize288_aug,
    'resize320_aug': resize320_aug,
}

resize_aug_list_for_binary0 = {
    'h_flip_aug': h_flip_aug,
    'v_flip_aug': v_flip_aug,
    'resize192_aug': resize192_aug,
    'resize224_aug': resize224_aug,
    'resize288_aug': resize288_aug,
    'resize320_aug': resize320_aug,
}

resize_aug_list_for_binary3 = {
    'resize192_aug': resize192_aug,
    'resize224_aug': resize224_aug,
    'resize288_aug': resize288_aug,
    'resize320_aug': resize320_aug,
}    

resize_aug_list_for_binary4 = {
    'h_flip_aug': h_flip_aug,
    'v_flip_aug': v_flip_aug,
    'resize192_aug': resize192_aug,
    'resize224_aug': resize224_aug,
    'resize288_aug': resize288_aug,
    'resize320_aug': resize320_aug,
}


# 检查不同类别像素数量 
def check_class(img):
    class_list = [0, 1, 2, 3, 4, 5, 6, 255]
    img = np.array(img)
    res = []
    for c in class_list:
        count = sum(sum(img == c))
        res.append(count)
    return res


# 统计分析结果
def generate_analyse_results():
    train_path = os.path.join(dataset_path, 'train_res.data')
    val_path = os.path.join(dataset_path, 'val_res.data')
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print('analyse data...')
        total_num = 145980
        train_res = {}
        val_res = {}
        for i in tqdm(range(0, total_num + 1), ncols=120):
            img_name = 'T%06d' % i
            if i % 50 == 49:
                val_label = Image.open(os.path.join(dataset_path, 'lab_train/%s.png' %img_name))
                val_res[img_name] = check_class(val_label)
            else:
                train_label = Image.open(os.path.join(dataset_path, 'lab_train/%s.png' %img_name))
                train_res[img_name] = check_class(train_label)
        with open(train_path, 'wb') as f:
            pickle.dump(train_res, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_res, f)


# 生成只包含类3/4/5的数据集
def generate_only_class0or3or4or5_dataset():
    total = 65536
    threhold = 0.01
    choose_class0_path = os.path.join(dataset_path, 'train_list_choose0.txt')
    choose_class3_path = os.path.join(dataset_path, 'train_list_choose3.txt')
    choose_class4_path = os.path.join(dataset_path, 'train_list_choose4.txt')
    choose_class5_path = os.path.join(dataset_path, 'train_list_choose5.txt')
    choose_class4And5_path = os.path.join(dataset_path, 'train_list_choose4And5.txt')
    without_class0_path = os.path.join(dataset_path, 'train_list_without0.txt')
    without_class3_path = os.path.join(dataset_path, 'train_list_without3.txt')
    without_class4_path = os.path.join(dataset_path, 'train_list_without4.txt')
    without_class5_path = os.path.join(dataset_path, 'train_list_without5.txt')
    without_class4And5_path = os.path.join(dataset_path, 'train_list_without4And5.txt')
    if not os.path.exists(choose_class0_path) or not os.path.exists(without_class0_path):
        print('generate only class0 data...')
        num = 0
        count = 0
        train = 'img_train/{}.jpg'
        lab = 'lab_train/{}.png'
        with open(choose_class0_path, 'w') as f:
            with open(without_class0_path, 'w') as w_f:
                for k, v in train_res.items():
                    if v[num] and v[num] / total > threhold:
                        count += 1
                        line = train.format(k) + ' ' + lab.format(k) + '\n'
                        f.write(line)
                    else:
                        line = train.format(k) + ' ' + lab.format(k) + '\n'
                        w_f.write(line)
        print('Thredhold: ', threhold)
        print('Get class0: ', count)
    if not os.path.exists(choose_class3_path) or not os.path.exists(without_class3_path):
        print('generate only class3 data...')
        num = 3
        count = 0
        train = 'img_train/{}.jpg'
        lab = 'lab_train/{}.png'
        with open(choose_class3_path, 'w') as f:
            with open(without_class3_path, 'w') as w_f:
                for k, v in train_res.items():
                    if v[num] and v[num] / total > threhold:
                        count += 1
                        line = train.format(k) + ' ' + lab.format(k) + '\n'
                        f.write(line)
                    else:
                        line = train.format(k) + ' ' + lab.format(k) + '\n'
                        w_f.write(line)
        print('Thredhold: ', threhold)
        print('Get class3: ', count)
    if not os.path.exists(choose_class4_path) or not os.path.exists(without_class4_path):
        print('generate only class4 data...')
        num = 4
        count = 0
        train = 'img_train/{}.jpg'
        lab = 'lab_train/{}.png'
        with open(choose_class4_path, 'w') as f:
            with open(without_class4_path, 'w') as w_f:
                for k, v in train_res.items():
                    if v[num] and v[num] / total > threhold:
                        count += 1
                        line = train.format(k) + ' ' + lab.format(k) + '\n'
                        f.write(line)
                    else:
                        line = train.format(k) + ' ' + lab.format(k) + '\n'
                        w_f.write(line)
        print('Thredhold: ', threhold)
        print('Get class4: ', count)
    if not os.path.exists(choose_class5_path) or not os.path.exists(without_class5_path):
        print('generate only class5 data...')
        num = 5
        count = 0
        train = 'img_train/{}.jpg'
        lab = 'lab_train/{}.png'
        with open(choose_class5_path, 'w') as f:
            with open(without_class5_path, 'w') as w_f:
                for k, v in train_res.items():
                    if v[num] and v[num] / total > threhold:
                        count += 1
                        line = train.format(k) + ' ' + lab.format(k) + '\n'
                        f.write(line)
                    else:
                        line = train.format(k) + ' ' + lab.format(k) + '\n'
                        w_f.write(line)
        print('Thredhold: ', threhold)
        print('Get class5: ', count)
    if not os.path.exists(choose_class4And5_path) or not os.path.exists(without_class4And5_path):
        print('generate only class4And5 data...')
        num1 = 4
        num2 = 5
        count = 0
        train = 'img_train/{}.jpg'
        lab = 'lab_train/{}.png'
        with open(choose_class4And5_path, 'w') as f:
            with open(without_class4And5_path, 'w') as w_f:
                for k, v in train_res.items():
                    if (v[num1] and v[num1] / total > threhold) or (v[num2] and v[num2] / total > threhold):
                        count += 1
                        line = train.format(k) + ' ' + lab.format(k) + '\n'
                        f.write(line)
                    else:
                        line = train.format(k) + ' ' + lab.format(k) + '\n'
                        w_f.write(line)
        print('Thredhold: ', threhold)
        print('Get class4And5: ', count)

    return choose_class0_path, choose_class3_path, choose_class4_path, choose_class5_path, choose_class4And5_path


# 检查目录是否存在
def check_dir(target_dir):
    if not os.path.isdir(target_dir):
        print('Create file directory => %s' % target_dir)
        os.makedirs(target_dir)


# 生成类4类5数据的增强数据集
def generate_class4And5_aug_dataset(source_list, path, train_target_dir, lab_target_dir, aug_list):
    output_list = source_list.split('.txt')[0] + '_aug' + '.txt'
    if not os.path.exists(output_list):
        print('generate class4And5 augumented data...')
        with open(source_list, 'r') as f:
            lines = f.readlines()
        train_target_dir1 = os.path.join(path, train_target_dir)
        lab_target_dir1 = os.path.join(path, lab_target_dir)
        check_dir(train_target_dir1)
        check_dir(lab_target_dir1)
        with open(output_list, 'w') as f:
            for k, aug in aug_list.items():
                train_target_dir2 = os.path.join(path, train_target_dir, k)
                lab_target_dir2 = os.path.join(path, lab_target_dir, k)
                check_dir(train_target_dir2)
                check_dir(lab_target_dir2)
                train = os.path.join(train_target_dir, k) + '/{}'
                lab = os.path.join(lab_target_dir, k) + '/{}'
                for i, line in tqdm(enumerate(lines), ncols=120, total=len(lines)):
                    train_img_file, lab_img_file = line.rstrip().split(' ')
                    train_img_name = train_img_file.split('/')[-1]
                    lab_img_name = lab_img_file.split('/')[-1]
                    train_img = np.array(Image.open(os.path.join(path, train_img_file)))
                    lab_img = np.array(Image.open(os.path.join(path, lab_img_file)))
                    aug_img = aug(image=train_img, mask=lab_img)
                    train_img, lab_img = aug_img['image'], aug_img['mask']
                    # cv2默认写入为BGR
                    train_img_pic_path = os.path.join(train_target_dir2, train_img_name)
                    lab_img_pic_path = os.path.join(lab_target_dir2, lab_img_name)
                    cv2.imwrite(train_img_pic_path, train_img[:, :, ::-1])
                    cv2.imwrite(lab_img_pic_path, lab_img)
                    line = train.format(train_img_name) + ' ' + lab.format(lab_img_name) + '\n'
                    f.write(line)
    return output_list


# 生成多分类数据增强数据集
def generate_multi_class_aug_dataset(choose_class4_path, choose_class5_path, choose_class4And5_path, class4And5_aug_path):
    only_class4_path = os.path.join(dataset_path, 'train_list_only4.txt')
    only_class5_path = os.path.join(dataset_path, 'train_list_only5.txt')
    add4And5_path = os.path.join(dataset_path, 'train_list_add4And5.txt')
    if not os.path.exists(only_class4_path) or not os.path.exists(only_class5_path) or not os.path.exists(add4And5_path):
        print('generate multi-class augumented data...')
        with open(choose_class4_path, 'r') as f:
            choose_class4_lines = f.readlines()
        with open(choose_class5_path, 'r') as f:
            choose_class5_lines = f.readlines()
        
        with open(class4And5_aug_path, 'r') as class4And5_aug_f:
            class4And5_aug_lines = class4And5_aug_f.readlines()
            # 类4
            with open(only_class4_path, 'w') as class4_f:
                class4_set = set()
                for class4_line in choose_class4_lines:
                    img_name = class4_line.split('.jpg')[0]
                    img_name = img_name.split('/')[-1]
                    class4_set.add(img_name)
                    class4_f.write(class4_line)
                # 类5
                with open(only_class5_path, 'w') as class5_f:
                    class5_set = set()
                    for class5_line in choose_class5_lines:
                        img_name = class5_line.split('.jpg')[0]
                        img_name = img_name.split('/')[-1]
                        class5_set.add(img_name)
                        class5_f.write(class5_line)
                    # 增强
                    for aug_line in class4And5_aug_lines:
                        img_name = aug_line.split('.jpg')[0]
                        img_name = img_name.split('/')[-1]
                        if img_name in class4_set:
                            class4_f.write(aug_line)
                        if img_name in class5_set:
                            class5_f.write(aug_line)
        
        total_path = os.path.join(dataset_path, 'train_list.txt')
        os.system('cat %s %s %s > %s' % (total_path, choose_class4And5_path, class4And5_aug_path, add4And5_path))
        

# 获取图片列表，用于生成二分类数据
def get_pic_list(source_dir):
    dirs = os.listdir(source_dir)
    pic_list = []
    for file in dirs:
        pic_list.append(file)
    return pic_list


# 生成二分类数据集
def generate_binary_class_dataset(source_dir, target_class_dir, class_label, choose_path):
    positive_train_path = os.path.join(dataset_path, 'class%d_positive_train.txt' % class_label)
    negetive_train_path = os.path.join(dataset_path, 'class%d_negetive_train.txt' % class_label)
    val_path = os.path.join(dataset_path, 'class%d_val.txt' % class_label)

    if not os.path.exists(positive_train_path) or not os.path.exists(negetive_train_path) or not os.path.exists(val_path):
        print("Write train/val label file for class %d" % class_label)

        # 创建二分类图
        total_edit_num = 0
        pic_list = get_pic_list(source_dir)
        for i, pic in tqdm(enumerate(pic_list), ncols=120, total=len(pic_list)):
            source_pic_path = os.path.join(source_dir, pic)
            target_pic_path = os.path.join(target_class_dir, pic)
            pic_name = pic.split('.')[0]
            if label_res[pic_name][class_label] > 0:
                total_edit_num += 1
                origin_pic = np.array(Image.open(source_pic_path))
                # 先转为不存在的类别127
                origin_pic[(origin_pic != class_label) == (origin_pic != 255)] = 127
                origin_pic[origin_pic == class_label] = 1
                origin_pic[origin_pic == 127] = 0
                cv2.imwrite(target_pic_path, origin_pic)
        print('Class: %d, Total edited: %d' % (class_label, total_edit_num))

        # 写入文件
        positive_train_file = open(positive_train_path, 'w')
        negetive_train_file = open(negetive_train_path, 'w')
        val_file = open(val_path, 'w')
        train = 'img_train/{}.jpg'
        lab = 'binary_lab/class%d/' % class_label + '{}.png'
        with open(choose_path, 'r') as f:
            lines = f.readlines()
            choose_set = set()
            for line in lines:
                img_name = line.split('.jpg')[0]
                img_name = img_name.split('/')[-1]
                choose_set.add(img_name)
        for label in train_label:
            line = train.format(label) + ' ' + lab.format(label) + '\n'
            if label in choose_set:
                positive_train_file.write(line)
            else:
                negetive_train_file.write(line)
        for label in val_label:
            line = train.format(label) + ' ' + lab.format(label) + '\n'
            val_file.write(line)
        positive_train_file.close()
        negetive_train_file.close()
        val_file.close()

    return positive_train_path


# 生成二分类增强数据
def generate_binary_class_aug_dataset(source_list, path, train_target_dir, lab_target_dir, aug_list):
    output_list = source_list.split('.txt')[0] + '_aug' + '.txt'
    if not os.path.exists(output_list):
        print('generate binary-class augumented data...')
        with open(source_list, 'r') as f:
            lines = f.readlines()
        train_target_dir1 = os.path.join(path, train_target_dir)
        lab_target_dir1 = os.path.join(path, lab_target_dir)
        check_dir(train_target_dir1)
        check_dir(lab_target_dir1)
        with open(output_list, 'w') as f:
            for k, aug in aug_list.items():
                train_target_dir2 = os.path.join(path, train_target_dir, k)
                lab_target_dir2 = os.path.join(path, lab_target_dir, k)
                check_dir(train_target_dir2)
                check_dir(lab_target_dir2)
                train = os.path.join(train_target_dir, k) + '/{}'
                lab = os.path.join(lab_target_dir, k) + '/{}'
                for i, line in tqdm(enumerate(lines), ncols=120, total=len(lines)):
                    train_img_file, lab_img_file = line.rstrip().split(' ')
                    train_img_name = train_img_file.split('/')[-1]
                    lab_img_name = lab_img_file.split('/')[-1]
                    train_img = np.array(Image.open(os.path.join(path, train_img_file)))
                    lab_img = np.array(Image.open(os.path.join(path, lab_img_file)))
                    aug_img = aug(image=train_img, mask=lab_img)
                    train_img, lab_img = aug_img['image'], aug_img['mask']
                    # cv2默认写入为BGR
                    train_img_pic_path = os.path.join(train_target_dir2, train_img_name)
                    lab_img_pic_path = os.path.join(lab_target_dir2, lab_img_name)
                    cv2.imwrite(train_img_pic_path, train_img[:, :, ::-1])
                    cv2.imwrite(lab_img_pic_path, lab_img)
                    line = train.format(train_img_name) + ' ' + lab.format(lab_img_name) + '\n'
                    f.write(line)
    return output_list



if __name__ == '__main__':
    dataset_path = sys.argv[1]

    # 统计数据集各类面积占比
    generate_analyse_results()
    label_res = {}
    train_path = os.path.join(dataset_path, 'train_res.data')
    val_path = os.path.join(dataset_path, 'val_res.data')
    with open(train_path, 'rb') as f:
        train_res = pickle.load(f)
        train_label = train_res.keys()
        label_res.update(train_res)
    with open(val_path, 'rb') as f:
        val_res = pickle.load(f)
        val_label = val_res.keys()
        label_res.update(val_res)

    # 筛选阈值大于0.01的各类图片
    choose_class0_path, choose_class3_path, choose_class4_path, choose_class5_path, choose_class4And5_path = generate_only_class0or3or4or5_dataset()

    # 生成全分类数据
    class4And5_aug_path = generate_class4And5_aug_dataset(choose_class4And5_path, dataset_path,
                          'class4And5_img_train', 'class4And5_lab_train', resize_aug_list)
    generate_multi_class_aug_dataset(choose_class4_path, choose_class5_path, 
                                     choose_class4And5_path, class4And5_aug_path)

    # 生成二分类数据
    source_dir = os.path.join(dataset_path, 'lab_train')
    target_dir = os.path.join(dataset_path, 'binary_lab')
    check_dir(target_dir)

    for class_label, choose_path in zip([0, 3, 4], 
        [choose_class0_path, choose_class3_path, choose_class4_path]):
        target_class_dir = os.path.join(target_dir, 'class%d' % class_label)
        check_dir(target_class_dir)
        positive_train_path = generate_binary_class_dataset(source_dir, target_class_dir, class_label, choose_path)
        if class_label == 0:
            aug_list = resize_aug_list_for_binary0
        elif class_label == 3:
            aug_list = resize_aug_list_for_binary3
        elif class_label == 4:
            aug_list = resize_aug_list_for_binary4
        generate_binary_class_aug_dataset(positive_train_path, dataset_path,
                          'binary_lab/class%d_aug_img_train' % class_label, 
                          'binary_lab/class%d_aug_lab_train' % class_label, aug_list)