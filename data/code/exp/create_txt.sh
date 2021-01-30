#!/bin/bash

original_dataset_path="$1"
target_dataset_path="$2"
current_path=`pwd`

# 设置绝对路径
if [ -n $original_dataset_path ]; then
    cd $original_dataset_path
    original_dataset_path=`pwd`
fi
cd $current_path
if  [ ! -d $target_dataset_path ]; then
    echo "create dataset directory..."
    mkdir $target_dataset_path
fi
cd $target_dataset_path
target_dataset_path=`pwd`

# 开始生成数据
cd $target_dataset_path
if [ ! -f $target_dataset_path/train_data/img_train.zip ] || [ ! -f $target_dataset_path/train_data/lab_train.zip ]; then
    echo `pwd`
    cd $original_dataset_path
    if [ -f train_data.zip ]; then
        echo "unzip train_data.zip..."
        unzip train_data.zip -d $target_dataset_path > /dev/null 2>&1
    fi
fi

cd $target_dataset_path/train_data
if [ -f img_train.zip ] && [ -f lab_train.zip ]; then
    if [ -f train_list.txt ]; then
        echo "File train_list.txt has existed."
    elif [ -f val_list.txt ]; then
        echo "File val_list.txt has existed."
    else 
        echo "unzip img_train.zip..."
        unzip img_train.zip -d $target_dataset_path/train_data > /dev/null 2>&1
        echo "unzip lab_train.zip..."
        unzip lab_train.zip -d $target_dataset_path/train_data > /dev/null 2>&1

        find img_train -type f | sort > train.ccf.tmp
        find lab_train -type f | sort > train.lab.ccf.tmp
        paste -d " " train.ccf.tmp train.lab.ccf.tmp > all.ccf.tmp
        
        awk '{if (NR % 50 != 0) print $0}' all.ccf.tmp > train_list.txt
        awk '{if (NR % 50 == 0) print $0}' all.ccf.tmp > val_list.txt
    
        rm *.ccf.tmp
        echo "Create train_list.txt and val_list.txt."
    fi
fi

cd $original_dataset_path
if [ -f img_testA.zip ]; then
    if [ -f $target_dataset_path/testA_list.txt ]; then
        echo "File testA_list.txt has existed."
    else
        echo "unzip img_testA.zip..."
        unzip img_testA.zip -d $target_dataset_path > /dev/null 2>&1

        cd $target_dataset_path
        find img_testA -type f | sort > testA_list.txt
        echo "Create testA_list.txt."
    fi
fi

cd $original_dataset_path
if [ -f img_testB.zip ]; then
    if [ -f $target_dataset_path/testB_list.txt ]; then
        echo "File testB_list.txt has existed."
    else
        echo "unzip img_testB.zip..."
        unzip img_testB.zip -d $target_dataset_path > /dev/null 2>&1

        cd $target_dataset_path
        find img_testB -type f | sort > testB_list.txt
        echo "Create testB_list.txt."
    fi
fi

