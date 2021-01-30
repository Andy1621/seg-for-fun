########################################################################
########################################################################
########################################################################
# seocrnet_lkc_1205_binary3_s1
# TTA test results
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/origin_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt
# r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_90_img_testB_list.txt
# r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_270_img_testB_list.txt

########################################################################################################
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_288_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_288_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_288_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_288_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt
# resize_288_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_288_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_90_img_testB_list.txt
# resize_288_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_288_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_270_img_testB_list.txt

########################################################################################################
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_320_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_320_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_320_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_320_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt
# resize_320_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_320_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_90_img_testB_list.txt
# resize_320_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_320_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_270_img_testB_list.txt

########################################################################################################
# resize_352
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_352_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_img_testB_list.txt
# resize_352_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_352_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_h_flip_img_testB_list.txt
# resize_352_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_352_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_v_flip_img_testB_list.txt
# resize_352_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_352_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_180_img_testB_list.txt
# resize_352_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_352_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_90_img_testB_list.txt
# resize_352_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_352_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_270_img_testB_list.txt

########################################################################################################
# resize_384
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_384_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_img_testB_list.txt
# resize_384_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_384_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_h_flip_img_testB_list.txt
# resize_384_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_384_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_v_flip_img_testB_list.txt
# resize_384_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_384_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_180_img_testB_list.txt
# resize_384_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_384_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_90_img_testB_list.txt
# resize_384_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/resize_384_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_270_img_testB_list.txt

########################################################################
########################################################################
########################################################################
# seocrnet_lkc_1205_binary3_s2
# TTA test results
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/origin_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt
# r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_90_img_testB_list.txt
# r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_270_img_testB_list.txt

########################################################################################################
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt
# resize_288_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_90_img_testB_list.txt
# resize_288_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_270_img_testB_list.txt

########################################################################################################
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt
# resize_320_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_90_img_testB_list.txt
# resize_320_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_270_img_testB_list.txt

########################################################################################################
# resize_352
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_img_testB_list.txt
# resize_352_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_h_flip_img_testB_list.txt
# resize_352_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_v_flip_img_testB_list.txt
# resize_352_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_180_img_testB_list.txt
# resize_352_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_90_img_testB_list.txt
# resize_352_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_270_img_testB_list.txt

########################################################################################################
# resize_384
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_img_testB_list.txt
# resize_384_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_h_flip_img_testB_list.txt
# resize_384_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_v_flip_img_testB_list.txt
# resize_384_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_180_img_testB_list.txt
# resize_384_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_90_img_testB_list.txt
# resize_384_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_270_img_testB_list.txt

########################################################################
########################################################################
########################################################################
# seocrnet_lkc_1205_binary3_s2
# TTA test results
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/origin_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt
# r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_90_img_testB_list.txt
# r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_270_img_testB_list.txt

########################################################################################################
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt
# resize_288_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_90_img_testB_list.txt
# resize_288_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_288_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_270_img_testB_list.txt

########################################################################################################
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt
# resize_320_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_90_img_testB_list.txt
# resize_320_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_320_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_270_img_testB_list.txt

########################################################################################################
# resize_352
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_img_testB_list.txt
# resize_352_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_h_flip_img_testB_list.txt
# resize_352_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_v_flip_img_testB_list.txt
# resize_352_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_180_img_testB_list.txt
# resize_352_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_90_img_testB_list.txt
# resize_352_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_352_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_270_img_testB_list.txt

########################################################################################################
# resize_384
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_img_testB_list.txt
# resize_384_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_h_flip_img_testB_list.txt
# resize_384_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_v_flip_img_testB_list.txt
# resize_384_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_180_img_testB_list.txt
# resize_384_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_90_img_testB_list.txt
# resize_384_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/resize_384_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_270_img_testB_list.txt

########################################################################
########################################################################
########################################################################
# sehrnet_cq_1214_binary3_s1
# TTA test results
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/origin_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt
# r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_90_img_testB_list.txt
# r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_270_img_testB_list.txt

########################################################################################################
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_288_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_288_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_288_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_288_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt
# resize_288_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_288_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_90_img_testB_list.txt
# resize_288_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_288_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_270_img_testB_list.txt

########################################################################################################
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_320_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_320_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_320_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_320_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt
# resize_320_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_320_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_90_img_testB_list.txt
# resize_320_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_320_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_270_img_testB_list.txt

########################################################################################################
# resize_352
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_352_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_img_testB_list.txt
# resize_352_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_352_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_h_flip_img_testB_list.txt
# resize_352_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_352_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_v_flip_img_testB_list.txt
# resize_352_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_352_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_180_img_testB_list.txt
# resize_352_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_352_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_90_img_testB_list.txt
# resize_352_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_352_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_270_img_testB_list.txt

########################################################################################################
# resize_384
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_384_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_img_testB_list.txt
# resize_384_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_384_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_h_flip_img_testB_list.txt
# resize_384_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_384_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_v_flip_img_testB_list.txt
# resize_384_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_384_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_180_img_testB_list.txt
# resize_384_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_384_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_90_img_testB_list.txt
# resize_384_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/resize_384_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_270_img_testB_list.txt

########################################################################
########################################################################
########################################################################
# seocrnet_lkc_1205_binary4_s1
# TTA test results
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/origin_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt
# r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_90_img_testB_list.txt
# r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_270_img_testB_list.txt

########################################################################################################
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_288_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_288_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_288_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_288_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt
# resize_288_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_288_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_90_img_testB_list.txt
# resize_288_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_288_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_270_img_testB_list.txt

########################################################################################################
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_320_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_320_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_320_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_320_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt
# resize_320_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_320_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_90_img_testB_list.txt
# resize_320_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_320_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_270_img_testB_list.txt

########################################################################################################
# resize_352
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_352_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_img_testB_list.txt
# resize_352_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_352_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_h_flip_img_testB_list.txt
# resize_352_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_352_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_v_flip_img_testB_list.txt
# resize_352_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_352_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_180_img_testB_list.txt
# resize_352_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_352_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_90_img_testB_list.txt
# resize_352_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_352_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_270_img_testB_list.txt

########################################################################################################
# resize_384
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_384_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_img_testB_list.txt
# resize_384_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_384_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_h_flip_img_testB_list.txt
# resize_384_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_384_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_v_flip_img_testB_list.txt
# resize_384_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_384_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_180_img_testB_list.txt
# resize_384_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_384_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_90_img_testB_list.txt
# resize_384_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/resize_384_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_270_img_testB_list.txt

########################################################################
########################################################################
########################################################################
# seocrnet_lkc_1205_binary4_s2
# TTA test results
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/origin_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt
# r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_90_img_testB_list.txt
# r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_270_img_testB_list.txt

########################################################################################################
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt
# resize_288_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_90_img_testB_list.txt
# resize_288_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_270_img_testB_list.txt

########################################################################################################
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt
# resize_320_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_90_img_testB_list.txt
# resize_320_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_270_img_testB_list.txt

########################################################################################################
# resize_352
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_img_testB_list.txt
# resize_352_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_h_flip_img_testB_list.txt
# resize_352_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_v_flip_img_testB_list.txt
# resize_352_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_180_img_testB_list.txt
# resize_352_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_90_img_testB_list.txt
# resize_352_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_270_img_testB_list.txt

########################################################################################################
# resize_384
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_img_testB_list.txt
# resize_384_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_h_flip_img_testB_list.txt
# resize_384_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_v_flip_img_testB_list.txt
# resize_384_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_180_img_testB_list.txt
# resize_384_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_90_img_testB_list.txt
# resize_384_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_270_img_testB_list.txt

########################################################################
########################################################################
########################################################################
# seocrnet_lkc_1205_binary4_s2
# TTA test results
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/origin_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt
# r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_90_img_testB_list.txt
# r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_270_img_testB_list.txt

########################################################################################################
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt
# resize_288_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_90_img_testB_list.txt
# resize_288_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_288_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_270_img_testB_list.txt

########################################################################################################
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt
# resize_320_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_90_img_testB_list.txt
# resize_320_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_320_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_270_img_testB_list.txt

########################################################################################################
# resize_352
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_img_testB_list.txt
# resize_352_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_h_flip_img_testB_list.txt
# resize_352_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_v_flip_img_testB_list.txt
# resize_352_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_180_img_testB_list.txt
# resize_352_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_90_img_testB_list.txt
# resize_352_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_352_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_270_img_testB_list.txt

########################################################################################################
# resize_384
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_img_testB_list.txt
# resize_384_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_h_flip_img_testB_list.txt
# resize_384_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_v_flip_img_testB_list.txt
# resize_384_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_180_img_testB_list.txt
# resize_384_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_90_img_testB_list.txt
# resize_384_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/resize_384_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_270_img_testB_list.txt

########################################################################
########################################################################
########################################################################
# sehrnet_cq_1214_binary4_s1
# TTA test results
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/origin_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt
# r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_90_img_testB_list.txt
# r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_270_img_testB_list.txt

########################################################################################################
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_288_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_288_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_288_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_288_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt
# resize_288_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_288_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_90_img_testB_list.txt
# resize_288_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_288_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_270_img_testB_list.txt

########################################################################################################
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_320_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_320_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_320_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_320_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt
# resize_320_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_320_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_90_img_testB_list.txt
# resize_320_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_320_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_270_img_testB_list.txt

########################################################################################################
# resize_352
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_352_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_img_testB_list.txt
# resize_352_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_352_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_h_flip_img_testB_list.txt
# resize_352_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_352_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_v_flip_img_testB_list.txt
# resize_352_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_352_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_180_img_testB_list.txt
# resize_352_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_352_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_90_img_testB_list.txt
# resize_352_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_352_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_270_img_testB_list.txt

########################################################################################################
# resize_384
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_384_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_img_testB_list.txt
# resize_384_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_384_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_h_flip_img_testB_list.txt
# resize_384_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_384_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_v_flip_img_testB_list.txt
# resize_384_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_384_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_180_img_testB_list.txt
# resize_384_r_90
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_384_r_90_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_90_img_testB_list.txt
# resize_384_r_270
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/resize_384_r_270_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/best_model_class1/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_270_img_testB_list.txt

########################################################################
########################################################################
########################################################################
# sehrnet_lkc_1124_binary3_s2
# TTA test results
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/origin_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt

########################################################################################################
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_288_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_288_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_288_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_288_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt

########################################################################################################
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_320_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_320_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_320_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_320_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt

########################################################################################################
# resize_352
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_352_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_img_testB_list.txt
# resize_352_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_352_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_h_flip_img_testB_list.txt
# resize_352_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_352_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_v_flip_img_testB_list.txt
# resize_352_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_352_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_180_img_testB_list.txt

########################################################################################################
# resize_384
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_384_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_img_testB_list.txt
# resize_384_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_384_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_h_flip_img_testB_list.txt
# resize_384_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_384_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_v_flip_img_testB_list.txt
# resize_384_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1124_binary3_s2/resize_384_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1124_binary3_s2/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_180_img_testB_list.txt

########################################################################
########################################################################
########################################################################
# sehrnet_cq_1122_s3
# TTA test results
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/origin_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt

########################################################################################################
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_288_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_288_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_288_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_288_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt

########################################################################################################
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_320_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_320_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_320_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_320_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt

########################################################################################################
# resize_352
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_352_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_img_testB_list.txt
# resize_352_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_352_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_h_flip_img_testB_list.txt
# resize_352_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_352_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_v_flip_img_testB_list.txt
# resize_352_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_352_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_352_r_180_img_testB_list.txt

########################################################################################################
# resize_384
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_384_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_img_testB_list.txt
# resize_384_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_384_h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_h_flip_img_testB_list.txt
# resize_384_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_384_v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_v_flip_img_testB_list.txt
# resize_384_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_cq_1122_s3/resize_384_r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_cq_1122_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_384_r_180_img_testB_list.txt

########################################################################
########################################################################
########################################################################
# sehrnet_lkc_1122_binary0_s3
# TTA test results
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary0_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1122_binary0_s3/origin_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1122_binary0_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary0_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1122_binary0_s3/h_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1122_binary0_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary0_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1122_binary0_s3/v_flip_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1122_binary0_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/exp_lkc_aug/sehrnet_w48_imagenet_batch64_multiStage_binary0_2Dice1BCE.yaml \
    --vis_dir ../user_data/saved_log/sehrnet_lkc_1122_binary0_s3/r_180_resultsB \
    TEST.TEST_MODEL ../user_data/saved_model/sehrnet_lkc_1122_binary0_s3/best_model/ \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt
