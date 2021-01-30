# TTA测多模型
###########################################################################
###########################################################################
# se_ocrnet_cq_1204_1
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/origin_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/resize_288_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/resize_288_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/resize_288_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/resize_288_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/resize_320_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/resize_320_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/resize_320_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/resize_320_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_cq_1204_1/best_model

###########################################################################
###########################################################################
# se_ocrnet_lkc_1130_class4And5
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/origin_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/resize_288_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/resize_288_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/resize_288_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/resize_288_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/resize_320_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/resize_320_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/resize_320_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/resize_320_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1130_class4And5/best_model

###########################################################################
###########################################################################
# se_ocrnet_lkc_1126_class4And5
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/origin_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/resize_288_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/resize_288_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/resize_288_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/resize_288_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/resize_320_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/resize_320_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/resize_320_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/resize_320_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class4And5/best_model

###########################################################################
###########################################################################
# hrnet_cq_1120_1
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/origin_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/resize_288_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/resize_288_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/resize_288_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/resize_288_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/resize_320_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/resize_320_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/resize_320_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1120_1/resize_320_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1120_1/best_model

###########################################################################
###########################################################################
# se_hrnet_lkc_1124_class4And5_s2
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/origin_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/resize_288_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/resize_288_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/resize_288_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/resize_288_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/resize_320_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/resize_320_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/resize_320_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml   \
    --vis_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/resize_320_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/best_model


###########################################################################
###########################################################################
# hrnet_lkc_1112_1
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/origin_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/resize_288_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/resize_288_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/resize_288_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/resize_288_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/resize_320_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/resize_320_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/resize_320_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_lkc_1112_1/resize_320_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_lkc_1112_1/best_model

###########################################################################
###########################################################################
# hrnet_cq_1112_1
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/origin_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/resize_288_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/resize_288_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/resize_288_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/resize_288_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/resize_320_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/resize_320_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/resize_320_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml   \
    --vis_dir ../user_data/saved_log/hrnet_cq_1112_1/resize_320_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/hrnet_cq_1112_1/best_model

###########################################################################
###########################################################################
# seocrnet_cq_1126_class4_s2
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/origin_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/resize_288_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/resize_288_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/resize_288_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/resize_288_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/resize_320_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/resize_320_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/resize_320_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml   \
    --vis_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/resize_320_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/seocrnet_cq_1126_class4_s2/best_model

###########################################################################
###########################################################################
# se_ocrnet_lkc_1126_class5_s2
# origin
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/origin_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model
# h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model
# v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model
# r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model
# resize_288
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/resize_288_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model
# resize_288_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/resize_288_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model
# resize_288_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/resize_288_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model
# resize_288_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/resize_288_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_288_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model
# resize_320
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/resize_320_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model
# resize_320_h_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/resize_320_h_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_h_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model
# resize_320_v_flip
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/resize_320_v_flip_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_v_flip_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model
# resize_320_r_180
CUDA_VISIBLE_DEVICES=0 \
    python3 pdseg/vis.py --use_gpu --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml   \
    --vis_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/resize_320_r_180_resultsB \
    DATASET.DATA_DIR ../user_data/dataset \
    DATASET.VIS_FILE_LIST ../user_data/dataset/aug_img_testB/resize_320_r_180_img_testB_list.txt \
    TEST.TEST_MODEL ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/best_model