# '../user_data/saved_log/se_ocrnet_cq_1204_1',
# '../user_data/saved_log/se_ocrnet_lkc_1130_class4And5',
# '../user_data/saved_log/se_ocrnet_lkc_1126_class4And5',
# '../user_data/saved_log/hrnet_cq_1120_1',
# '../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2',
# '../user_data/saved_log/hrnet_lkc_1112_1',
# '../user_data/saved_log/hrnet_cq_1112_1',
# '../user_data/saved_log/seocrnet_cq_1126_class4_s2',
# '../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2'

# 训练多分类模型

# 原数据类4类5重采样，训SE_OCRNet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w64_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5_aug2.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/se_ocrnet_cq_1204_1/ \
    BATCH_SIZE 64 \
    TRAIN.CLASS_NUM '[3, 4, 5]' \
    SOLVER.NUM_EPOCHS 150

# 原数据类4类5重采样，训SE_OCRNet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w48_batch128_lr0.015_poly150_augBaseline_guassnoise_add4And5_aug2.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/se_ocrnet_lkc_1130_class4And5/

# 原数据类4类5重采样，训SE_OCRNet
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class4And5/ 

# SEHRNet + batch 64 + lr 0.015 + poly decay 120 epoch + BN
# mirror + flip + z-score + rotate 30 + blur 0.1 + gaussnoise 0.1
# add class 4 and 5
# [EVAL]#image=2919 acc=0.9382 IoU=0.7405
# [EVAL]Category IoU: [0.8326 0.9120 0.8790 0.9380 0.4495 0.3995 0.7728]
# [EVAL]Category Acc: [0.9067 0.9436 0.9414 0.9708 0.6895 0.6820 0.8846]
# [EVAL]Kappa:0.9063
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_add4And5.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/hrnet_cq_1120_1/

# 增大类4类5比例，对原始图片进行水平/垂直翻转
# 常规增强，batch64
# stage1 => P: NP = 2: 1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s1/ \
    DATASET.NEGETIVE_RATIO 0.5 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s1/ \
    SOLVER.NUM_EPOCHS 100 \
    SOLVER.LR 15e-3
# stage2 => P: NP = 1: 1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_class4And5.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/se_hrnet_lkc_1124_class4And5_s2/ \
    DATASET.NEGETIVE_RATIO 1 \
    TRAIN.PRETRAINED_MODEL_DIR ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s1/ \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/se_hrnet_lkc_1124_class4And5_s2/ \
    SOLVER.NUM_EPOCHS 50 \
    SOLVER.LR 15e-4

# SEHRNet + batch 64 + lr 0.015 + poly decay 120 epoch + BN
# mirror + flip + z-score + rotate 30 + blur 0.1 + gaussnoise 0.1 + rotate90 0.3
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise_rotate90.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/hrnet_lkc_1112_1/

# SEHRNet + batch 64 + lr 0.015 + poly decay 120 epoch + BN
# mirror + flip + z-score + rotate 30 + blur 0.1 + gaussnoise 0.1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/SEHRNet_w48_batch64_lr0.015_poly120_augBaseline_guassnoise.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir /media/sdc/qingchen/Code/SegForFun/saved_log/hrnet_cq_1112_1/

# 增大类4比例，对原始图片进行水平/垂直翻转
# 常规增强，batch64
# stage1 => P: NP = 2: 1
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s1/ \
    DATASET.NEGETIVE_RATIO 0.5 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/seocrnet_cq_1126_class4_s1/ \
    SOLVER.NUM_EPOCHS 100 \
    SOLVER.LR 15e-3
# stage2 => P: NP = 1: 1
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class4.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/seocrnet_cq_1126_class4_s2/ \
    DATASET.NEGETIVE_RATIO 1 \
    TRAIN.PRETRAINED_MODEL_DIR ../user_data/saved_model/seocrnet_cq_1126_class4_s1/ \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/seocrnet_cq_1126_class4_s2/ \
    SOLVER.NUM_EPOCHS 50 \
    SOLVER.LR 15e-4


# 增大类5比例，对原始图片进行水平/垂直翻转
# 常规增强，batch64
# stage1 => P: NP = 2: 1
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s1/ \
    DATASET.NEGETIVE_RATIO 0.5 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s1/ \
    SOLVER.NUM_EPOCHS 100 \
    SOLVER.LR 15e-3
# stage2 => P: NP = 1: 1
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_class5.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/se_ocrnet_lkc_1126_class5_s2/ \
    DATASET.NEGETIVE_RATIO 1 \
    TRAIN.PRETRAINED_MODEL_DIR ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s1/ \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/se_ocrnet_lkc_1126_class5_s2/ \
    SOLVER.NUM_EPOCHS 50 \
    SOLVER.LR 15e-4
