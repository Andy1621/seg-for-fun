# 训练二分类模型

# '../user_data/saved_log/sehrnet_lkc_1122_binary0_s3'

# '../user_data/saved_log/sehrnet_lkc_1124_binary3_s2',
# '../user_data/saved_log/seocrnet_lkc_1205_binary3_s1',
# '../user_data/saved_log/seocrnet_lkc_1205_binary3_s2',
# '../user_data/saved_log/seocrnet_lkc_1211_binary3_s1',
# '../user_data/saved_log/sehrnet_cq_1214_binary3_s1'

# '../user_data/saved_log/sehrnet_cq_1122_s3',
# '../user_data/saved_log/seocrnet_lkc_1205_binary4_s1',
# '../user_data/saved_log/seocrnet_lkc_1205_binary4_s2',
# '../user_data/saved_log/seocrnet_lkc_1211_binary4_s1',
# '../user_data/saved_log/sehrnet_cq_1214_binary4_s1'

# sehrnet_lkc_1122_binary0_s3
# 多阶段训练类0，常规增强，batch64，2 dice + 1 bce
# stage1 => P: NP = 1: 3
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_binary0_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/sehrnet_lkc_1122_binary0_s1/ \
    DATASET.NEGETIVE_RATIO 3 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_log/saved_model/sehrnet_lkc_1122_binary0_s1/ \
    SOLVER.NUM_EPOCHS 100 \
    SOLVER.LR 15e-3
# stage2 => P: NP = 1: 5
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_binary0_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/sehrnet_lkc_1122_binary0_s2/ \
    DATASET.NEGETIVE_RATIO 5 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_log/saved_model/sehrnet_lkc_1122_binary0_s2/ \
    TRAIN.PRETRAINED_MODEL_DIR /../user_data/saved_log/saved_model/sehrnet_lkc_1122_binary0_s1/best_model/ \
    SOLVER.NUM_EPOCHS 50 \
    SOLVER.LR 15e-4
# stage2 => P: NP = 1: 7
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_binary0_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/sehrnet_lkc_1122_binary0_s3/ \
    DATASET.NEGETIVE_RATIO 7 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_log/saved_model/sehrnet_lkc_1122_binary0_s3/ \
    TRAIN.PRETRAINED_MODEL_DIR ../user_data/saved_log/saved_model/sehrnet_lkc_1122_binary0_s2/best_model/ \
    SOLVER.NUM_EPOCHS 25 \
    SOLVER.LR 15e-5

# sehrnet_lkc_1124_binary3_s2
# 多阶段训练类3，常规增强，batch64，2 dice + 1 bce
# stage1 => P: NP = 1: 1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/saved_log/sehrnet_lkc_1124_binary3_s1/ \
    DATASET.NEGETIVE_RATIO 1 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_log/saved_model/sehrnet_lkc_1124_binary3_s1/ \
    SOLVER.NUM_EPOCHS 100 \
    SOLVER.LR 15e-3
# stage2 => P: NP = 1: 3
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/saved_log/sehrnet_lkc_1124_binary3_s2/ \
    DATASET.NEGETIVE_RATIO 3 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_log/saved_model/sehrnet_lkc_1124_binary3_s2/ \
    TRAIN.PRETRAINED_MODEL_DIR /../user_data/saved_log/saved_model/sehrnet_lkc_1124_binary3_s1/best_model/ \
    SOLVER.NUM_EPOCHS 50 \
    SOLVER.LR 15e-4

# seocrnet_lkc_1205_binary3_s1
# SEOCRNet_W64训类3，增加类3样本（scale）
# 常规增强+随机旋转90/180/270，batch64
# stage1 => P: NP = 8: 1
# epoch: 100
# lr: 0.015
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s1/ \
    DATASET.NEGETIVE_RATIO 0.125 \
    BATCH_SIZE 56 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/ \
    TRAIN.CLASS_NUM '[1]' \
    SOLVER.NUM_EPOCHS 80 \
    SOLVER.LR 15e-3
# seocrnet_lkc_1205_binary3_s2
# stage2 => P: NP = 4: 1
# epoch: 50
# lr: 0.0015
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/seocrnet_lkc_1205_binary3_s2/ \
    DATASET.NEGETIVE_RATIO 0.25 \
    BATCH_SIZE 56 \
    TRAIN.PRETRAINED_MODEL_DIR ../user_data/saved_model/seocrnet_lkc_1205_binary3_s1/best_model_class1/ \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/seocrnet_lkc_1205_binary3_s2/ \
    TRAIN.CLASS_NUM '[1]' \
    SOLVER.NUM_EPOCHS 40 \
    SOLVER.LR 15e-4

# seocrnet_lkc_1211_binary3_s1
# SEOCRNet_W48训类3，增加类3样本（scale）
# 常规增强+随机旋转90/180/270，batch64
# stage1 => P: NP = 8: 1
# epoch: 80
# lr: 0.015
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/seocrnet_lkc_1211_binary3_s1/ \
    DATASET.NEGETIVE_RATIO 0.125 \
    BATCH_SIZE 64 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/seocrnet_lkc_1211_binary3_s1/ \
    TRAIN.CLASS_NUM '[1]' \
    SOLVER.NUM_EPOCHS 80 \
    SOLVER.LR 15e-3

# sehrnet_cq_1214_binary3_s1
# SEHRNet_W64训类3，增加类3样本（scale）
# 常规增强+随机旋转90/180/270，batch128
# stage1 => P: NP = 4: 1
# epoch: 60
# lr: 0.015
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w64_imagenet_batch64_multiStage_binary3_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/sehrnet_cq_1214_binary3_s1/ \
    DATASET.NEGETIVE_RATIO 0.125 \
    BATCH_SIZE 64 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/sehrnet_cq_1214_binary3_s1/ \
    TRAIN.CLASS_NUM '[1]' \
    SOLVER.NUM_EPOCHS 80 \
    SOLVER.LR 15e-3

################################################################
# 类4
# sehrnet_cq_1122_s1
# 多阶段训练，常规增强，batch64，2 dice + 1 bce
# stage1 => P: NP = 1: 3
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/saved_log/sehrnet_cq_1122_s1/ \
    DATASET.NEGETIVE_RATIO 3 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_log/saved_model/sehrnet_cq_1122_s1/ \
    TRAIN.RESUME_MODEL_DIR ../user_data/saved_log/saved_model/sehrnet_cq_1122_s1/latest \
    SOLVER.NUM_EPOCHS 100 \
    SOLVER.LR 15e-3
# stage2 => P: NP = 1: 5
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/saved_log/sehrnet_cq_1122_s2/ \
    DATASET.NEGETIVE_RATIO 5 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_log/saved_model/sehrnet_cq_1122_s2/ \
    TRAIN.PRETRAINED_MODEL_DIR ../user_data/saved_log/saved_model/sehrnet_cq_1122_s1/best_model/ \
    SOLVER.NUM_EPOCHS 50 \
    SOLVER.LR 15e-4
# stage3 => P: NP = 1: 7
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/saved_log/sehrnet_cq_1122_s3/ \
    DATASET.NEGETIVE_RATIO 7 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_log/saved_model/sehrnet_cq_1122_s3/ \
    TRAIN.PRETRAINED_MODEL_DIR ../user_data/saved_log/saved_model/sehrnet_cq_1122_s2/best_model/ \
    SOLVER.NUM_EPOCHS 25 \
    SOLVER.LR 15e-5

# seocrnet_lkc_1205_binary4_s1
# SEOCRNet_W64训类4，增加类4样本（scale）
# 常规增强+随机旋转90/180/270，batch64
# stage1 => P: NP = 8: 1
# epoch: 100
# lr: 0.015
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s1/ \
    DATASET.NEGETIVE_RATIO 0.125 \
    BATCH_SIZE 56 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/ \
    TRAIN.CLASS_NUM '[1]' \
    SOLVER.NUM_EPOCHS 80 \
    SOLVER.LR 15e-3
# seocrnet_lkc_1205_binary4_s2
# stage2 => P: NP = 4: 1
# epoch: 50
# lr: 0.0015
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/seocrnet_lkc_1205_binary4_s2/ \
    DATASET.NEGETIVE_RATIO 0.25 \
    BATCH_SIZE 56 \
    TRAIN.PRETRAINED_MODEL_DIR ../user_data/saved_model/seocrnet_lkc_1205_binary4_s1/best_model_class1/ \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/seocrnet_lkc_1205_binary4_s2/ \
    TRAIN.CLASS_NUM '[1]' \
    SOLVER.NUM_EPOCHS 40 \
    SOLVER.LR 15e-4

# seocrnet_lkc_1211_binary4_s1
# SEOCRNet_W48训类4，增加类4样本（scale）
# 常规增强+随机旋转90/180/270，batch64
# stage1 => P: NP = 8: 1
# epoch: 80
# lr: 0.015
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/seocrnet_w48_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/seocrnet_lkc_1211_binary4_s1/ \
    DATASET.NEGETIVE_RATIO 0.125 \
    BATCH_SIZE 64 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/seocrnet_lkc_1211_binary4_s1/ \
    TRAIN.CLASS_NUM '[1]' \
    SOLVER.NUM_EPOCHS 80 \
    SOLVER.LR 15e-3

# sehrnet_cq_1214_binary4_s1
# SEHRNet_W64训类4，增加类4样本（scale）
# 常规增强+随机旋转90/180/270，batch128
# stage1 => P: NP = 4: 1
# epoch: 60
# lr: 0.015
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m paddle.distributed.launch pdseg/train.py --use_gpu \
    --cfg ./exp/model_config/sehrnet_w64_imagenet_batch64_multiStage_binary4_2Dice1BCE.yaml \
    --use_mpio --do_eval --use_vdl --vdl_log_dir ../user_data/saved_log/sehrnet_cq_1214_binary4_s1/ \
    DATASET.NEGETIVE_RATIO 0.125 \
    BATCH_SIZE 64 \
    TRAIN.MODEL_SAVE_DIR ../user_data/saved_model/sehrnet_cq_1214_binary4_s1/ \
    TRAIN.CLASS_NUM '[1]' \
    SOLVER.NUM_EPOCHS 80 \
    SOLVER.LR 15e-3