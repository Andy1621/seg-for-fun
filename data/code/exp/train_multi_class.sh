echo "=====Start training multi-class models====="

if [ $1 == true ]; then
    python3 ./pretrained_model/download_model.py hrnet_w64_bn_imagenet
    python3 ./pretrained_model/download_model.py HRNet_W48_C_ssld
    bash ./exp/train_multi_class_all.sh
fi

echo "=====Training multi-class models over====="