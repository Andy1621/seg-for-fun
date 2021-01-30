echo "=====Start testing multi-class models====="

# 测试多分类模型
if [ $1 == true ]; then
    bash ./exp/test_multi_class_TTA.sh
fi

# TTA结果恢复
if [ $2 == true ]; then
    python3 ./pdseg/tools/invert_multi_class_results.py ../user_data/dataset
fi


echo "=====Testing multi-class models over====="