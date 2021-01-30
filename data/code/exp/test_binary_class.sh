echo "=====Start testing binary-class models====="

# 测试二分类模型
if [ $1 == true ]; then
    bash ./exp/test_binary_class_TTA.sh
fi

# TTA结果恢复
if [ $2 == true ]; then
    python3 ./pdseg/tools/invert_binary_class_results.py ../user_data/dataset
fi


echo "=====Testing binary-class models over====="