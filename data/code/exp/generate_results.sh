echo "=====Start generating results====="

# 全分类模型voting融合
# 二分类模型voting融合
if  [ ! -d ../user_data/temp_results ]; then
    mkdir ../user_data/temp_results
fi
if [ $1 == true ]; then
    python3 ./pdseg/tools/multi_class_voting.py ../user_data/dataset ../user_data/temp_results
    python3 ./pdseg/tools/binary_class_voting.py ../user_data/dataset ../user_data/temp_results
fi


# 后处理
if [ $2 == true ]; then
    python3 ./pdseg/tools/post_processing.py ../user_data/dataset ../user_data/temp_results
fi

# 结果打包
if  [ ! -f ../prediction_result/results.zip ]; then
    zip -r ../prediction_result/results.zip ../user_data/temp_results/resultsB
fi


echo "=====Generating results over====="