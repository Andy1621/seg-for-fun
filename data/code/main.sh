# 安装依赖
pip install -r ./requirements.txt

# ./exp/prepare_dataset.sh划分训练集与验证集，对数据进行预处理（划分正负样本、原数据重采样、生成二分类数据）
if  [ ! -d ../user_data/dataset ]; then
    mkdir ../user_data/dataset
fi
bash ./exp/prepare_dataset.sh

# ./exp/train_multi_class.sh和./exp/train_binary_class.sh分别训练多分类模型与二分类模型
if  [ ! -d ../user_data/saved_model ]; then
    mkdir ../user_data/saved_model
fi
if  [ ! -d ../user_data/saved_log ]; then
    mkdir ../user_data/saved_log
fi
# 默认false关闭模型训练
bash ./exp/train_multi_class.sh false
bash ./exp/train_binary_class.sh false

# ./exp/test_multi_class.sh和./exp/test_binary_class.sh分别测试多分类模型与二分类模型
# 默认false关闭模型测试
bash ./exp/test_multi_class.sh false false
bash ./exp/test_binary_class.sh false false

# ./exp/generate_results.sh对测试结果进行voting融合、后处理以及打包
# 第一个默认false关闭模型融合与后处理，第二个默认true打包输出结果
bash ./exp/generate_results.sh false true
