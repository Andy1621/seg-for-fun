# CCF遥感影像地块分割

**Team:paddle针不戳**

|      | 初赛（Top2） | 复赛（Top1） |
| ---- | ------------ | ------------ |
| A榜  | 0.73418844   | 0.71751271   |
| B榜  | 0.74324144   | 0.71104908   |

## 说明

- [2020 CCF BDCI 遥感地块分割比赛](https://www.datafountain.cn/competitions/475)Top1解决方案
- 方案介绍见知乎专栏[2020 CCF BDCI 地块分割Top1方案 & 语义分割trick整理](https://zhuanlan.zhihu.com/p/346862877)
- 基于[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)静态图版本开发，现已推出正式2.0动态图版本，建议使用新版动态图，改写更加方便。方案及代码仅供参考，不建议实际运行训练脚本（狗头）
- 模型和结果见image目录下的README.md。打比赛时由于并未限制模型规模、推理效率，模型融合时把所有的模型都用上了。实际上相同的网络结构，不同规模的模型，融合后提升很小，部分甚至对performance有损害。再者，TTA时尺度放缩影响不大，不同的翻转即可很好提升融合结果

### 数据目录

```
|-- data
	|-- user_data
    	|-- 处理后数据
	    |-- 模型文件
	    |-- 中间预测
    |-- prediction_result
	    |-- result.zip
    |-- code
	    |-- 这里是docker执行时需要的代码（main.sh为程序入口）
    |-- raw_data
        |-- 比赛的数据集文件（train_data.zip, img_testA.zip, img_testB.zip)
    |--README.md
```

### 仓库结构

核心代码位于`code`目录，在`PaddleSeg`基础上开发

1. `main.sh`为程序入口，根据下面**算法运行**介绍运行训练，见`main.sh`注释
2. 主要修改`pdseg`目录下文件，下面分别介绍复现代码相关文件以及无关文件
3. 相关文件：
   1. `modeling/hrnet.py`在原`hrnet`结构上加入`SE/CBAM/SCSE attention`结构，以及`xavier`初始化
   2. `tools`目录封装了数据集生成以及后处理等文件
      1. `generate_my_dataset.py`生成数据集核心文件，见`README.md`，包括正负样本划分、重采样、二分类样本增强等
      2. `binary_class_voting.py`生成二分类模型测试结果，方便后续快速投票，减少遍历
      3. `invert_binary_class_results.py`变换二分类TTA测试后结果
      4. `multi_class_voting.py`生成多分类模型测试结果，方便后续快速投票，减少遍历
      5. `invert_multi_class_results.py`变换多分类TTA测试后结果
      6. `post_processing.py`对结果进行投票以及后处理
   3. `data_aug_lkc.py`中封装了一些调用`albumentations`库写的数据增强，并修改了`reader.py`读取并混合正负样本
   4. `utils/config.py`中对`config`进行了相应的修改
   5. `train.py`做出了相应修改，以存储不同结果
4. 无关文件
   1. `cq_iou.py`计算测试结果，针对验证集用，计算`iou`和`acc`
   2. `param_avg.py`进行模型参数融合，无用
   3. `vis_tta.py`直接进行团投票，无用

### 算法运行

`cd ./code`目录运行`bash main.sh`会依次执行下面文件，默认`false`关闭训练、测试以及模型voting过程，仅开启后处理过程，如有需要可自行打开。

**训练时注意是否正常开启训练，可能会因为数据集路径问题（yaml文件、数据集txt等）导致读取文件失败，直接跳过训练进行测试（PaddleSeg遇到异常不中断，需手动debug）**。

```shell
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
```

上述程序会自动运行如下脚本

1. `./exp/prepare_dataset.sh`解压压缩包并划分训练集与验证集，对数据进行预处理（划分正负样本、原数据重采样、生成二分类数据）

   ```shell
   # 解压压缩包并划分训练集与验证集
   ...
   # 划分类4类5正负样本, 原数据类4类5重采样, 生成类0类3类4二分类数据
   ...
   ```

2. `./exp/train_multi_class.sh`和`./exp/train_binary_class.sh`分别训练多分类模型与二分类模型

   ```shell
   # 训练多分类模型
   ...
   ```

   ```shell
   # 训练二分类模型
   ...
   ```

3. `./exp/test_multi_class.sh`和`./exp/test_binary_class.sh`分别测试多分类模型与二分类模型

   ```shell
   # 测试多分类模型
   ...
   # TTA结果恢复
   ...
   ```

   ```shell
   # 测试二分类模型
   ...
   # TTA结果恢复
   ...
   ```

4. `./exp/generate_results.sh`对测试结果进行voting融合、后处理以及打包

   ```shell
   # 全分类模型voting融合
   ...
   # 二分类模型voting融合
   ...
   # 后处理
   ...
   # 结果打包
   ...
   ```

   



