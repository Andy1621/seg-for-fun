# CCF遥感影像地块分割

**Team:paddle针不戳**

|      | 初赛（Top2） | 复赛（Top1） |
| ---- | ------------ | ------------ |
| A榜  | 0.73418844   | 0.71751271   |
| B榜  | 0.74324144   | 0.71104908   |

## 一些声明

- 详细的处理过程见`/data/README.md`。
- 由于打比赛时我们并没有考虑实际工程需要，训练的模型较大，并且将比赛过程中较好的模型都进行了集成，导致最后训练了9个全分类模型以及11个二分类模型，训练过程存在很大的冗余。再者，最后的测试结果使用了不同尺度的冗余TTA，在实际生产环境中难以应用。希望以后的赛制可以提前把结果判分规则说明，如果模型大小与数量作为约束的话，选手一开始也会尽可能利用有限的资源进行设计。
- **我们的核心思路有三**：
  1. **对小样本进行重采样**，最暴力的方法是在在原始数据集中抽取出包含小样本类的图片集，并进行合适的数据增强，并入原始数据集中，实验发现可以明显增强对小样本的判别效果。
  2. **转换为二分类问题解决连通性问题**，可划分正负样本多阶段训练类3类4，负样本指不包含类3类4的数据，多阶段渐进增加负样本比例，紧接着将TTA预测的多个预测voting到原始预测，实验发现可以很好地提升预测结果连通性。
  3. **传统图像后处理**，利用膨胀腐蚀、开闭运算、**骨架提取膨胀**、小区域孤立团去除，可以进一步增强图像连通性。

## docker运行

1. **导入仓库`git clone https://github.com/Andy1621/seg-for-fun.git`**

2. **下载数据集压缩包到`data/raw_data`目录下**

3. 下载镜像并放入imame目录下：链接：https://pan.baidu.com/s/14w9XkSaj5ebyloiJZ0YQ5Q ，提取码：tqhr 

4. 默认运行会关闭训练、测试以及voting过程（花费时间较长），仅启动后处理过程。**请下载后处理所需文件，并解压到对应目录后运行**，更详细的处理过程见`/data/README.md`。
   - **后处理所需文件下载地址（解压后放入`data/user_data/temp_results`）**：链接：https://pan.baidu.com/s/18EVb-GoK8TzP3qry7S_ZWQ ，提取码：zhbo 
   - 处理后训练数据较大，建议直接运行脚本生成数据
   - 训练模型下载地址（解压后放入`data/user_data/saved_model`）：链接：https://pan.baidu.com/s/1nXH8gTjFNVsPmefpTQVZQw ，提取码：fbj4 
   - 测试中间结果下载地址（解压后放入`data/user_data/saved_log`）：链接：https://pan.baidu.com/s/18OTzfb5wHojR4sHdlgOnwQ ，提取码：p4sa 

5. 按如下步骤进行（任意装有`PaddlePaddle`的环境都可以直接进入`code`目录运行`main.sh`，自动安装相关依赖）

   1. 下载并读入镜像

      ```shell
      cat paddle.tar | sudo docker import - test 
      ```

   2. 运行并把本机绝对目录`seg-for-fun` （**请使用绝对路径，相对路径可能报错**）映射到`/data`

      ```shell
      sudo docker run -it -v /home/user/seg-for-fun/:/data  test  /bin/bash
      ```

   3. 进入`image`目录运行`run.sh`

      ```shell
      cd /data/image
      bash ./run.sh
      ```

      

