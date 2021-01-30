# 动态图执行

## 下载及添加路径
```
git clone https://github.com/PaddlePaddle/PaddleSeg
cd PaddleSeg
export PYTHONPATH=$PYTHONPATH:`pwd`
cd dygraph
```

## 训练
```
python3 train.py --model_name unet \
--dataset OpticDiscSeg \
--input_size 192 192 \
--iters 10 \
--save_interval_iters 1 \
--do_eval \
--save_dir output
```

## 评估
```
python3 val.py --model_name unet \
--dataset OpticDiscSeg \
--input_size 192 192 \
--model_dir output/best_model
```

## 预测
```
python3 infer.py --model_name unet \
--dataset OpticDiscSeg \
--model_dir output/best_model \
--input_size 192 192
```
