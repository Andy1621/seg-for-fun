import os
import pickle
import six

path = '/data5/ckli/exp_paddle/saved_model/se_ocrnet_lkc_1130_class4And5'

model_dir_list = [
    'best_model',
    '140',
    'iou41_class5',
]

model_dict_list = []

for model_dir in model_dir_list:
    with open(os.path.join(path, model_dir, 'model.pdparams'), 'rb') as f:
        model_dict_list.append(pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1'))

avg_model = {}

for k, v in model_dict_list[0].items():
    value = 0.0
    for i in range(len(model_dir_list)):
        value += model_dict_list[i][k]
    avg_model[k] = value / len(model_dir_list)

output_dir = os.path.join(path, 'avg')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
with open(os.path.join(output_dir, 'model.pdparams'), 'wb') as f:
    pickle.dump(avg_model, f)

print(type(avg_model), 'ok')