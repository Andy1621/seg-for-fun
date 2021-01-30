# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import paddle.fluid as fluid
from paddle.fluid.dygraph.parallel import ParallelEnv

from dygraph.datasets import DATASETS
import dygraph.transforms as T
#from dygraph.models import MODELS
from dygraph.cvlibs import manager
from dygraph.utils import get_environ_info
from dygraph.utils import logger
from dygraph.core import train


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # params of model
    parser.add_argument(
        '--model_name',
        dest='model_name',
        help='Model type for training, which is one of {}'.format(
            str(list(manager.MODELS.components_dict.keys()))),
        type=str,
        default='UNet')

    # params of dataset
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help="The dataset you want to train, which is one of {}".format(
            str(list(DATASETS.keys()))),
        type=str,
        default='OpticDiscSeg')
    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help="dataset root directory",
        type=str,
        default=None)

    # params of training
    parser.add_argument(
        "--input_size",
        dest="input_size",
        help="The image size for net inputs.",
        nargs=2,
        default=[512, 512],
        type=int)
    parser.add_argument(
        '--iters',
        dest='iters',
        help='iters for training',
        type=int,
        default=10000)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=2)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=0.01)
    parser.add_argument(
        '--pretrained_model',
        dest='pretrained_model',
        help='The path of pretrained model',
        type=str,
        default=None)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help='The path of resume model',
        type=str,
        default=None)
    parser.add_argument(
        '--save_interval_iters',
        dest='save_interval_iters',
        help='The interval iters for save a model snapshot',
        type=int,
        default=5)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Eval while training',
        action='store_true')
    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='Display logging information at every log_iters',
        default=10,
        type=int)
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to record the data to VisualDL during training',
        action='store_true')

    return parser.parse_args()


def main(args):
    env_info = get_environ_info()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['\n', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    places = fluid.CUDAPlace(ParallelEnv().dev_id) \
        if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] \
        else fluid.CPUPlace()

    if args.dataset not in DATASETS:
        raise Exception('`--dataset` is invalid. it should be one of {}'.format(
            str(list(DATASETS.keys()))))
    dataset = DATASETS[args.dataset]

    with fluid.dygraph.guard(places):
        # Creat dataset reader
        train_transforms = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.ResizeStepScaling(0.5, 2.0, 0.25),
            T.RandomPaddingCrop(args.input_size),
            T.RandomDistort(),
            T.Normalize(),
        ])
        train_dataset = dataset(
            dataset_root=args.dataset_root,
            transforms=train_transforms,
            mode='train')

        eval_dataset = None
        if args.do_eval:
            eval_transforms = T.Compose([T.Normalize()])
            eval_dataset = dataset(
                dataset_root=args.dataset_root,
                transforms=eval_transforms,
                mode='val')

        model = manager.MODELS[args.model_name](
            num_classes=train_dataset.num_classes,
            pretrained_model=args.pretrained_model)

        # Creat optimizer
        # todo, may less one than len(loader)
        num_iters_each_epoch = len(train_dataset) // (
            args.batch_size * ParallelEnv().nranks)
        lr_decay = fluid.layers.polynomial_decay(
            args.learning_rate, args.iters, end_learning_rate=0, power=0.9)
        optimizer = fluid.optimizer.Momentum(
            lr_decay,
            momentum=0.9,
            parameter_list=model.parameters(),
            regularization=fluid.regularizer.L2Decay(regularization_coeff=4e-5))

        train(
            model,
            train_dataset,
            places=places,
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            save_dir=args.save_dir,
            iters=args.iters,
            batch_size=args.batch_size,
            resume_model=args.resume_model,
            save_interval_iters=args.save_interval_iters,
            log_iters=args.log_iters,
            num_classes=train_dataset.num_classes,
            num_workers=args.num_workers,
            use_vdl=args.use_vdl)


if __name__ == '__main__':
    args = parse_args()
    main(args)
