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


import os
import math

import cv2
import numpy as np
import paddle

from ssm import utils
from ssm.core import infer
from ssm.utils import logger, progbar, visualize

try:
    from osgeo import gdal
except ImportError:
    import gdal


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def tensor2numpy(im, convert_bgr=True):
    im = paddle.squeeze(im).numpy()
    im = im.transpose((1, 2, 0)).astype('uint8')
    if convert_bgr:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im


# TODO: big_map
def predict(model,
            model_path,
            infer_dataset,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None,
            num_workers=0,
            color_weight=0.6,
            custom_color=None):
    """
    predict and visualize the image_list.
    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        infer_dataset (ssm.datasets.RSDataset): Used to read and process infer datasets.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        color_weight (float, optional): visualize's weight. Default: 0.6.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.
    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    # local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
    batch_sampler = paddle.io.DistributedBatchSampler(
        infer_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        infer_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )
    total_iters = len(loader)

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=total_iters, verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        for iter, data in enumerate(loader):
            if len(data) == 2:
                im1 = data[0]
                im2 = None
                im1_path = data[1][0]
            else:  # len(data) == 3
                im1 = data[0]
                im2 = data[1]
                im1_path = data[2][0]
            ori_shape = im1.shape[-2:]
            if aug_pred:
                pred = infer.aug_inference(
                    model,
                    im1,
                    im2,
                    ori_shape=ori_shape,
                    transforms=infer_dataset.transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer.inference(
                    model,
                    im1,
                    im2,
                    ori_shape=ori_shape,
                    transforms=infer_dataset.transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            # get the saved name
            im_file = os.path.basename(im1_path)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # save added image
            added_image_1 = utils.visualize.visualize(
                tensor2numpy(im1), pred, color_map, weight=color_weight)
            if im2 is None:
                added_image_path_1 = os.path.join(added_saved_dir, im_file)
            else:
                added_image_2 = utils.visualize.visualize(
                    tensor2numpy(im2), pred, color_map, weight=color_weight)
                imp = os.path.splitext(im_file)
                added_image_path_2 = os.path.join(added_saved_dir, (imp[0] + "_B." + imp[1]))
                mkdir(added_image_path_2)
                cv2.imwrite(added_image_path_2, added_image_2)
                added_image_path_1 = os.path.join(added_saved_dir, (imp[0] + "_A." + imp[1]))
            mkdir(added_image_path_1)
            cv2.imwrite(added_image_path_1, added_image_1)

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            pred_saved_path = os.path.join(
                pred_saved_dir,
                os.path.splitext(im_file)[0] + "_P.png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            # pred_im = utils.visualize(im_path, pred, weight=0.0)
            # pred_saved_path = os.path.join(pred_saved_dir, im_file)
            # mkdir(pred_saved_path)
            # cv2.imwrite(pred_saved_path, pred_im)

            progbar_pred.update(iter + 1)