# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import paddle
import numpy as np
from .raster import Raster
from ssm.datasets.transforms import Compose


class RSDataset(paddle.io.Dataset):
    def __init__(self,
                 transforms,
                 dataset_root,
                 num_classes,
                 mode='train',
                 work='seg',
                 file_path=None,
                 separator=' ',
                 ignore_index=255,
                 rgb_bands=[1, 2, 3],
                 big_map=False, 
                 grid_size=[512, 512],
                 overlap=[0, 0]):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms, rgb_bands)
        self.file_list = list()
        mode = mode.lower()
        work = work.lower()
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.big_map = big_map
        self.cimw_1 = None
        self.cimw_2 = None
        self.claw = None
        self.all_grids_number = 0
        self.__idx = 0
        if mode.lower() not in ['train', 'val', 'infer']:
            raise ValueError(
                "mode should be 'train', 'val' or 'infer', but got {}.".format(mode))
        if work.lower() not in ['seg', 'det', 'cd']:
            raise ValueError(
                "work should be 'seg', 'det' or 'cd', but got {}.".format(work))
        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")
        self.dataset_root = dataset_root
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))
        if file_path is None:
            raise ValueError(
                '`file_path` is necessary, but it is None.'
            )
        elif not os.path.exists(file_path):
            raise FileNotFoundError(
                '`file_path` is not found: {}'.format(file_path))
        else:
            file_path = file_path
        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) == 1 and work != "cd" and mode == "infer":
                    image1_path = os.path.join(self.dataset_root, items[0])
                    image2_path = None
                    label_path = None
                elif len(items) == 2 and work == "cd" and mode == "infer":
                    image1_path = os.path.join(self.dataset_root, items[0])
                    image2_path = os.path.join(self.dataset_root, items[1])
                    label_path = None
                elif len(items) == 2 and work != "cd" and mode != "infer":
                    image1_path = os.path.join(self.dataset_root, items[0])
                    image2_path = None
                    label_path = os.path.join(self.dataset_root, items[1])
                elif len(items) == 3 and work == "cd" and mode != "infer":
                    image1_path = os.path.join(self.dataset_root, items[0])
                    image2_path = os.path.join(self.dataset_root, items[1])
                    label_path = os.path.join(self.dataset_root, items[2])
                else:
                    raise ValueError(
                        "File list format incorrect! In training or evaluation task it should be"
                        " image1_name[{}image2_name]{}label_name\\n".format(separator))
                image1_worker = Raster(image1_path, rgb_bands, big_map, grid_size, overlap)
                image2_worker = Raster(image2_path, rgb_bands, big_map, grid_size, overlap) \
                                if image2_path is not None else None
                label_worker = Raster(label_path, [1, 1, 1], big_map, grid_size, overlap) \
                               if label_path is not None else None
                self.all_grids_number += image1_worker.grids_number
                self.file_list.append([image1_worker, image2_worker, label_worker])

    def __getitem__(self, idx):
        if idx < self.__len__():
            if self.big_map is False:
                self.__idx = idx
                self.cimw_1, self.cimw_2, self.claw = self.file_list[self.__idx]
            else:
                if (self.cimw_1 is None and self.claw is None) or \
                (self.cimw_1.cyc_grid is True and self.claw.cyc_grid is True):
                    if self.__idx >= self.file_len:
                        self.__idx = 0
                    self.cimw_1, self.cimw_2, self.claw = self.file_list[self.__idx]
                    self.__idx += 1
            if self.mode == 'infer':
                im1, im2, _ = self.transforms(im1=self.cimw_1.getData(),
                                              im2=self.cimw_2.getData() if \
                                                  self.cimw_2 is not None else None)
                if im2 is None:
                    return im1, self.cimw_1.file_path
                else:
                    return im1, im2, self.cimw_1.file_path
            elif self.mode == 'val':
                im1, im2, _ = self.transforms(im1=self.cimw_1.getData(),
                                              im2=self.cimw_2.getData() if \
                                                  self.cimw_2 is not None else None)
                label = self.claw.getData()
                label = label[np.newaxis, :, :]
                if im2 is None:
                    return im1, label
                else:
                    return im1, im2, label
            else:
                im1, im2, label = self.transforms(im1=self.cimw_1.getData(),
                                                  im2=self.cimw_2.getData() if \
                                                      self.cimw_2 is not None else None ,
                                                  label=self.claw.getData())
                if im2 is None:
                    return im1, label
                else:
                    return im1, im2, label
        else:
            raise StopIteration

    @property
    def file_len(self):
        return len(self.file_list)

    def __len__(self):
        return self.all_grids_number