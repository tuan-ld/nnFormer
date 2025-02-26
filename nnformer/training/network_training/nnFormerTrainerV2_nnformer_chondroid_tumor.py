#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    You may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnformer.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnformer.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnformer.utilities.to_torch import maybe_to_torch, to_cuda
from nnformer.network_architecture.nnFormer_acdc import nnFormer
from nnformer.network_architecture.initialization import InitWeights_He
from nnformer.network_architecture.neural_network import SegmentationNetwork
from nnformer.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
from nnformer.training.dataloading.dataset_loading import unpack_dataset
from nnformer.training.network_training.nnFormerTrainer import nnFormerTrainer
from nnformer.utilities.nd_softmax import softmax_helper
from torch import nn
from torch.cuda.amp import autocast
from nnformer.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *


class nnFormerTrainerV2_nnformer_chondroid_tumor(nnFormerTrainer):
    """
    Trainer for Chondroid Tumor segmentation based on nnFormer architecture.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.load_pretrain_weight = False  # No pretrain weight used

        self.load_plans_file()

        if len(self.plans['plans_per_stage']) == 2:
            Stage = 1
        else:
            Stage = 0

        self.crop_size = self.plans['plans_per_stage'][Stage]['patch_size']
        self.input_channels = self.plans['num_modalities']
        self.num_classes = self.plans['num_classes'] + 1
        self.conv_op = nn.Conv3d

        self.embedding_dim = 96
        self.depths = [2, 2, 2, 2]
        self.num_heads = [3, 6, 12, 24]
        self.embedding_patch_size = [1, 4, 4]
        self.window_size = [[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]]
        self.down_stride = [[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]]
        self.deep_supervision = False

    def initialize_network(self):
        """
        Initialize the network.
        """
        self.network = nnFormer(crop_size=self.crop_size,
                                embedding_dim=self.embedding_dim,
                                input_channels=self.input_channels,
                                num_classes=self.num_classes,
                                conv_op=self.conv_op,
                                depths=self.depths,
                                num_heads=self.num_heads,
                                patch_size=self.embedding_patch_size,
                                window_size=self.window_size,
                                down_stride=self.down_stride,
                                deep_supervision=self.deep_supervision)

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        """
        Set up data augmentation parameters for Chondroid Tumor dataset.
        """
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        self.data_aug_params = default_3D_augmentation_params
        self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['patch_size_for_spatialtransform'] = self.crop_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def initialize_optimizer_and_scheduler(self):
        """
        Initialize the optimizer and scheduler.
        """
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        Perform a single training iteration.
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                loss = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(loss).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            loss = self.loss(output, target)

            if do_backprop:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        return loss.detach().cpu().numpy()

    def run_training(self):
        """
        Run the training process.
        """
        self.maybe_update_lr(self.epoch)
        ds = self.network.do_ds
        if self.deep_supervision:
            self.network.do_ds = True
        else:
            self.network.do_ds = False
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
