import io, requests
import os
import attr
import math
import torch
import torch.nn as nn
from mmcv.cnn import trunc_normal_init
from collections import OrderedDict
from functools import partial

from ..builder import HEADS
from .base import BaseHead

from ..dall_e import load_model


@HEADS.register_module()
class BEVTMultiSourceHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 vae_weight_path,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 init_std=0.02,
                 loss_weight_2d=1.0,
                 loss_weight_3d=1.0,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.lm_head = nn.Linear(self.in_channels, self.num_classes)
        self.vae_encoder = load_model(os.path.join(vae_weight_path, "encoder.pkl"))
        self.vae_encoder.eval()
        self.loss_weight_2d = loss_weight_2d
        self.loss_weight_3d = loss_weight_3d

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.lm_head, std=self.init_std)

    def train(self, mode=True):
        super().train(mode)
        self.vae_encoder.eval()
        for p in self.vae_encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.lm_head(x)

    def loss_mask(self, cls_score_3d, cls_score_2d, imgs_3d, imgs_2d, position_masks, position_masks_2d, **kwargs):
        losses = dict()
        clip_len = position_masks.shape[1]
        with torch.no_grad():
            # [N, clip_len, h*w]
            position_masks = position_masks.flatten(2).to(torch.bool)
            position_masks_2d = position_masks_2d.flatten(1).to(torch.bool)
            num_imgs_3d = imgs_3d.size(0)
            num_imgs_2d = imgs_2d.size(0)
            input_ids = torch.argmax(self.vae_encoder(torch.cat((imgs_3d, imgs_2d), dim=0)), axis=1).flatten(1)
            input_ids_3d = input_ids[:num_imgs_3d]
            input_ids_2d = input_ids[num_imgs_3d:]
            input_ids_3d = input_ids_3d.view(-1, clip_len, input_ids_3d.shape[1])
            labels_3d = input_ids_3d[position_masks]
            # 2d masks
            input_ids_2d = input_ids_2d.view(num_imgs_2d, -1)
            labels_2d = input_ids_2d[position_masks_2d]
            # [N, clip_len, h*w]
            position_masks = position_masks.flatten(1)

        cls_score_3d = cls_score_3d[position_masks]
        cls_score_2d = cls_score_2d[position_masks_2d]
        with torch.no_grad():
            pred_labels_3d = torch.argmax(cls_score_3d, dim=1)
            mask_acc_3d = torch.true_divide(torch.sum(pred_labels_3d == labels_3d), labels_3d.shape[0])
            pred_labels_2d = torch.argmax(cls_score_2d, dim=1)
            mask_acc_2d = torch.true_divide(torch.sum(pred_labels_2d == labels_2d), labels_2d.shape[0])
        loss_cls_3d = self.loss_weight_3d * self.loss_cls(cls_score_3d, labels_3d, **kwargs)
        loss_cls_2d = self.loss_weight_2d * self.loss_cls(cls_score_2d, labels_2d, **kwargs)

        if isinstance(loss_cls_3d, dict):
            losses.update(loss_cls_3d)
        else:
            losses['loss_cls_3d'] = loss_cls_3d

        if isinstance(loss_cls_2d, dict):
            losses.update(loss_cls_2d)
        else:
            losses['loss_cls_2d'] = loss_cls_2d

        losses['mask_acc_3d'] = mask_acc_3d
        losses['mask_acc_2d'] = mask_acc_2d

        return losses
