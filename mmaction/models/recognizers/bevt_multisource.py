import numpy as np
import torch
from torch import nn
from einops import rearrange

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class BEVTMultiSource(BaseRecognizer):
    """3D recognizer model framework."""

    def __init__(self,
                 backbone,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, neck=neck,
                         train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)

    def forward(self, imgs, label=None, return_loss=True, **kwargs):
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(imgs, label, **kwargs)

        return self.forward_test(imgs)

    def train_step(self, data_batch, optimizer, data_type=None, **kwargs):
        source_num = len(data_batch)
        video_imgs = []
        video_imgs_second = []
        video_input_position_masks = []
        video_output_position_masks = []
        image_imgs = []
        image_imgs_second = []
        image_input_position_masks = []
        image_output_position_masks = []
        if data_type is None:
            data_type = ['video'] * source_num
        for i in range(source_num):
            if data_type[i] == 'video':
                video_imgs.append(data_batch[i]['imgs'])
                video_imgs_second.append(data_batch[i]['imgs_second'])
                video_input_position_masks.append(data_batch[i]['input_position_masks'])
                video_output_position_masks.append(data_batch[i]['output_position_masks'])
            elif data_type[i] == 'image':
                image_imgs.append(data_batch[i]['imgs'])
                image_imgs_second.append(data_batch[i]['imgs_second'])
                image_input_position_masks.append(data_batch[i]['input_position_masks'])
                image_output_position_masks.append(data_batch[i]['output_position_masks'])
        if len(video_imgs) > 0:
            # video: [B, N, C, T, H, W]
            video_imgs = torch.cat(video_imgs, dim=0)
            video_imgs_second = torch.cat(video_imgs_second, dim=0)
            video_input_position_masks = torch.cat(video_input_position_masks, dim=0)
            video_output_position_masks = torch.cat(video_output_position_masks, dim=0)
            video_imgs = video_imgs.reshape((-1,) + video_imgs.shape[2:])
            video_imgs_second = video_imgs_second.reshape((-1,) + video_imgs_second.shape[2:])
            if len(video_input_position_masks.shape) > 4:
                video_input_position_masks = video_input_position_masks.view((-1,) + video_input_position_masks.shape[-3:])
            if len(video_output_position_masks.shape) > 4:
                video_output_position_masks = video_output_position_masks.view((-1,) + video_output_position_masks.shape[-3:])
        if len(image_imgs) > 0:
            # image: [B, N, C, H, W]
            image_imgs = torch.cat(image_imgs, dim=0)
            image_imgs_second = torch.cat(image_imgs_second, dim=0)
            image_input_position_masks = torch.cat(image_input_position_masks, dim=0)
            image_output_position_masks = torch.cat(image_output_position_masks, dim=0)
            image_imgs = image_imgs.reshape((-1,) + image_imgs.shape[2:])
            image_imgs_second = image_imgs_second.reshape((-1,) + image_imgs_second.shape[2:])
            if len(image_input_position_masks.shape) > 3:
                image_input_position_masks = image_input_position_masks.view(
                    (-1,) + image_input_position_masks.shape[-2:])
            if len(image_output_position_masks.shape) > 3:
                image_output_position_masks = image_output_position_masks.view(
                    (-1,) + image_output_position_masks.shape[-2:])

        aux_info = {}

        losses = self(
            (video_imgs, video_imgs_second, image_imgs, image_imgs_second),
            (video_input_position_masks, video_output_position_masks, image_input_position_masks, image_output_position_masks),
            return_loss=True,
            **aux_info
        )

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            # num_samples=sum([len(next(iter(data_batch[i].values()))) for i in range(len(data_batch))])
            num_samples=len(next(iter(data_batch[0].values())))
        )

        return outputs

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        video_imgs, video_imgs_second, image_imgs, image_imgs_second = imgs
        video_input_position_masks, video_output_position_masks, image_input_position_masks, image_output_position_masks = labels
        losses = dict()
        is_video_train = len(video_imgs) > 0
        is_image_train = len(image_imgs) > 0

        if is_video_train:
            x = self.backbone(video_imgs, video_input_position_masks)
        if is_image_train:
            x_2d = self.backbone(image_imgs, image_input_position_masks, is_img=True)


        cls_score = None
        cls_score_2d = None
        if is_video_train:
            cls_score = self.cls_head(x)
        if is_image_train:
            cls_score_2d = self.cls_head(x_2d)

        loss_cls = self.cls_head.loss_mask(
            cls_score, cls_score_2d, video_imgs_second, image_imgs_second, video_output_position_masks, image_output_position_masks, **kwargs
        )
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
            if self.with_neck:
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            # perform spatio-temporal pooling
            avg_pool = nn.AdaptiveAvgPool3d(1)
            if isinstance(feat, tuple):
                feat = [avg_pool(x) for x in feat]
                # concat them
                feat = torch.cat(feat, axis=1)
            else:
                feat = avg_pool(feat)
            # squeeze dimensions
            feat = feat.reshape((batches, num_segs, -1))
            # temporal average pooling
            feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat)
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs,)

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
