# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from .checkpoint import save_checkpoint
import apex
from torch.optim import Optimizer
from mmcv.runner.builder import RUNNERS
from mmaction.core import (DistEvalHook, EvalHook, OmniSourceDistSamplerSeedHook, MultiSourceRunner)
import os


@RUNNERS.register_module()
class EpochBasedRunnerAmp(mmcv.runner.EpochBasedRunner):
    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None,
                 amp=False):
        super().__init__(
                 model,
                 batch_processor,
                 optimizer,
                 work_dir,
                 logger,
                 meta,
                 max_iters,
                 max_epochs)
        self.amp = amp

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta, amp=self.amp)
        dst_file = osp.join(out_dir, 'latest.pth')
        shutil.copy(filepath, dst_file)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='cpu',
               resume_amp=False):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        if 'amp' in checkpoint and resume_amp:
            apex.amp.load_state_dict(checkpoint['amp'])
            self.logger.info('load amp state dict')

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)
        del checkpoint
        torch.cuda.empty_cache()

    def auto_resume(self):
        linkname = osp.join(self.work_dir, 'latest.pth')
        if osp.exists(linkname):
            self.logger.info('latest checkpoint found')
            self.resume(linkname)


@RUNNERS.register_module()
class MultiSourceRunnerAmp(MultiSourceRunner):
    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None,
                 amp=False):
        super().__init__(
                 model,
                 batch_processor,
                 optimizer,
                 work_dir,
                 logger,
                 meta,
                 max_iters,
                 max_epochs)
        self.amp = amp

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta, amp=self.amp)
        dst_file = osp.join(out_dir, 'latest.pth')
        shutil.copy(filepath, dst_file)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='cpu',
               resume_amp=False):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        if 'amp' in checkpoint and resume_amp:
            apex.amp.load_state_dict(checkpoint['amp'])
            self.logger.info('load amp state dict')

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)
        del checkpoint
        torch.cuda.empty_cache()

    def auto_resume(self):
        linkname = osp.join(self.work_dir, 'latest.pth')
        if osp.exists(linkname):
            self.logger.info('latest checkpoint found')
            self.resume(linkname)