import random
import math
import numpy as np
import torch
import itertools
from scipy.spatial.distance import cdist

from ..builder import PIPELINES

"""
This class is based on BEIT (https://github.com/microsoft/unilm/tree/master/beit)
"""
class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


@PIPELINES.register_module()
class MaskedPositionGenerate:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            masked_tube_range=(0.5, 0.75), temporal_scale=1
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size
        self.masked_position_generator = MaskingGenerator(
            input_size, num_masking_patches=num_masking_patches,
            max_num_patches=max_num_patches,
            min_num_patches=min_num_patches,
        )
        self.masked_tube_range = masked_tube_range
        self.temporal_scale = temporal_scale

    def __call__(self, results):
        """Performs the Mask Generation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])
        mask_num = results['imgs'].shape[0]
        clip_len = results['clip_len'] // self.temporal_scale
        masked_tube = np.zeros((mask_num, clip_len, self.height, self.width), dtype=np.int)
        min_masked_tube_len, max_masked_tube_len = self.masked_tube_range
        min_masked_tube_len = max(int(min_masked_tube_len * clip_len), 1)
        max_masked_tube_len = min(int(max_masked_tube_len * clip_len), clip_len)
        for i in range(mask_num):
            mask = self.masked_position_generator()
            masked_tube_start = random.randint(0, clip_len - 1)
            masked_tube_len = random.randint(min_masked_tube_len, max_masked_tube_len)
            if masked_tube_start + masked_tube_len > clip_len:
                masked_tube[i, masked_tube_start:] = mask
                masked_tube[i, 0:masked_tube_start + masked_tube_len - clip_len] = mask
            else:
                masked_tube[i, masked_tube_start:masked_tube_start + masked_tube_len] = mask

        masked_tube = torch.from_numpy(masked_tube)
        results['input_position_masks'] = masked_tube
        results['output_position_masks'] = torch.nn.functional.interpolate(
            masked_tube.unsqueeze(1).to(torch.float32),
            (int(masked_tube.size(1) * self.temporal_scale), masked_tube.size(2), masked_tube.size(3)),
            mode='nearest'
        ).squeeze(1).to(torch.bool)
        return results


@PIPELINES.register_module()
class Masked2DPositionGenerate:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size
        self.masked_position_generator = MaskingGenerator(
            input_size, num_masking_patches=num_masking_patches,
            max_num_patches=max_num_patches,
            min_num_patches=min_num_patches,
        )

    def __call__(self, results):
        """Performs the Mask Generation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])
        mask_num = results['imgs'].shape[0]
        masked_tube = np.zeros((mask_num, self.height, self.width), dtype=np.int)
        for i in range(mask_num):
            masked_tube[i] = self.masked_position_generator()

        masked_tube = torch.from_numpy(masked_tube)
        results['input_position_masks'] = masked_tube
        results['output_position_masks'] = masked_tube
        return results

