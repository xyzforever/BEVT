import math
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler as _DistributedSampler
import random
import threading


def pre_fetch(fn_fetch, index):
    fn_fetch(index)


class DistributedChunkSampler(torch.utils.data.Sampler):
    def __init__(self,
                 dataset,
                 chunk_sizes=None,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 shuffle_chunk=True):
        if num_replicas is None:
            try:
                num_replicas = dist.get_world_size()
            except Exception:
                num_replicas = torch.cuda.device_count()
        if rank is None:
            try:
                rank = dist.get_rank()
            except Exception:
                rank = torch.cuda.current_device()
        if chunk_sizes is None:
            chunk_sizes = [len(dataset)]
        if torch.cuda.is_available():
            self.gpus_per_node = torch.cuda.device_count()
        self.dataset = dataset
        self.num_replicas = num_replicas  # num of GPUs
        self.rank = rank  # GPU id
        self.chunk_sizes = chunk_sizes
        self.min_chunk_size = min(self.chunk_sizes) - (min(self.chunk_sizes) % self.gpus_per_node)
        self.epoch = 0
        self.num_samples = int(
            math.ceil(
                (len(self.chunk_sizes) * self.min_chunk_size) * 1.0 / self.num_replicas
            )
        )  # num of samples per GPU
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.shuffle_chunk = shuffle_chunk
        self.indices = None

    def _shuffle_chunk_elements(self, chunk_indices):
        """
        Generate randomly shuffled indices chunk-by-chunk.
        The generated indices are randomized in both chunk- and instance-level.

        Example::
        Input:
            chunk_size: [100, 100, 100, 100, 100]
            accum_chunk_sizes: [0, 100, 200, 300, 400, 500]
            chunk_indices: [1, 3, 2, 5, 4]
        Output:
            [12, 47, 29, ...
            283, 247, 212, ...
            192, 148, 183, ...
            482, 457, 431, ...
            314, 367, 352, ...]
        """
        accum_chunk_sizes = [0]
        for size in self.chunk_sizes:
            accum_chunk_sizes += [accum_chunk_sizes[-1] + size]

        # In case that the data size is greater than local cache (e.g., blobfuse),
        # reverse the order of consuming data between epochs to reduce the impact of cache miss.
        num_nodes = int(self.num_replicas / self.gpus_per_node)
        num_tsvs = int(len(chunk_indices) / num_nodes)
        if self.epoch % 2:
            for i in range(num_nodes):
                chunk_indices[i*num_tsvs:(i+1)*num_tsvs] = chunk_indices[
                    i*num_tsvs:(i+1)*num_tsvs][::-1]

        indices = []
        for idx in range(len(chunk_indices)):
            shuffled_chunk_elements = list(
                range(
                    accum_chunk_sizes[chunk_indices[idx] - 1],
                    accum_chunk_sizes[chunk_indices[idx]]
                )
            )
            random.shuffle(shuffled_chunk_elements)
            shuffled_chunk_elements = shuffled_chunk_elements[
                :self.min_chunk_size
            ]
            # insert tsv file index for pre-loading, skip the last tsv file
            if (idx+1) % num_tsvs:
                shuffled_chunk_elements[0] = (
                    shuffled_chunk_elements[0],
                    chunk_indices[min(idx + 1, len(chunk_indices) - 1)] - 1,
                    False
                )
            if idx % num_tsvs == 0:
                shuffled_chunk_elements[1] = (
                    shuffled_chunk_elements[1],
                    chunk_indices[idx] - 1,
                    True
                )
            indices += shuffled_chunk_elements

        return indices

    def __iter__(self):
        for item in self.indices:
            if isinstance(item, tuple):
                index = item[0]
                index_chunk = item[1]
                if item[2]:
                    pre_fetch(
                        self.dataset.fetch_blob,
                        index_chunk
                    )
                else:
                    x = threading.Thread(
                        target=pre_fetch,
                        args=(
                            self.dataset.fetch_blob,
                            index_chunk
                        ),
                        daemon=True
                    )
                    x.start()
            else:
                index = item
            yield index

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        # Deterministically shuffle based on epoch
        self.epoch = epoch
        random.seed(self.epoch)

        if self.shuffle:
            chunk_indices = list(range(1, len(self.chunk_sizes) + 1))
            if self.shuffle_chunk:
                random.shuffle(chunk_indices)
            self.indices = self._shuffle_chunk_elements(chunk_indices)
        else:
            self.indices = list(range(len(self.dataset)))
            self.indices = self.indices[:self.total_size]

        assert len(self.indices) == self.total_size, \
            'indices: {} vs total_size: {}'.format(
                len(self.indices), self.total_size
            )

        # Subsample
        rank = self.rank % self.gpus_per_node
        node_idx = int(self.rank / self.gpus_per_node)
        idx_start = self.gpus_per_node * node_idx * self.num_samples
        idx_end = self.gpus_per_node * (node_idx + 1) * self.num_samples
        self.indices = self.indices[idx_start:idx_end]
        idx_start = rank
        idx_end = self.num_samples * self.gpus_per_node
        idx_step = self.gpus_per_node
        self.indices = self.indices[idx_start:idx_end:idx_step]

        assert len(self.indices) == self.num_samples, \
            'indices: {} vs num_samples: {}'.format(
                len(self.indices), self.num_samples
            )


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    In pytorch of lower versions, there is no ``shuffle`` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class ClassSpecificDistributedSampler(_DistributedSampler):
    """ClassSpecificDistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    Samples are sampled with a class specific probability, which should be an
    attribute of the dataset (dataset.class_prob, which is a dictionary that
    map label index to the prob). This sampler is only applicable to single
    class recognition dataset. This sampler is also compatible with
    RepeatDataset.

    The default value of dynamic_length is True, which means we use
    oversampling / subsampling, and the dataset length may changed. If
    dynamic_length is set as False, the dataset length is fixed.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 dynamic_length=True,
                 shuffle=True,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

        if type(dataset).__name__ == 'RepeatDataset':
            dataset = dataset.dataset

        assert hasattr(dataset, 'class_prob')

        self.class_prob = dataset.class_prob
        self.dynamic_length = dynamic_length
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        class_indices = defaultdict(list)

        # To be compatible with RepeatDataset
        times = 1
        dataset = self.dataset
        if type(dataset).__name__ == 'RepeatDataset':
            times = dataset.times
            dataset = dataset.dataset
        for i, item in enumerate(dataset.video_infos):
            class_indices[item['label']].append(i)

        if self.dynamic_length:
            indices = []
            for k, prob in self.class_prob.items():
                prob = prob * times
                for i in range(int(prob // 1)):
                    indices.extend(class_indices[k])
                rem = int((prob % 1) * len(class_indices[k]))
                rem_indices = torch.randperm(
                    len(class_indices[k]), generator=g).tolist()[:rem]
                indices.extend(rem_indices)
            if self.shuffle:
                shuffle = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in shuffle]

            # re-calc num_samples & total_size
            self.num_samples = math.ceil(len(indices) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # We want to keep the dataloader length same as original
            video_labels = [x['label'] for x in dataset.video_infos]
            probs = [
                self.class_prob[lb] / len(class_indices[lb])
                for lb in video_labels
            ]

            indices = torch.multinomial(
                torch.Tensor(probs),
                self.total_size,
                replacement=True,
                generator=g)
            indices = indices.data.numpy().tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # retrieve indices for current process
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
