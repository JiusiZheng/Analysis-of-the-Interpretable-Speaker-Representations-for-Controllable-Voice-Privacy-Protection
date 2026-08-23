from torch.utils import data
import torch
import math
import numpy as np
import random
from torch.utils.data import BatchSampler

def StandardSampler(dataset, shuffle, distributed=False,
                    world_size=None, rank=None):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle,
                                                   num_replicas=world_size, rank=rank)
    if shuffle:
        return data.RandomSampler(dataset)
    return data.SequentialSampler(dataset)

def RandomBucketSampler(nbuckets, length, batch_size, drop_last, distributed=False,
                        world_size=None, rank=None):
    if distributed:
        return DistributedRandomBucketSampler(nbuckets, length, batch_size, drop_last, world_size, rank)
    return SingleRandomBucketSampler(nbuckets, length, batch_size, drop_last)

class SingleRandomBucketSampler(data.Sampler):
    def __init__(self, nbuckets, length, batch_size, drop_last):
        self.length = length
        self.batch_size = batch_size
        self.drop_last = drop_last
        indices = np.argsort([-x for x in length])
        split = len(indices) // nbuckets
        self.indices = []
        for i in range(nbuckets):
            self.indices.append(indices[i*split:(i+1)*split])
        if nbuckets * split < len(length):
            self.indices.append(indices[nbuckets*split:])

    def __iter__(self):
        random.shuffle(self.indices)
        for x in self.indices:
            random.shuffle(x)
        idxs = [i for x in self.indices for i in x]
        batches, batch, sum_len, max_len = [], [], 0, 0
        for idx in idxs:
            batch.append(idx)
            sum_len += self.length[idx]
            max_len = max(self.length[idx], max_len)
            if max_len * len(batch) > self.batch_size:
                batches.append(batch[:-1])
                batch, sum_len, max_len = [batch[-1]], self.length[idx], self.length[idx]
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        # print('############################')
        # # print(self.drop_last)
        #print(len(batches)) ### len(batches) == 765
        # print('############################')
        random.shuffle(batches)
        return iter(batches)



### Modified for multi-GPU training
# class DistributedRandomBucketSampler(BatchSampler):
#     def __init__(self, nbuckets, length, batch_size,
#                  drop_last, num_replicas, rank, seed=1234, sampler=None):
#         if rank >= num_replicas or rank < 0:
#             raise ValueError(
#                 "Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1)
#             )
#         indices = np.argsort(length)
#         split = len(indices) // nbuckets
#         self.length = length
#         self.batch_size = math.ceil(batch_size / num_replicas)  # <-- 修改这里
#         self.indices = []
#         self.drop_last = drop_last
#         self.sampler = sampler
#         for i in range(nbuckets):
#             self.indices.append(indices[i*split:(i+1)*split])
#         if nbuckets * split < len(length):
#             self.indices.append(indices[nbuckets*split:])
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.seed = seed

#     def __iter__(self):
#         # Deterministic shuffling
#         random.Random(self.epoch + self.seed).shuffle(self.indices)
#         for i, x in enumerate(self.indices):
#             seed = self.epoch + self.seed + i * 5
#             random.Random(seed).shuffle(x)
#         indices = [i for x in self.indices for i in x]

#         # Batching
#         batches, batch, sum_len, max_len = [], [], 0, 0
#         for idx in indices:
#             batch.append(idx)
#             sum_len += self.length[idx]
#             max_len = max(self.length[idx], max_len)
#             if max_len * len(batch) > self.batch_size:
#                 batches.append(batch[:-1])
#                 batch, sum_len, max_len = [batch[-1]], self.length[idx], self.length[idx]

#         # 分布式切分，每张 GPU 只处理自己 batch
#         batches = batches[self.rank::self.num_replicas]  # <-- 修改这里
#         print(len(batches),'###',self.batch_size)
#         # Stochastic shuffle
#         random.shuffle(batches)
#         return iter(batches)

#     def set_epoch(self, epoch):
#         self.epoch = epoch

class DistributedRandomBucketSampler(BatchSampler):
    def __init__(self, nbuckets, length, batch_size,
                 drop_last, num_replicas, rank, seed=1234, sampler=None):
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        indices = np.argsort(length)
        split = len(indices) // nbuckets
        self.length = length
        self.batch_size = batch_size
        self.sampler = sampler
        self.drop_last = drop_last
        self.indices = []
        for i in range(nbuckets):
            self.indices.append(indices[i*split:(i+1)*split])
        if nbuckets * split < len(length):
            self.indices.append(indices[nbuckets*split:])
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        #Deterministic shuffling
        random.Random(self.epoch + self.seed).shuffle(self.indices)
        for i, x in enumerate(self.indices):
            seed = self.epoch + self.seed + i * 5
            random.Random(seed).shuffle(x)
        indices = [i for x in self.indices for i in x]

        #Batching
        batches, batch, sum_len, max_len = [], [], 0, 0
        for idx in indices:
            batch.append(idx)
            sum_len += self.length[idx]
            max_len = max(self.length[idx], max_len)
            # print('#############################')
            # print('max_len',max_len,'len(batch)',len(batch))
            # print('batch[:-1]',batch[:-1])
            # print('#############################')
            if max_len * len(batch) > self.batch_size:
                batches.append(batch[:-1])
                batch, sum_len, max_len = [batch[-1]], self.length[idx], self.length[idx]
        # subsample
        num_samples = math.ceil((len(batches) - self.num_replicas) / self.num_replicas)
        total_size = num_samples * self.num_replicas
        batches = batches[:total_size]
        batches = batches[self.rank*num_samples: (self.rank+1)*num_samples]
        assert len(batches) == num_samples

        #print(len(batches),'###',num_samples) #190
        #Stochastic suffling
        random.shuffle(batches)
        return iter(batches)

    def set_epoch(self, epoch):
        self.epoch = epoch

