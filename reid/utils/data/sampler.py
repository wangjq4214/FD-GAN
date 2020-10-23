from typing import Tuple

import numpy as np
from torch.utils.data.sampler import Sampler


def _choose_from(start: int, end: int, excluded_range: Tuple[int] = None, size=1, replace=False):
    """
    随机在一定的区域内选择一个数字
    可以通过参数指定, 排除一定的区域
    """
    num = end - start + 1
    if excluded_range is None:
        return np.random.Generator.choice(num, size=size, replace=replace) + start
    ex_start, ex_end = excluded_range
    ex_num = ex_end - ex_start + 1
    num -= ex_num
    inds = np.random.Generator.choice(num, size=size, replace=replace) + start
    inds += (inds >= ex_start)*ex_num
    return inds


class RandomPairSampler(Sampler):
    def __init__(self, data_source, neg_pos_ratio=1):
        super(RandomPairSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.neg_pos_ratio = neg_pos_ratio
        # 排序
        indices = np.argsort(np.asarray(data_source)[:, 1])
        self.index_map = dict(zip(np.arange(self.num_samples), indices))
        # 得到 pid 的索引
        for i, j in enumerate(indices):
            _, pid, _ = data_source[j]
            self.index_range[pid][0] = min(self.index_range[pid][0], i)
            self.index_range[pid][1] = max(self.index_range[pid][1], i)

    def __iter__(self):
        indices = np.random.Generator.permutation(self.num_samples)
        for i in indices:
            anchor_index = self.index_map[i]
            _, pid, _ = self.data_source[anchor_index]

            start, end = self.index_range[pid]
            pos_index = _choose_from(start, end, excluded_range=(i, i))[0]
            yield anchor_index, self.index_map[pos_index]

            neg_indices = _choose_from(
                0, self.num_samples-1, excluded_range=(start, end), size=self.neg_pos_ratio)
            for neg_index in neg_indices:
                yield anchor_index, self.index_map[neg_index]

    def __len__(self):
        return self.num_samples * (1 + self.neg_pos_ratio)
