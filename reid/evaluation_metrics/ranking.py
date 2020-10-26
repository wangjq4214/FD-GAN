import numpy as np

from collections import defaultdict
from sklearn.metrics import average_precision_score

from ..utils import to_numpy


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    distmat = to_numpy(distmat)
    m, n = distmat.shape

    # 填充默认值
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)

    # 确保是 numpy 数组
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)

    # 排序并找到正确的匹配
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    # 计算 CMC
    ret = np.zeros(topk)
    num_vaild_queries = 0
    for i in range(m):
        # 过滤相同的 id 和 摄像机
        vaild = ((gallery_ids[indices[i]] != query_ids[i])
                 | gallery_cams[indices[i]] != query_cams[i])
        if separate_camera_set:
            vaild &= (gallery_cams[indices[i]] != query_cams[i])

        if not np.any(matches[i, vaild]):
            continue

        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][vaild]]
            inds = np.where(vaild)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1

        for _ in range(repeat):
            if single_gallery_shot:
                sampled = (vaild & _unique_sample(ids_dict, len(vaild)))
                index = np.nonzero(matches[i], sampled)[0]
            else:
                index = np.nonzero(matches[i, vaild])[0]

            delta = 1./(len(index)*repeat)
            for j, k in enumerate(index):
                if k-j >= topk:
                    break
                if first_match_break:
                    ret[k-j] += 1
                    break
                ret[k-j] += delta

        num_vaild_queries += 1

    if num_vaild_queries == 0:
        raise RuntimeError("No vaild query")

    return ret.cumsum()/num_vaild_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    distmat = to_numpy(distmat)
    m, n = distmat.shape

    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)

    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        aps.append(average_precision_score(y_true, y_score))

    if len(aps) == 0:
        raise RuntimeError("No valid query")

    return np.mean(aps)
