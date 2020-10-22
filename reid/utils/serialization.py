import json
import os.path as osp
import shutil
from typing import Dict

import torch
from torch.nn import Parameter, Module

from .osutils import mkdir_if_missing


def read_json(fpath: string):
    """
    读取 json 文件
    """
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath: string):
    """
    写入 json 文件
    """
    mkdir_if_missing(osp.dirname(fpath))

    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best: bool, fpath='checkpoint.pth.tar'):
    """
    保存模型参数, 模型最优时进行复制
    """
    mkdir_if_missing(osp.dirname(fpath))

    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath: string):
    """
    加载模型参数
    """
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print(f'=> Loaded checkpoint "{fpath}"')
        return checkpoint
    else:
        raise ValueError(f'No checkpoint found at "{fpath}"')


def copy_state_dict(state_dict: Dict[string, Parameter], model: Module, strip=None):
    """
    将字典中的参数复制到模型中
    """
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys())-copied_names
    if len(missing) > 0:
        print('missing keys in state_dict:', missing)

    return model
