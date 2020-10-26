import pathlib
import random
import torch

import numpy as np

from os import path

from PIL import Image
from scipy import ndimage
from torchvision.transforms import Normalize, Compose, ToTensor, RandomHorizontalFlip
from .transforms import RectScale, RandomSizedEarser


class Preprocessor:
    """
    图片预处理
    """

    def __init__(self, dataset, root=None, with_pose=False, pose_root=None, pid_imgs=None, height=256, width=128, pose_aug='no', transform=None):
        self.dataset = dataset
        self.root = root
        self.with_pose = with_pose
        self.pose_root = pose_root
        self.pid_imgs = pid_imgs
        self.height = height
        self.width = width
        self.pose_aug = pose_aug

        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[
                              0.229, 0.224, 0.225])

        # 图片增广
        if transform is None:
            self.transform = Compose([
                RectScale(height, width),
                RandomSizedEarser(),
                RandomHorizontalFlip(),
                ToTensor,
                normalize
            ])
        else:
            self.transform = transform

        # pose 的处理
        self.transform_p = Compose([
            RectScale(height, width),
            ToTensor,
            normalize
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            if not self.with_pose:
                return [self._get_single_item(index) for index in indices]
            else:
                return [self._get_single_item_with_pose(index) for index in indices]

        if not self.with_pose:
            return self._get_single_item(indices)
        else:
            return self._get_single_item_with_pose(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = str(pathlib.Path.joinpath(self.root, fname))

        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)

        return img, fname, pid, camid

    def _get_single_item_with_pose(self, index):
        fname, pid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = str(pathlib.Path.joinpath(self.root, fname))
        img = Image.open(fpath).convert('RGB')

        img = self.transform(img)

        pid_query = list(self.pid_imgs[pid])
        if fname in pid_query and len(pid_query) > 1:
            pid_query.remove(fname)
        pname = path.splitext(random.choice(pid_query))[0]

        ppath = pname+'.txt'
        if self.root is not None:
            ppath = str(pathlib.Path.joinpath(self.pose_root, ppath))
        gtpath = pname+'.jpg'
        if self.root is not None:
            gtpath = str(pathlib.Path.joinpath(self.pose_root, gtpath))

        gt_img = Image.open(gtpath).convert('RGB')
        landmark = self._load_landmark(
            ppath, self.height/gt_img.size[1], self.width/gt_img.size[0])

        maps = self._generate_pose_map(landmark)

        flip_flag = random.choice([True, False])
        if flip_flag:
            gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)
            maps = np.flip(maps, 2)

        maps = torch.from_numpy(maps.copy()).float()
        gt_img = self.transform_p(gt_img)

        return {
            'origin': img,
            'target': gt_img,
            'posemap': maps,
            'pid': torch.LongTensor([pid])
        }

    def _load_landmark(self, path, scale_h, scale_w):
        landmark = []
        with open(path, 'r') as f:
            landmark_file = f.readlines()

        for line in landmark_file:
            line1 = line.strip().split(' ')
            h0 = int(float(line1[0])*scale_h)
            w0 = int(float(line1[1])*scale_w)

            if h0 < 0:
                h0 = -1
            if w0 < 0:
                w0 = -1
            landmark.append(torch.Tensor([[h0, w0]]))

        landmark = torch.cat(landmark).long()
        return landmark

    def _generate_pose_map(self, landmark, gauss_sigma=5):
        maps = []
        randnum = landmark.size(0)+1
        if self.pose_aug == 'erase':
            randnum = random.randrange(landmark.size(0))
        elif self.pose_aug == 'gauss':
            gauss_sigma = random.randint(gauss_sigma-1, gauss_sigma+1)
        elif self.pose_aug != 'no':
            assert (
                'Unknown landmark augmentation method, choose from [no|erase|gauss]')

        for i in range(landmark.size(0)):
            map = np.zeros([self.height, self.width])
            if landmark[i, 0] != -1 and landmark[i, 1] != -1 and i != randnum:
                map[landmark[i, 0], landmark[i, 1]] = 1
                map = ndimage.filters.gaussian_filter(map, sigma=gauss_sigma)
                map = map/map.max()
            maps.append(map)

        maps = np.stack(maps, axis=0)
        return maps
