from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
import functools

np.random.seed(0)

class PoseGenerator(Dataset):
    def __init__(self, poses_3d, poses_2d, actions, subset=1):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._actions = functools.reduce(lambda x, y: x + y, actions)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        print('Generating {} poses...'.format(len(self._actions)))
        if subset < 1:
            total_num = self._poses_3d.shape[0]
            subset_idx = np.random.choice(total_num, int( subset * total_num), replace=False)
            self._poses_3d = self._poses_3d[subset_idx]
            self._poses_2d = self._poses_2d[subset_idx]
            self._actions = list(np.array(self._actions)[subset_idx]) #self._actions[subset_idx]
            print('Subsetting!!: Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        out_action = self._actions[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d, out_action

    def __len__(self):
        return len(self._actions)
