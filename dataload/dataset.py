import torch.utils.data as data

from PIL import Image
import os
import numpy as np
import torchvision
from numpy.random import randint
import torch
from torch.utils.data import DataLoader, Dataset
from dataload.simple_dataset import SimpleVideoDataset


class SetDataset:
    '''
    SetDataset用于few-shot的数据加载。它首先将meta-train/meta-val/meta-test中的所有数据(image, video)按类别放到self.sub_meta:dict
    中，self.sub_meta[i]对应类别i的所有视频，它是一个list，然后利用sub_meta来创建SubDataset类，每个类别对应创建一个Dataset。然后，根据每
    个Dataset创建一个Dataloader，batch_size的大小等于support+query(episode per class)的数目。对SetDataset[i]索引，可以返回类别i的
    一个episode的数据，len(SetDataset)就是整个meta-train/val/test的类别数目。
    '''
    def __init__(self, list_file, batch_size):
        self.list_file = list_file
        self.sub_meta = {}

        self.all_videos, self.labels = [], []
        for line in open(list_file, 'r'):
            video_path, label = line.split(' ')
            label = int(label)
            self.all_videos.append(video_path)
            self.labels.append(label)

        self.cl_list = np.arange(len(np.unique(self.labels))) # cls#1, cls#2, ..., cls#n
        for cl in self.cl_list:
            self.sub_meta[cl] = []
        for video, label in zip(self.all_videos, self.labels):
            self.sub_meta[label].append(video)

        self.sub_dataloader = [] # 对每个类别都有一个dataloader
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=4,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, )
            self.sub_dataloader.append(DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, item):
        return next(iter(self.sub_dataloader[item]))

    def __len__(self):
        return len(self.cl_list)




class SubDataset(SimpleVideoDataset):
    '''
    根据类别建立这个类别下的dataset
    '''
    def __init__(self, sub_meta, cl, split='train', clip_len=16, min_size=50):

        self.all_videos = sub_meta   # TODO: the format of sub_meta contains
        if len(self.all_videos) < min_size: # 如果说这个类别下的样本数量少于50，那么我们补充一些重复的样本，直到达到50个
            idxs = [i % len(self.all_videos) for i in range(min_size)]  # at least 50 samples in each class
            self.sub_meta = np.array(self.sub_meta[idxs]).tolist()

        self.labels = np.repeat(cl, len(self.all_videos))
        self.split = split
        self.clip_len = clip_len


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

if __name__ == '__main__':
    list_file = '/opt/data/private/DL_Workspace/fsv-baseline/datafile/hmdb51/train_subtrain.txt'
    setdataset = SetDataset(list_file, batch_size=10)
    print(setdataset[0])