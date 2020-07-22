import torch.utils.data as data

from PIL import Image
import os
import numpy as np
import torchvision
from numpy.random import randint
import torch
from torch.utils.data import DataLoader, Dataset

# 用于封装视频内容，保存每个视频的信息，包括图片路径、frames的数量、标签信息
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,  # TODO：这里的list_file是什么东西
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length  # TODO：new_length是什么东西
        # self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        # self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        # if self.modality == 'RGBDiff':
        #     self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    # 读取图片，给定路径和索引idx，返回读取的图片的list。事实上，对于rgb输入，这里一个list只存放了一张图片，对于光流，则存在两张图片(x, y)
    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= 3]

        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    # 从视频里抽取若干frame，返回frame的索引的列表
    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        self.num_segments用于指定从视频中抽取几张图片（frames），具体的抽取方法分成两种：dense_sample和normal sample。
        dense_sample--首先预留一段长度（这里是64帧），从前面[0, record.num_frames-64]帧里随机选择一个起始帧，然后以固定步长=64/num_segments
        抽取后续帧，这里预留的64帧以及步长64/num_segments保证了在抽取帧的索引不会超出视频帧数的范围
        normal_sample--首先将视频帧分成num_segments段，如num_segments=3，那么就将视频帧分成3段，然后在这每一小段中随机选取一帧。
        从时间维度上看，dense_sample抽取的帧在时间上更加密集，而normal_sample更加稀疏
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1) # 从第64帧开始采样
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]  # 根据索引idx取出视频的record
        # check this is a legit video folder
        segment_indices = self._sample_indices(record)
        print(segment_indices, record.num_frames)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            seg_imgs = self._load_image(record.path, p)  # seg_imgs是一个list，对于rgb，只有一个元素，对于光流，有两张图片
            images.extend(seg_imgs)
            if p < record.num_frames:  #TODO:这个if有什么作用？
                p += 1
        if self.transform is not None:
            process_data = self.transform(images)  # 对从一个视频中抽出的一组图片来做data transform
        else:
            process_data = images
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

class SetDataset:
    '''
    SetDataset用于few-shot的数据加载。它首先将meta-train/meta-val/meta-test中的所有数据(image, video)按类别放到self.sub_meta:dict
    中，self.sub_meta[i]对应类别i的所有视频，它是一个list，然后利用sub_meta来创建SubDataset类，每个类别对应创建一个Dataset。然后，根据每
    个Dataset创建一个Dataloader，batch_size的大小等于support+query(episode per class)的数目。对SetDataset[i]索引，可以返回类别i的
    一个episode的数据，len(SetDataset)就是整个meta-train/val/test的类别数目。
    '''
    def __init__(self, list_file, batch_size, transform):
        self.list_file = list_file
        self.sub_meta = {}
        self._parse_list()
        self.cl_list = np.arange(len(np.unique([item.label for item in self.video_list]))) # cls#1, cls#2, ..., cls#n
        for cl in self.cl_list:
            self.sub_meta[cl] = []
        for item in self.video_list:
            self.sub_meta[item.label].append(item)

        self.sub_dataloader = [] # 对每个类别都有一个dataloader
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform, dense_sample=False)
            self.sub_dataloader.append(DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, item):
        return next(iter(self.sub_dataloader[item]))

    def __len__(self):
        return len(self.cl_list)

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= 3]

        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))


class SubDataset(Dataset):
    '''
    根据类别建立这个类别下的dataset
    '''
    def __init__(self,
                 sub_meta,
                 cl,
                 transform=None,
                 min_size=50,
                 dense_sample=False,
                 image_tmpl='img_{:05d}.jpg',
                 num_segments=3,
                 new_length=1):
        self.sub_meta = sub_meta   # TODO: the format of sub_meta contains
        self.cl = cl
        self.transform = transform
        self.dense_sample = dense_sample
        self.image_tmpl = image_tmpl
        self.num_segments = num_segments
        self.new_length = new_length
        if len(self.sub_meta) < min_size: # 如果说这个类别下的样本数量少于50，那么我们补充一些重复的样本，直到达到50个
            idxs = [i % len(self.sub_meta) for i in range(min_size)]  # at least 50 samples in each class
            self.sub_meta = np.array(self.sub_meta[idxs]).tolist()

    # 读取图片，给定路径和索引idx，返回读取的图片的list。事实上，对于rgb输入，这里一个list只存放了一张图片，对于光流，则存在两张图片(x, y)
    def _load_image(self, video_path, idx):
        return [Image.open(os.path.join(video_path, self.image_tmpl.format(idx))).convert('RGB')]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1) # 从第64帧开始采样
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def __getitem__(self, i):
        record = self.sub_meta[i] # self.sub_meta[i]应包含完整的视频路径
        segment_indices = self._sample_indices(record)
        # print(segment_indices, record.num_frames)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            seg_imgs = self._load_image(record.path, p)  # seg_imgs是一个list，对于rgb，只有一个元素，对于光流，有两张图片
            images.extend(seg_imgs)
            if p < record.num_frames:  #TODO:这个if有什么作用？
                p += 1
        if self.transform is not None:
            process_data = self.transform(images)  # 对从一个视频中抽出的一组图片来做data transform
        else:
            process_data = images
        return process_data, record.label

    def __len__(self):
        return len(self.sub_meta)


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
    root_path = '/opt/data/private/FSL_Datasets/HMDB_51_frames'
    list_file = root_path + '/hmdb51_train_videofolder.txt'
    # from transforms import *
    # transform = torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66]),
    #                                                    GroupRandomHorizontalFlip(is_flow=False)])
    dataset = TSNDataSet(root_path, list_file, num_segments=3, dense_sample=True)

    print(dataset[0])