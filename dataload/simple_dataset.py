import os

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import sys

class SimpleVideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, root_file, split='train', clip_len=16):
        # dataset: [ucf101, hmdb51]
        self.clip_len = clip_len
        self.split = split
        folder = root_file

        # The following three parameters are chosen as described in the paper section 4.1
        self._resize_height = 128
        self._resize_width = 128
        self._crop_size = 128

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.all_videos, self.labels = [], []
        for line in open(folder, 'r'):
            video_path, label = line.split(' ')
            label = int(label)
            self.all_videos.append(video_path)
            self.labels.append(label)

    @property
    def resize_height(self):
        return 128

    @property
    def resize_width(self):
        return 128

    @property
    def crop_size(self):
        return 128

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.all_videos[index])  # get a video frames folder and load it as a ndarray
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.labels[index])

        if self.split == 'test' or self.split == 'val':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)  # TODO: I want to check the normalize result
        buffer = self.to_tensor(buffer)  # transpose the dimention as (C, n_frames, H, W)
        return torch.from_numpy(buffer), torch.from_numpy(labels)


    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            # frame -= np.array([[[90.0, 98.0, 102.0]]])
            # TODO:i'm not sure this normalize code will make sense
            frame = frame / 255
            frame = (frame - np.array([[[0.485, 0.456, 0.406]]])) / np.array([[[0.229, 0.224, 0.225]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        # buffer represent a video, which size is (n_frames, H, W, C)
        # note this will read all frames into buffer, not crop on the time dimension
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        # clip_len represent the final frame length of a video
        time_index = np.random.randint(buffer.shape[0] - clip_len)  # TODO: not sure whether use time crop or not
        # TODO: I just crop the time dimension, for the spatial size is 128, which has no need to crop
        # I'm not sure this will derease the performance or not
        # Randomly select start indices in order to crop the video
        # height_index = np.random.randint(buffer.shape[1] - crop_size)
        # width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len]

        return buffer


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = SimpleVideoDataset(root_file='/opt/data/private/FSL_Datasets/HMDB_51_V2', split='train', clip_len=8)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    # expected output for input: (100, 3, 8, 112, 112)
    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)
        print(inputs)
        if i == 1:
            break