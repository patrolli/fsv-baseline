from dataload.dataset import SetDataset, EpisodicBatchSampler
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from dataload.transforms import *
import torchvision


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass

class SetDataManager(DataManager):
    def __init__(self, n_way, k_shot, q_query, n_episode=100):
        super(SetDataManager, self).__init__()
        self.n_way = n_way
        self.batch_size = k_shot + q_query
        self.n_episode = n_episode

    def get_data_loader(self, data_file, aug):
        transform = aug
        dataset = SetDataset(data_file, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler)
        data_loader = DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    # 创建并使用dataloader的示例
    input_size = 224
    transform = torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                                GroupRandomHorizontalFlip(is_flow=False),
                                                Stack(),
                                                ToTorchFormatTensor(),
                                                GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_file = '/opt/data/private/FSL_Datasets/HMDB_51_frames/hmdb51_train_videofolder.txt'
    datamgr = SetDataManager(n_way=5, k_shot=2, q_query=1)
    loader = datamgr.get_data_loader(data_file, transform)
    print(iter(loader).__next__())