from dataload.dataset import SetDataset, EpisodicBatchSampler
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
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

    def get_data_loader(self, list_file):
        dataset = SetDataset(list_file, self.batch_size)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler)
        data_loader = DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    # 创建并使用dataloader的示例
    datamgr = SetDataManager(n_way=5, k_shot=2, q_query=5)
    list_file = '/opt/data/private/DL_Workspace/fsv-baseline/datafile/hmdb51/train_subtrain.txt'
    loader = datamgr.get_data_loader(list_file)
    x, y = iter(loader).__next__()
    print(y)
    print(x.size(), y.size())  # expect output of x: (n_way, k_shot+q_query, C, n_frame, H, W)