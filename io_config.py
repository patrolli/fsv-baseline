import argparse
import os
import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--k_shot', default=2, type=int)
    parser.add_argument('--q_query', default=2, type=int)
    parser.add_argument('--resume', action='store_true', help='resume to train')
    parser.add_argument('--epi_train', action='store_true', default=True, help='use episode training or use standard softmax training')
    parser.add_argument('--start_epoch', default=0)
    parser.add_argument('--stop_epoch', default=100, help='for episode training, each epoch contains 100 episodes')
    parser.add_argument('--save_freq', default=10, help='save model in each save_freq epochs')
    parser.add_argument('--lr', default=0.01, help='set the learning rate')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--dataset', default='hmdb51', choices=['hmdb51', 'ucf101'])
    parser.add_argument('--batch_size', default=64, type=int)
    return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
