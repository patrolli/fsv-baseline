import os
import glob
def count_file(root_path):
    # root_path: '/opt/data/private/FSL_Datasets/HMDB_51_V2'
    # 统计train dataset下面每个class的视频数目，以及每个视频包含的图片（帧）数目
    split_file_path = './datafile/hmdb51/hmdb51_train_split.txt'
    with open(split_file_path, 'r') as f:
        classes = [x.rstrip() for x in f.readlines()]
    with open('./hmdb_train_statistics.txt', 'w') as f:
        for video_class in classes:
            video_class_path = os.path.join(root_path, video_class)
            video_num_per_class = len(os.listdir(video_class_path))
            f.write('{}: {} videos\n'.format(video_class, video_num_per_class))
        for video_class in classes:
            print('process class: {}'.format(video_class))
            video_class_path = os.path.join(root_path, video_class)
            for video in os.listdir(video_class_path):
                frame_num_per_video = len(os.listdir(os.path.join(video_class_path, video)))
                f.write('class: {} video: {} frames: {}\n'.format(video_class, video, frame_num_per_video))
    print('over~')

def split_train(root_path):
    # split source train dataset as sub-train and sub-val
    # this holds for pretraining method, which performs normal classification task to train feature extractor
    # root_path: '/opt/data/private/FSL_Datasets/HMDB_51_V2'
    split_file_path = './datafile/hmdb51/hmdb51_train_split.txt'
    with open(split_file_path, 'r') as f:
        classes = [x.rstrip() for x in f.readlines()]
    with open('./datafile/hmdb51/train_subtrain.txt', 'w') as ftrain, open('./datafile/hmdb51/train_subval.txt', 'w') as fval:
        for i, video_class in enumerate(classes):
            print('processing class: {}...'.format(video_class))
            videos = os.listdir(os.path.join(root_path, video_class))
            train_num = len(videos) * 4 // 5  # ratio 8:2 to split the source train dataset as sub-train and sub-val
            train_videos = videos[: train_num]
            val_videos = videos[train_num: ]
            list(map(lambda x: ftrain.write('{} {}\n'.format(x, i)),
                [os.path.join(root_path, video_class, x) for x in train_videos]))
            list(map(lambda x: fval.write('{} {}\n'.format(x, i)),
                [os.path.join(root_path, video_class, x) for x in val_videos]))



if __name__ == '__main__':
    root_path = '/opt/data/private/FSL_Datasets/HMDB_51_V2'
    # count_file('/opt/data/private/FSL_Datasets/HMDB_51_V2')
    split_train(root_path)