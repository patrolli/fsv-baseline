import sys
import torch
import torch.nn as nn
import torch.optim as optim
from models.C3D import C3D
from models.MENet import FS_ResNet, FS_MENet
from torch.utils.data import DataLoader, Dataset
from dataload.simple_dataset import SimpleVideoDataset
from dataload.datamgr import SetDataManager
import os
from io_config import *
import numpy as np
import time
from apex import amp

save_freq = 10
print_freq = 20
learning_rate = 0.01
dataset = 'hmdb51'
optimization = 'SGD' # choices: [SGD, Adam]
train_classes = 31 if dataset == 'hmdb51' else 10  # TODO: ucf101
resume = True

args = parse_args()

# 加载数据
if args.epi_train is True:
    train_file = '/opt/data/private/DL_Workspace/fsv-baseline/datafile/{}/few-shot-train.txt'.format(args.dataset)
    val_file = '/opt/data/private/DL_Workspace/fsv-baseline/datafile/{}/few-shot-val.txt'.format(args.dataset)
    tr_data_mgr = SetDataManager(args.n_way, args.k_shot, args.q_query)
    tr_loader = tr_data_mgr.get_data_loader(train_file)
    val_data_mgr = SetDataManager(args.n_way, args.k_shot, args.q_query, n_episode=30)
    val_loader = val_data_mgr.get_data_loader(val_file)
else:
    train_file = '/opt/data/private/DL_Workspace/fsv-baseline/datafile/{}/train_subtrain.txt'
    val_file = '/opt/data/private/DL_Workspace/fsv-baseline/datafile/{}/train_subval.txt'
    tr_dataset = SimpleVideoDataset(train_file, split='train', clip_len=8)
    val_dataset = SimpleVideoDataset(val_file, split='val', clip_len=8)
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

# 建立模型
if args.model == 'FS_ResNet':
    model = FS_ResNet(args.k_shot, args.backbone_size)
elif args.model == 'FS_MENet':
    model = FS_MENet(args.k_shot, args.backbone_size)
elif args.model == 'C3D':
    model = C3D(train_classes, pooling=args.pooling)
    args.backbone_size = ''
else:
    raise NotImplementedError("Not implement such a model")
model = model.cuda()
if args.optim == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = optim.SGD([
    #     {'params': model.resnet.parameters(), 'lr': args.lr/10.0},
    #     {'params': model.menet.parameters(), 'lr': args.lr/10.0}
    # ])
elif args.optim == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = None
    raise NotImplementedError("Not support for this optim:{}".format(args.optim))
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

# 模型保存，导入相关，日志
save_dir = './chechpoints'
log_dir = './log'

save_name = '{}_{}_{}_lr_{}_epi_{}_epoch_{}'.format(
    args.model+str(args.backbone_size), dataset, optimization, args.lr, args.epi_train, args.stop_epoch) + args.posifix
if not os.path.exists(os.path.join(save_dir, save_name)):
    os.makedirs(os.path.join(save_dir, save_name))
if not os.path.exists(os.path.join(log_dir, save_name)):
    os.makedirs(os.path.join(log_dir, save_name))

log_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
log_file = open(os.path.join(log_dir, save_name, save_name+log_time+'.txt'), 'w')

save_dir = os.path.join(save_dir, save_name)  # 这两句似乎没什么作用啊?
log_dir = os.path.join(log_dir, save_name)  # TODO: add local time to logfile name

if args.resume:
    resume_file = get_resume_file(save_dir)
    if resume_file is not None:
        tmp = torch.load(resume_file)
        start_epoch = tmp['epoch']+1
        model.load_state_dict(tmp['state'])

# 训练模型
max_acc = 0

for epoch in range(args.start_epoch, args.stop_epoch):
    model.train()
    avg_loss = 0
    for i, (x, _) in enumerate(tr_loader):
        x = x.cuda()
        y = torch.from_numpy(np.repeat(range(args.n_way), args.q_query)).cuda()
        logit = model(x)
        loss = loss_fn(logit, y)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 200)
        optimizer.step()
        avg_loss = avg_loss + loss.item()
        if (i+1) % print_freq == 0:
            print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i+1, len(tr_loader), avg_loss/float(i+1)))
            log_file.write('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}\n'.format(epoch, i+1, len(tr_loader), avg_loss/float(i+1)))
            print(logit)
            print(calc_grad_norm(model))
    test_iter_num = len(val_loader)
    acc_all = []
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(val_loader):
            x = x.cuda()
            y = torch.from_numpy(np.repeat(range(args.n_way), args.q_query)).cuda()
            score = model(x)
            prediction = torch.argmax(score, dim=1)
            correct = (prediction == y).sum().float()
            acc = (correct / len(y)).detach().cpu().numpy() * 100
            acc_all.append(acc)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (test_iter_num, acc_mean, 1.96 * acc_std / np.sqrt(test_iter_num)))
        log_file.write('%d Test Acc = %4.2f%% +- %4.2f%%\n' % (test_iter_num, acc_mean, 1.96 * acc_std / np.sqrt(test_iter_num)))

    if acc_mean > max_acc:
        print('best model! save...')
        log_file.write('best model! save...\n')
        max_acc = acc_mean
        outfile = os.path.join(save_dir, 'best_model.tar')
        torch.save({'epoch': epoch, 'state': model.state_dict(), 'max_acc': max_acc}, outfile)

    if (epoch % args.save_freq==0) or (epoch==args.stop_epoch-1):
        outfile = os.path.join(save_dir, '{:d}.tar'.format(epoch))
        torch.save({'epoch':epoch, 'state':model.state_dict(), 'max_acc': max_acc}, outfile)
    # scheduler.step()
print(max_acc)
log_file.write('best acc is: {}\n'.format(max_acc))
log_file.close