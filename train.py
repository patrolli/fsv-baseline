import sys
import torch
import torch.nn as nn
import torch.optim as optim
from models.C3D import C3D
from torch.utils.data import DataLoader, Dataset
from dataload.simple_dataset import SimpleVideoDataset
import os
from io_config import *
import numpy as np
import time

stop_epochn = 100
start_epoch = 0
save_freq = 10
print_freq = 20
learning_rate = 0.01
dataset = 'hmdb51'
optimization = 'SGD' # choices: [SGD, Adam]
train_classes = 31 if dataset == 'hmdb51' else 10  # TODO: ucf101
resume = True

args = parse_args()

# 加载数据
train_file = '/opt/data/private/DL_Workspace/fsv-baseline/datafile/{}/train_subtrain.txt'.format(args.dataset)
val_file = '/opt/data/private/DL_Workspace/fsv-baseline/datafile/{}/train_subval.txt'.format(args.dataset)
train_dataset = SimpleVideoDataset(train_file, split='train', clip_len=16)
val_dataset = SimpleVideoDataset(val_file, split='val', clip_len=16)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# 建立模型
model = C3D(train_classes, pooling=args.pooling)
model = model.cuda()
model = nn.DataParallel(model)
if args.optim == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
elif args.optim == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = None
    raise NotImplementedError("Not support for this optim:{}".format(args.optim))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()

# 模型保存，导入相关，日志
save_dir = './chechpoints'
log_dir = './log'

save_name = 'C3D_{}_{}_lr_{}_batch_{}_epoch_{}_pool_{}'.format(dataset, optimization, args.lr, args.batch_size, args.stop_epoch, str(args.pooling)) + '_3fc'
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
    for i, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        logit = model(x)
        loss = loss_fn(logit, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = avg_loss + loss.item()
        if i % print_freq == 0:
            print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
            log_file.write('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}\n'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
    test_iter_num = len(val_loader)
    acc_all = []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.cuda()
            y = y.cuda()
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
        torch.save({'epoch': epoch, 'state': model.module.state_dict(), 'max_acc': max_acc}, outfile)

    if (epoch % args.save_freq==0) or (epoch==args.stop_epoch-1):
        outfile = os.path.join(save_dir, '{:d}.tar'.format(epoch))
        torch.save({'epoch':epoch, 'state':model.module.state_dict(), 'max_acc': max_acc}, outfile)
    scheduler.step()
print(max_acc)
log_file.write('best acc is: {}\n'.format(max_acc))
log_file.close()
