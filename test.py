from io_config import parse_args, get_best_file, get_resume_file, get_assigned_file
from dataload.datamgr import SetDataManager
from models.C3D import C3D
import torch
import numpy as np

def euclidean_dist(x, y):
    # x: NxD
    # y: MxD
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(0).expand(m, n, d)
    y = y.unsqueeze(1).expand(m, n, d)

    dist = torch.pow(x - y, 2).sum(2)

    return -dist

# 导入测试的参数
args = parse_args()
print(args)
# 加载测试数据
test_file = '/opt/data/private/DL_Workspace/fsv-baseline/datafile/{}/few-shot-test.txt'.format(args.dataset)
data_mgr = SetDataManager(args.n_way, args.k_shot, args.q_query, 600)
loader = data_mgr.get_data_loader(test_file)
# 加载模型
model = C3D(num_classes=31, need_feature=True, pooling=args.pooling)  # 这里 num_classes 是一个没有用的参数,但必须加上, 保证和保存的模型参数匹配
model = model.cuda()
check_dir = './chechpoints/C3D_hmdb51_SGD_lr_0.01_batch_32_epoch_100_pool_avg_3fc'
print(check_dir)
best_file = get_best_file(check_dir)
# best_file = get_resume_file(check_dir)
tmp = torch.load(best_file)

model.load_state_dict(tmp['state'])
print('best model loaded!!!')
# 测试 loop
with torch.no_grad():
    acc_all = []
    iter_num = len(loader)
    for i, (x, _) in enumerate(loader):
        x = x.reshape(x.size(0)*x.size(1), *x.size()[2:])
        x = x.cuda()
        x_feature = model(x)
        x_feature = x_feature.reshape(args.n_way, -1, x_feature.size()[-1])
        spt_x = x_feature[:, :args.k_shot, ...]
        qry_x = x_feature[:, args.k_shot:, ...]
        qry_x = qry_x.reshape(qry_x.size(0)*qry_x.size(1), -1) # (n_way*q_query, d)
        proto = torch.mean(spt_x, dim=1).squeeze() # (n_way, d)
        dist = euclidean_dist(proto, qry_x)
        # print(dist.size())
        y_query = np.repeat(np.arange(args.n_way), args.q_query) # [0,0,1,1...,5,5]
        _, top1_label = torch.topk(dist, 1, dim=-1)
        top1_ind  = top1_label.detach().cpu().numpy()
        top1_correct = np.sum(top1_ind[:, 0] == y_query)
        acc_all.append(float(top1_correct)/len(y_query)*100)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))