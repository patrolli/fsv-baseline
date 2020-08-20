# ResNet34 -> Motion Excitation Module
# Bottleneck block
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from io_config import get_best_file


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

class BottleneckBlock(nn.Module):
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
        self.BN1 = nn.BatchNorm2d(bottleneckdim)
        self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
        self.BN2 = nn.BatchNorm2d(bottleneckdim)
        self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
        self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
        bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(1)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 1, 1]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out

def ResNet34(flatten = True):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50(flatten=True):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], [256,512,1024,2048], flatten)


class MEModule(nn.Module):
    """ Motion exciation module

    :param reduction=16
    :param n_segment=8/16
    """

    def __init__(self, channel, reduction=16, n_segment=8):
        super(MEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel // self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel // self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel // self.reduction,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.identity = nn.Identity()

    def forward(self, x):
        nt, c, h, w = x.size()
        bottleneck = self.conv1(x)  # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck)  # nt, c//r, h, w

        # t feature
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
        t_fea, __ = reshape_bottleneck.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w

        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea  # n, t-1, c//r, h, w
        #print('*'*10 + 'diff_fea' + '*'*10)
        #print(diff_fea[0, 0, :, 0, 0])
        # pad = (0,0,0,0,0,0,0,1)
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
        diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  # nt, c//r, h, w
        y = self.avg_pool(diff_fea_pluszero)  # nt, c//r, 1, 1
        y = self.conv3(y)  # nt, c, 1, 1
        y = self.bn3(y)  # nt, c, 1, 1
        y = self.sigmoid(y)  # nt, c, 1, 1
        return y


class FS_ResNet(nn.Module):
    def __init__(self, k_shot, res_size=34):
        nn.Module.__init__(self)
        self.k_shot = k_shot
        if res_size == 34:
            self.resnet = ResNet34(flatten=False)
            outchannel = 512
        elif res_size == 50:
            self.resnet = ResNet50(flatten=False)
            outchannel = 2048
        else:
            raise ValueError("no such resnet size!")

    # just remove the motion excitation module
    def forward(self, inp):
        n_way, all_shots, C, T, H, W = inp.size()
        inp = inp.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, C, H,
                                                                 W)  # inp: [n_way*(k_shot+q_query)*T, C, H, W)
        inp = self.resnet(inp)
        inp = inp.reshape(n_way, all_shots, T, *inp.size()[1:])
        h, w = inp.size()[-2:]  # feature map height and width
        support_x = inp[:, :self.k_shot, ...]  # (n_way, k_shot, T, c, h, w)
        query_x = inp[:, self.k_shot:, ...]  # (n_way, q_query, T, c, h, w)

        support_proto = torch.mean(support_x, dim=1).squeeze().reshape(n_way * T, *support_x.size()[3:])

        # second-order pooling
        q_x = query_x.permute(0, 1, 2, 4, 5, 3).contiguous()
        q_x = q_x.reshape(q_x.size()[0] * q_x.size()[1], -1, q_x.size()[-1])  # (n_way*q_query, T*H*W, c)
        q_x = torch.bmm(q_x.transpose(1, 2).contiguous(), q_x).div_(h * w * T)  # (n_way*q_query, c, c)
        q_x = q_x.reshape(q_x.size()[0], -1).unsqueeze(1)  # (n_way*q_query, 1, cxc)

        s_x = support_proto.reshape(n_way, T, *support_proto.size()[1:]).permute(0, 1, 3, 4,
                                                                                 2).contiguous()  # (n_way, T, H, W, C)
        s_x = s_x.reshape(s_x.size()[0], -1, s_x.size()[-1])  # (n_way, T*H*W, C)
        s_x = torch.bmm(s_x.transpose(1, 2).contiguous(), s_x).div_(h * w * T)
        s_x = s_x.reshape(s_x.size()[0], -1).unsqueeze(0)  # (1, n_way, cxc)

        # calculate dist
        dist = torch.pow(s_x - q_x, 2).sum(2)  # (n_way*q_query, n_way, 1)

        return -dist


# input_dim: (n_way, k_shot+q_query, C, n_frame, H, W)
# FS_MENet, stands for Few-Shot Motion Excitation Network
class FS_MENet(nn.Module):
    def __init__(self, k_shot, res_size=34):
        nn.Module.__init__(self)
        self.k_shot = k_shot
        if res_size == 34:
            self.resnet = ResNet34(flatten=False)
            #导入预训练模型
            # pretrain_model = get_best_file('./chechpoints/FS_ResNet34_hmdb51_SGD_lr_0.01_epoch_100')
            # tmp = torch.load(pretrain_model)
            # new_tmp = tmp['state'].copy()
            # for key in tmp['state'].keys():
            #     new_key = key.split('net.')[-1]
            #     new_tmp[new_key] = tmp['state'][key]
            #     new_tmp.pop(key)
            # self.resnet.load_state_dict(new_tmp)
            outchannel = 512
        elif res_size == 50:
            self.resnet = ResNet50(flatten=False)
            outchannel = 2048
        else:
            raise ValueError("no such resnet size!")
        self.menet = MEModule(outchannel)
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.cos_dist = nn.CosineSimilarity(dim=-1, eps=1e-5)

    def forward(self, inp):
        n_way, all_shots, C, T, H, W = inp.size()
        inp = inp.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, C, H, W)  # inp: [n_way*(k_shot+q_query)*T, C, H, W)
        inp = self.resnet(inp)
        h, w = inp.size()[-2:]
        inp = inp.reshape(n_way, all_shots, T, *inp.size()[1:])
        support_x = inp[:, :self.k_shot, ...]  # (n_way, k_shot, T, c, h, w)
        query_x = inp[:, self.k_shot:, ...]  # (n_way, q_query, T, c, h, w)

        support_proto = torch.mean(support_x, dim=1).squeeze().reshape(n_way*T, *support_x.size()[3:])  # (n_way*T, c, h, w)
        motion_score = self.menet(support_proto)
        motion_score = motion_score.reshape(n_way, T, *motion_score.size()[1:]) # (n_way, T, C, 1, 1)
        #print('*' * 10 + 'motion score' + '*' * 10)
        #print(motion_score[0, 0, :, ...].squeeze())
        # query feature map according to each class's motion score
        query_x = query_x.reshape(n_way*(all_shots-self.k_shot), *query_x.size()[2:]).unsqueeze(1)  #(n_way*q_query, 1, T, c, h, w)
        query_att_fe = query_x + query_x * motion_score.unsqueeze(0)  # (n_way*q_query, n_way, T, c, h, w)
        # second-order pooling
        # q_x = query_att_fe.permute(0, 1, 2, 4, 5, 3).contiguous()
        # q_x = q_x.reshape(q_x.size()[0]*q_x.size()[1], -1, q_x.size()[-1])  #(n_way*q_query*n_way, T*H*W, c)
        # q_x = torch.bmm(q_x.transpose(1, 2).contiguous(), q_x).div_(h * w * T)  # (n_way*q_query*n_way, c, c)
        # q_x = q_x.reshape(n_way*(all_shots-self.k_shot), n_way, -1)  # (n_way*q_query, n_way, cxc)

        # s_x = support_proto.reshape(n_way, T, *support_proto.size()[1:]).permute(0, 1, 3, 4, 2).contiguous()  #(n_way, T, H, W, C)
        # s_x = s_x.reshape(s_x.size()[0], -1, s_x.size()[-1])  #(n_way, T*H*W, C)
        # s_x = torch.bmm(s_x.transpose(1, 2).contiguous(), s_x).div_(h * w * T)
        # s_x = s_x.reshape(n_way, -1).unsqueeze(0) #(n_way*q_query, n_way, cxc)
        # avgpooling over T, H, W
        q_x = query_att_fe.reshape(query_att_fe.size()[0] * query_att_fe.size()[1], *query_att_fe.size()[2:]) # (n_way*query*n_way, T, c, h, w)
        q_x = q_x.permute(0, 2, 1, 3, 4).contiguous()
        q_x = self.pool(q_x).reshape(query_att_fe.size()[0], query_att_fe.size()[1], -1) # (n_way*query, n_way, c)
        s_x = support_proto.reshape(n_way, T, *support_proto.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()  #(n_way, c, T, h, w)
        s_x = self.pool(s_x).reshape(s_x.size()[0], -1).unsqueeze(0) # (1, n_way, c)
        # calculate dist
        # dist = torch.pow(s_x - q_x, 2).sum(2)  # (n_way*q_query, n_way, 1)
        dist = self.cos_dist(s_x, q_x) * 10  # 10 is the scale parameter
        return dist

if __name__ == '__main__':
    inp = torch.randn((5, 5, 3, 8, 128, 128))
    inp = inp.cuda()
    # model = FS_MENet(k_shot=1).cuda()
    model = FS_ResNet(k_shot=1, res_size=50).cuda()
    out = model(inp)
    print(out.size())
