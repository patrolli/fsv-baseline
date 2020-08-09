import torch
import torch.nn as nn
# from mypath import Path  # TODO: in fact, i think mypath is useful

def pn_func(x):
    sigma = 0.1
    return (1-torch.exp(sigma*x)) / (1 + torch.exp(sigma*x))

class C3D(nn.Module):
    """
    The C3D network.
    We adopt a 3D-Conv-64-4 architecture, which means we have four convolution layer, each layer contains
    64 3D conv kernel. The architecture is:
    conv1(3, 3, 3)->pooling1(1, 2, 2)->conv2(3, 3, 3)->pooling2(2, 2, 2)->conv3(3, 3, 3)->pooling3(2, 2, 2)->
    conv4(3, 3, 3)->pooling4(2, 2, 2)
    input_size is (N, C, D, H, W), suppose (D, H, W) is (16, 128, 128), size changes as:
    (16, 128, 128)->(16, 64, 64)->(8, 32, 32)->(4, 16, 16)->(2, 8, 8)-(flatten)->(128x64=8192)
    # pooling chioces  = ['avg', 'max', 'bilinaer']
    # TODO: I suppose the feature dim is so high...(8192)
    """

    def __init__(self, num_classes, pretrained=False, need_feature=False, pooling=None):
        super(C3D, self).__init__()

        self.need_feature = need_feature
        self.pooling = pooling

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        if pooling == 'avg':
            self.final_pooling = nn.AdaptiveAvgPool3d((1, None, None))
        elif pooling == 'max':
            self.final_pooling = nn.AdaptiveMaxPool3d((1, None, None))

        else:
            self.final_pooling = nn.Identity()

        if pooling == None:
            self.fc1 = nn.Linear(8192, 4096)
            self.fc2 = nn.Linear(4096, 4096)
            self.fc3 = nn.Linear(4096, num_classes)
        else:
            self.fc1 = nn.Linear(4096, 2048)
            self.fc2 = nn.Linear(2048, 2048)
            self.fc3 = nn.Linear(2048, num_classes)

        # self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = self.final_pooling(x)
        if self.pooling == 'bilinear':
            T, H, W = x.size()[2:]
            x = x.reshape(x.size()[0], x.size()[1], -1) # (n, c, TxHxW)
            x = torch.bmm(x, x.transpose(1, 2).contiguous()).div_(H*W*T) # (n, c, c)
            x = pn_func(x)
            # print(x)
        x = x.reshape(x.size()[0], -1)
        if not self.need_feature:
            x = self.relu(self.fc1(x))
        #    x = self.dropout(x)
            x = self.relu(self.fc2(x))
        #    x = self.dropout(x)
            logits = self.fc3(x)
            return logits
        else: # 直接返回 feature
            return x

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3, model.conv4, model.fc]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 128, 128)
    net = C3D(num_classes=31, pretrained=False, pooling='bilinear')

    outputs = net.forward(inputs)
    print(outputs.size())