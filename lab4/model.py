# add your source codes regarding the model and its layers here
import torch as pt
import torch.nn as nn
import numpy as np
from torch import from_numpy


# class EmbedBlock(nn.Module):
#     def __init__(self, num_to_pad=400) -> None:
#         super(EmbedBlock, self).__init__()
#         self.pad = num_to_pad

#     def forward(self, seq_ptr):
#         seq = seq_ptr[0]
#         ptr = seq_ptr[1]
#         seq_ = pt.empty(1, self.pad)
#         for i in range(len(ptr) - 2):
#             s = seq[ptr[i] + 1:ptr[i + 1] - 1]  # 去掉开始和结束的21,22
#             zero_pad = self.pad - len(s)
#             if zero_pad % 2 == 0:
#                 s = nn.functional.pad(s, (zero_pad//2, zero_pad//2), "constant", 0)
#             else:
#                 s = nn.functional.pad(s, (zero_pad//2, zero_pad//2+1), "constant", 0)
#             seq_ = pt.cat((seq_, s[None,:]), dim=0)
#         seq_ = seq_[1:]

#         return seq_[:,None,:].float().cuda()
class EmbedBlock(nn.Module):
    def __init__(self, num_to_pad=400) -> None:
        super(EmbedBlock, self).__init__()
        self.pad = num_to_pad

    def forward(self, seq_ptr):
        seq = seq_ptr[0]
        ptr = seq_ptr[1]
        seq_ = []
        for i in range(len(ptr) - 2):
            s = seq[ptr[i] + 1:ptr[i + 1] - 1]  # 去掉开始和结束的21,22
            zero_pad = self.pad - len(s)
            if zero_pad % 2 == 0:
                s = np.insert(s, 0, [0] * (zero_pad // 2))
                s = np.append(s, [0] * (zero_pad // 2))
            else:
                s = np.insert(s, 0, [0] * (zero_pad // 2))
                s = np.append(s, [0] * (zero_pad // 2 + 1))
            seq_.append(s)
        seq_ = np.array(seq_)

        return from_numpy(seq_).float().unsqueeze(1).cuda()


class Conv1d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, 
                 stride:int=1, padding:int=0, bias:bool=True) -> None:
        super(Conv1d, self).__init__()
        self.stride = stride
        self.pad = padding
        self.ker_size = kernel_size        
        self.out_ch = out_channels
        self.kernel = nn.Parameter(nn.init.xavier_normal_(pt.randn(out_channels, in_channels, kernel_size)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(pt.zeros(out_channels))

    def forward(self, x):    
        # x：(batch_size, in_channel, in_len)
        out_len = (x.shape[2] + 2*self.pad - self.ker_size)//self.stride + 1
        x = nn.functional.pad(x, (self.pad, self.pad), "constant", 0)
        
        # (batch_size, in_ch, 1, in_len) -> (batch_size, in_ch*ker_size, out_len)
        x_unfold = nn.functional.unfold(x[:,:,None,:], 
                                        kernel_size=(1,self.ker_size), stride=(1,self.stride))
        out = self.kernel.view(self.out_ch, -1).matmul(x_unfold).view(x.shape[0], self.out_ch, out_len)
                 
        if self.bias is not None:
            out += self.bias[None,:,None]
 
        return out


# 第一次卷积减小尺寸，第二次卷积不改变尺寸
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1):
        super(BasicBlock, self).__init__()
        pad=1
        if kernel_size==5:
            pad=2
        elif kernel_size==7:
            pad=3

        self.conv1 = Conv1d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, 
                               stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:            
            self.shortcut = Conv1d(in_channels=in_channels, out_channels=out_channels, 
                                      kernel_size=1, stride=stride, padding=0, bias=False)
        self.relu = nn.ReLU()

        self.model=nn.Sequential(self.conv1, self.bn1 ,nn.ReLU(), self.conv2)

    def forward(self, x):
        out = self.relu(self.bn2(self.model(x)+self.shortcut(x)))
        return out


# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(
        self, num_blocks_list, input_len=400, num_classes=6630, block=BasicBlock):
        super(ResNet, self).__init__()
        self.len=input_len
        self.emb = EmbedBlock()
        self.conv1 = Conv1d(in_channels=1, out_channels=16, kernel_size=3,
                                stride=1, padding=1, bias=False)  # 1*400->16*400
        self.in_channels = 16
        self.bn = nn.BatchNorm1d(num_features=16)
        self.layer1 = self._make_layer(block, num_blocks_list[0], 32, 3, stride=1)  # 16*400->64*400
        self.layer2 = self._make_layer(block, num_blocks_list[1], 64, 3, stride=2)  # 64*400->128*200  
        self.layer3 = self._make_layer(block, num_blocks_list[2], 128, 3, stride=2)  # 128*200->256*100
        self.layer4 = self._make_layer(block, num_blocks_list[3], 256, 3, stride=2)  # 256*100->512*50
        self.linear = nn.Linear(self.in_channels * self.len//2, num_classes)  #

        self.model = nn.Sequential(self.emb, self.conv1, self.bn, nn.ReLU(),
                                   self.layer1, self.layer2, self.layer3, self.layer4,
                                   nn.AvgPool1d(2), nn.Flatten(), self.linear)
    def _make_layer(self, block, num_blocks, out_channels, kernel_size, stride):
        # 巧妙，第一个block的stride为stride，其余为1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, kernel_size, s))
            self.in_channels = out_channels  # 巧妙，每次都更新输入通道数
            self.len=(self.len-1)//s+1
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
