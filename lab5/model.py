# add your source codes regarding the model and its layers here
import torch as pt
import torch.nn as nn
import numpy as np
from torch import from_numpy


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
        x_unfold = nn.functional.unfold(x.unsqueeze(2), 
                                        kernel_size=(1,self.ker_size), stride=(1,self.stride))
        out = self.kernel.view(self.out_ch, -1).matmul(x_unfold).view(x.shape[0], self.out_ch, out_len)
                 
        if self.bias is not None:
            b = pt.unsqueeze(self.bias, 0).unsqueeze(2)
            out += b
 
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int,stride=1):
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
                                      kernel_size=1, stride=stride, bias=False)
        self.relu = nn.ReLU()

        self.model=nn.Sequential(self.conv1, self.bn1 ,nn.ReLU(), self.conv2)

    def forward(self, x):
        out = self.relu(self.bn2(self.model(x)+self.shortcut(x)))
        return out


class SelfAttention(nn.Module):
    def __init__(self, d_m, d_k) -> None:
        super(SelfAttention, self).__init__()
        self.d_m = d_m
        self.d_k = d_k
        self.Wq = nn.Linear(in_features=self.d_m, out_features=self.d_k)
        self.Wk = nn.Linear(in_features=self.d_m, out_features=self.d_k)
        self.Wv = nn.Linear(in_features=self.d_m, out_features=self.d_k)

    def forward(self, x):
        q,k,v = self.Wq(x), self.Wk(x),self.Wv(x)
        score = pt.einsum('nci, ncj -> nc',q,k)
        score /= pt.sqrt(pt.tensor(self.d_k))
        score = nn.functional.softmax(score, dim=-1)
        out = v * score[:,:,None]
        return out
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim:int, num_heads=1) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, 'embed_dim must be divided by num_heads'
        self.heads = nn.ModuleList([SelfAttention(d_m=embed_dim, d_k=embed_dim//num_heads) 
                                    for _ in range(num_heads)])
        self.Wo = nn.Linear(in_features=embed_dim, out_features=embed_dim)
    
    def forward(self, x):
        Z = pt.cat([head(x) for head in self.heads], dim=-1)
        if len(self.heads) > 1:
            return self.Wo(Z)
        else:
            return Z


class TransformerBlock(nn.Module):
    def __init__(self, dim:int, len:int, num_heads:int, dropout=0.1) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(len, num_heads)
        self.shortcut1 = nn.Sequential()
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x_n = self.bn1(x)
        out1 = self.relu1(self.shortcut1(x) + self.attention(x_n))
        return out1


class HeadBlock(nn.Module):
    def __init__(self, in_ch:int, length:int, num_classes:int):
        super(HeadBlock, self).__init__()
        self.pool=nn.AvgPool1d(2)
        self.norm=nn.BatchNorm1d(in_ch)
        self.flat=nn.Flatten()
        self.fc = nn.Linear(in_features=in_ch*length//2, out_features=num_classes)
        self.model = nn.Sequential(self.pool, self.norm, self.flat, self.fc)

    def forward(self, x):
        return self.model(x)


class Convformer(nn.Module):
    def __init__(self, seq_len=400, num_classes=6630):
        super(Convformer, self).__init__()
        self.seq_len = seq_len
        self.in_ch = 1
        self.emb = EmbedBlock()
        self.layer0 = self._make_layer(BasicBlock, ou_ch=16, stride=1)
        self.layer1 = self._make_layer(BasicBlock, ou_ch=32, stride=1)
        self.layer2 = self._make_layer(BasicBlock, ou_ch=64, stride=1)
        self.layer3 = self._make_layer(BasicBlock, ou_ch=128, stride=2)
        self.layer4 = self._make_layer(BasicBlock, ou_ch=256, stride=2)
        self.layer5 = self._make_layer(BasicBlock, ou_ch=320, stride=2)
        self.layer6 = self._make_layer(TransformerBlock, ou_ch=320, num_heads=2)
        self.head = HeadBlock(in_ch=320, length=self.seq_len, num_classes=num_classes)
        self.model = nn.Sequential(self.emb, self.layer0, self.layer1, self.layer2,
                                   self.layer3, self.layer4, self.layer5, 
                                   self.head)      

    def _make_layer(self, block, ou_ch, stride=2, num_heads=2):
        if block == BasicBlock:
            self.block = BasicBlock(in_channels=self.in_ch, out_channels=ou_ch, 
                                    kernel_size=3, stride=stride)
            self.seq_len = (self.seq_len - 1)//stride + 1
            self.in_ch = ou_ch
        elif block == TransformerBlock:
            self.block = TransformerBlock(self.in_ch, self.seq_len, num_heads=num_heads)

        return nn.Sequential(self.block)

    def forward(self, x):
        return self.model(x)
