import torch as pt
from data import names


class FakeModel(pt.nn.Module):
    def __init__(self):
        super(FakeModel, self).__init__()


    def forward(self, x):
        N = len(x)
        # return a random tensor of shape N * len(names)
        return pt.rand(N, len(names))


