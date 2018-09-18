import math
import torch

from torch import nn
from torch.nn import functional
from torch.autograd.function import Function, InplaceFunction

# a sparse Linear module taking in input
# the index and the weight associate to those index


class SparseLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(SparseLinear, self).__init__()
        self.embedding = nn.Embedding(input_size, output_size,
                                      scale_grad_by_freq=False,
                                      padding_idx=0)
        self.output_size = output_size

    def forward(self, x):
        index, weight = 0, 0
        if(type(x) == tuple):
            index, weight = x
        else:

            index, weight = x[:, :, 0].long(), x[:, :, 1]

        embed = self.embedding(index)
        return weight.unsqueeze(2).expand_as(embed).mul(embed).sum(1).squeeze()
