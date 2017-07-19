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


# from module.sparse import SparseLinear
# from  torch.autograd import Variable
# import torch
# module = SparseLinear(5,10)
# w = torch.rand(3,7)
# x = (torch.rand(3,7)*5).long()
# module(Variable(x),Variable(w))

class EmbeddingSum(nn.Module):
    def __init__(self, input_size, output_size):
        super(EmbeddingSum, self).__init__()
        self.embedding = nn.Embedding(input_size, output_size,
                                      scale_grad_by_freq=True,
                                      padding_idx=0)
        self.output_size = output_size

    def forward(self, x):
        return self.embedding(x).sum(1).squeeze()


class STEECOCSparseLinearTriplet(nn.Module):
    def __init__(self, code_size, vocabulary_size, nb_classe):
        super(STEECOCSparseLinearTriplet, self).__init__()
        self.encoder = SparseLinear(vocabulary_size, code_size)
        self.ste = StraightThroughEstimator()
        self.decoder = nn.Linear(code_size, nb_classe)
        self.training = True

    def forward(self, v):

        if(self.training):
            keys1 = v[:, :, 0, 0].long()
            values1 = v[:, :, 1, 0]
            self.cencoded1 = self.encoder((keys1, values1))
            self.ccode1 = self.ste(self.cencoded1)
            keys2 = v[:, :, 0, 1].long()
            values2 = v[:, :, 1, 1]
            keys3 = v[:, :, 0, 2].long()
            values3 = v[:, :, 1, 2]
            self.cencoded2 = self.encoder((keys2, values2))
            self.ccode2 = self.ste(self.cencoded2)
            self.cencoded3 = self.encoder((keys3, values3))
            self.ccode3 = self.ste(self.cencoded3)
            return self.decoder(self.ccode1),\
                self.decoder(self.ccode2),\
                self.decoder(self.ccode2)
        else:
            keys1 = 0
            values1 = 0
            if(v.dim() == 3):
                keys1 = v[:, :, 0].long()
                values1 = v[:, :, 1]
            else:
                keys1 = v[:, :, 0, 0].long()
                values1 = v[:, :, 1, 0]
            self.cencoded1 = self.encoder((keys1, values1))
            self.ccode1 = self.ste(self.cencoded1)
            return self.decoder(self.ccode1)
