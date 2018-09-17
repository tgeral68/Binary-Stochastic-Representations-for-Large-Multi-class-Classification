import math
import torch

from torch import nn
from torch.nn import functional
from torch.autograd.function import Function, InplaceFunction
from .sparse import SparseLinear
# a bernouilli sampling function which
# backward the output gradient


# A gradient estimor for a bernouilli sampling
class BernoulliNoGrad_func(torch.autograd.Function):
    """
        Sample over n bernouilli distribution (vector in [0,1[^n)
        and use as backward value the gradient of the identity 
        function
    """

    @staticmethod
    def forward(ctx, input):

        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class StraightThroughEstimator(nn.Module):
    def __init__(self, stochastic=True, slope=1.):
        super(StraightThroughEstimator, self).__init__()
        self.stochastic = stochastic
        self.func_module = BernoulliNoGrad_func()
        self.slope = slope

    def forward(self, policy):
        if(self.stochastic):
            policy.data.mul_(self.slope)
            self.a = torch.sigmoid(policy)

            return self.func_module.apply(self.a)
        else:
            # generally used for prediction step
            return torch.round((torch.sigmoid(policy)))


class STEECOCSparseLinearTriplet(nn.Module):
    def __init__(self, code_size, vocabulary_size, nb_classe, sparse=False):
        super(STEECOCSparseLinearTriplet, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.code_size = code_size
        if(sparse):
            self.encoder = SparseLinear(self.vocabulary_size,
                                        self.code_size)
        else:
            self.encoder = nn.Linear(self.vocabulary_size, self.code_size)

        self.ste = StraightThroughEstimator()
        self.decoder = nn.Linear(code_size, nb_classe)
        self.training = True

    def forward(self, v):

        if(self.training and v.dim() == 4):


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
        elif(self.training and v.dim() == 3):
            values1 = v[:, :, 0]

            self.cencoded1 = self.encoder(values1)
            self.ccode1 = self.ste(self.cencoded1)
            values2 = v[:, :, 1]
            values3 = v[:, :, 2]
            self.cencoded2 = self.encoder(values2)
            self.ccode2 = self.ste(self.cencoded2)
            self.cencoded3 = self.encoder(values3)
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
            elif(v.dim() == 4):
                keys1 = v[:, :, 0, 0].long()
                values1 = v[:, :, 1, 0]
            if(v.dim() == 2):
                self.cencoded1 = self.encoder(v)
                self.ccode1 = self.ste(self.cencoded1)
            else:
                self.cencoded1 = self.encoder((keys1, values1))
                self.ccode1 = self.ste(self.cencoded1)
            return self.decoder(self.ccode1)
