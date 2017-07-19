import math
import torch

from torch import nn
from torch.nn import functional
from torch.autograd.function import Function, InplaceFunction
from .sparse import SparseLinear
# a bernouilli sampling function which
# backward the output gradient


class BernoulliNoGrad_func(Function):
    def __init__(self):
        super(BernoulliNoGrad_func, self).__init__()

    def forward(self, probs):
        return probs.new().resize_as_(probs).bernoulli_(probs)

    def backward(self, grad_output):
        return grad_output


class BernoulliNoGrad_func2(Function):
    def __init__(self):
        super(BernoulliNoGrad_func2, self).__init__()

    def forward(self, probs):
        return probs.new().resize_as_(probs).\
            bernoulli_((probs/2.) + 0.5) * 2 - 1

    def backward(self, grad_output):
        return grad_output
# a bernouilli sampling function which
# allow reinforce


class BernoulliUnit(nn.Module):
    def __init__(self):
        super(BernoulliUnit, self).__init__()

    def forward(self, policy):
        self.a = functional.sigmoid(policy)
        self.action = self.a.bernoulli()
        return self.action

    def reinforce(self, reward):
        self.action.reinforce(reward)

# a module applying the algorithm described
# in "Estimating or Propagating Gradients Through
# Stochastic Neurons for Conditional Computation"


class StraightThroughEstimator(nn.Module):
    def __init__(self, stochastic=True, slope=1.):
        super(StraightThroughEstimator, self).__init__()
        self.stochastic = stochastic
        self.func_module = BernoulliNoGrad_func()
        self.slope = slope

    def forward(self, policy):
        if(self.stochastic):
            policy.data.mul_(self.slope)
            self.a = functional.sigmoid(policy)

            return self.func_module(self.a)
        else:
            # only for forward
            return torch.round((functional.sigmoid(policy)))


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
