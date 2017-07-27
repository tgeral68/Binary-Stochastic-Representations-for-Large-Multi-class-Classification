import torch
from torch.nn import functional as F
from torch.autograd import Variable


class MultiTensor(Object):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list

    def __mul__(self, other_multitensor):
        for i in range(len(self.tensor_list)):
            self.tensor_list[i].mul_(other_multitensor[i])
        return self

    def __pow__(self, power):
        for i in range(power):
            self.__mul__(self)
        return self

    def __add__(self, other_multitensor):
        for i in range(len(self.tensor_list)):
            self.tensor_list[i].add_(other_multitensor[i])
        return self

    def __truediv__(self, p2):
        if(type(p2) == int):
            for i in range(len(self.tensor_list)):
                self.tensor_list[i].div_(p2)
            return self

    def get_variable(self):
        variable_list = []
        for i in range(len(self.tensor_list)):
            variable_list.append(Variable(self.tensor_list[i]))
        return MultiVariable(variable_list)


class MultiVariable(Object):
    def __init__(self, variable_list):
        self.variable_list = variable_list

    def __mul__(self, other_multivariable):

        return MultiVariable([self.variable_list[i]*(other_multivariable[i])
                             for i in range(len(self.variable_list))])

    def __pow__(self, power):
        x = self
        for i in range(power):
            x = x * self
        return x

    def __add__(self, other_multivariable):

        return MultiVariable([self.variable_list[i] + (other_multivariable[i])
                             for i in range(len(self.variable_list))])

    def __sub__(self, other_multivariable):

        return MultiVariable([self.variable_list[i] - (other_multivariable[i])
                             for i in range(len(self.variable_list))])

    def get_data(self):
        tensor_list = []
        for i in range(len(self.variable_list)):
            tensor_list.append(self.variable_list[i].data)
        return MultiTensor(tensor_list)


def get_weight(model):
    weight_list = []
    for param in model.parameters():
        weight_list.append(param)
    return MultiVariable(weight_list)


def get_grad(model):
    weight_grad_list = []
    for param in model.parameters():
        weight_grad_list.append(param.grad)
    return MultiVariable(weight_grad_list)


def compute_fisher(model, dataloader):
    weight_list = get_weight(model)
    vx = Variable()
    for data in dataloader:
        x, y = data
        vx.resize_as_(x).copy_(x)
        vy = model(vx)
        dvy = F.softmax(vy)
        index_y = torch.multinomial(vy.data, 1).data[0][0]
        vy[index_y].backward()
        gthteta = get_grad(model).get_data()
        fisher = (gtheta**2)/len(x)
    return fisher
