import torch
from torch import sparse


def sparsify(self):
    lsize = list(self.size())
    lsize.reverse()

    list_index = []
    cp = 1
    for i, s in enumerate(lsize):
        cl = []
        for i in range(s):
            cl += cp * [i]
        list_index.insert(0, cl)
        for j in range(len(list_index)-1):
            list_index[j+1] = list_index[j+1] * s
        cp *= s
    return torch.sparse.FloatTensor(torch.LongTensor(list_index),
                                    self.view(-1),
                                    self.size())


torch.FloatTensor.to_sparse = sparsify
