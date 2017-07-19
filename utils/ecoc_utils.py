import torch
import random


# generate random code
# with :
# code_size : the size of the code
# nb_classes : the number of classes
# vs : the probabilitie of 1 (respectively 0) value bits
def random_codes(code_size, nb_classe, vs=(0.25, 0.25)):
    cs = []
    for i in range(nb_classe):
        cc = []
        for j in range(code_size):
            rv = random.random()
            if(rv < vs[0]):
                # value 1
                cc.append(1)
            elif(rv < vs[0] + vs[1]):
                # value -1
                cc.append(-1)
            else:
                # not take into account
                cc.append(0)
        cs.append(cc)
    return torch.LongTensor(cs)


# generate random code matrix selecting the best
# with :
# code_size : the size of the code
# nb_classe : the number of classes
# nb_code : number of generated codes by classes
# vs : the probabilitie of 1 (respectively 0) value bits
def MHamming_random_codes_fast(code_size, nb_classe, nb_code, vs=(0.25, 0.25)):
    best_codes = torch.Tensor()
    max_min_distance = -1
    for i in range(nb_code):
        cg = random_codes(code_size, nb_classe, vs)
        v = [(-(((cg - c.expand_as(cg)).abs() *
              -1).sum(1)).squeeze().topk(2)[0][1]) for c in cg]

        if(torch.Tensor(v).min() > max_min_distance):
            max_min_distance = torch.Tensor(v).min()
            best_codes = cg
    return best_codes

def MHamming_random_codes(code_size, nb_classe, nb_code, vs=(0.25, 0.25)):
    best_codes = torch.LongTensor()
    max_min_distance = -1
    for i in range(nb_classe):
        for j in range(nb_code):

            cg = random_codes(code_size, 1, vs)
            cg = torch.cat((best_codes, cg), 0)
            v = [Hamming_distanceFast(c, cg).max() for c in cg]
            if(torch.Tensor(v).min() > max_min_distance or j == 0):
                max_min_distance = torch.Tensor(v).min()
                best_codes = cg
    return best_codes

def Hamming_distanceFast(x, B):
    return (x.unsqueeze(0).expand_as(B) - B).abs().sum(1)


# def Mean_distance_by_classes(Bx, By):
#     return


def MHamming_codes_fast(code_size, nb_classe, nb_code, vs=(0.25, 0.25)):

    # (  1 | -1 | -1  |  1 |  1 | )
    # (  1 | -1 |  1  | -1 | -1 |  )
    # ( -1 |  1 |  1  | -1 | -1 |)
    # ( -1 |  1 | -1  |  1 | -1 |)

    # start with ones vector
    codes = torch.ones(nb_code)

    return best_codes
