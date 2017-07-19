import torch
import json
import io


def json_loader(filepath):
    if(filepath.split('.')[-1] == 'json'):
        load_file = io.open(filepath, 'r')
        return json.load(load_file)


def sparse_dct_transform(x):
    value = torch.zeros(x['vocab_size'])
    for v in x['data']:
        value[v[0]] = v[1]
    return value


def one_hot(y, nb_classes):
    label = torch.zeros(nb_classes)
    label[y] = 1.
    return label
