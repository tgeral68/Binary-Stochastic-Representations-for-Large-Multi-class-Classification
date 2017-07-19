import zipfile
import torch
import shutil
import json
import io
import os

from random import randint
from torch.utils.data import Dataset
from .downloader import DMOZ1KDownloader, DMOZ12KDownloader
# A json datasets loader which
# take json containing two keys
# 'X' and 'Y'


class TXTDataset(Dataset):
    def __init__(self, filepath='', max_lenght=500,
                 word_index_map=None, class_map={}):
        super(TXTDataset, self).__init__()
        self.Y = []
        self.Xkeys = []
        self.Xval = []
        self.lenX = []
        self.new_map_index = word_index_map is None

        self.word_index_map = {} if self.new_map_index else word_index_map
        self.class_map = class_map
        with open(filepath) as f:
            for line in f:
                line_split = line.split()

                wkeys = []
                wvalues = []
                for i in range(len(line_split)-1):
                    key_val_split = line_split[i+1].split(':')
                    if(int(key_val_split[0]) not in self.word_index_map
                       and self.new_map_index):
                        self.word_index_map[int(key_val_split[0])] = \
                            len(self.word_index_map)+1
                    if(int(key_val_split[0]) in self.word_index_map):
                        wkeys.append(self.word_index_map
                                     [int(key_val_split[0])])
                        wvalues.append(float(key_val_split[1]))

                if(len(wkeys) <= max_lenght):
                    self.lenX.append(len(wkeys))
                    self.Xkeys.append(wkeys + ([0] *
                                      (max_lenght - len(wkeys))))
                    self.Xval.append(wvalues + ([0] *
                                     (max_lenght - len(wkeys))))
                    if(int(line_split[0]) not in self.class_map):
                        self.class_map[int(line_split[0])] =\
                            len(self.class_map)
                    self.Y.append(self.class_map[int(line_split[0])])

        self.Xkeys = torch.Tensor(self.Xkeys)
        self.Xval = torch.Tensor(self.Xval)
        self.X = torch.cat([self.Xkeys.unsqueeze(2), self.Xval.unsqueeze(2)],
                           2)
        # print(self.X.size())
        self.Y = torch.LongTensor(self.Y)

    def __getitem__(self, index):
        return self.X.__getitem__(index), \
            self.Y.__getitem__(index)

    def __len__(self):
        return len(self.X)


class TXTDatasetTriplet(TXTDataset):
    def __init__(self, filepath='', max_lenght=500,
                 word_index_map=None, class_map={},
                 verbose=0):
        super(TXTDatasetTriplet, self).__init__(filepath,
                                                max_lenght,
                                                word_index_map,
                                                class_map)
        self.classr_dict = {}
        for i in range(self.Y.size(0)):
            if(self.Y[i] not in self.classr_dict):
                self.classr_dict[self.Y[i]] = []
            self.classr_dict[self.Y[i]].append(i)

    def __getitem__(self, index):
        adv1 = randint(0, len(self.X)-1)
        adv2 = self.classr_dict[self.Y[index]][randint(0, len(self.classr_dict[self.Y[index]])-1)]
        return torch.cat((self.X.__getitem__(index).unsqueeze(2),
                          self.X.__getitem__(adv1).unsqueeze(2),
                          self.X.__getitem__(adv2).unsqueeze(2)), 2),\
            torch.LongTensor([self.Y.__getitem__(index)])

    def __len__(self):
        return (len(self.X))


class JSONDataset(Dataset):
    def __init__(self, filepath='', train=True):
        super(JSONDataset, self).__init__()
        json_file = io.open(filepath, 'r')
        data = json.load(json_file)
        json_file.close()

        self.X = torch.Tensor(data['X'])
        self.Y = torch.Tensor(data['Y']).long()

    def cuda(self):
        self.X = self.X.cuda()
        self.Y = self.Y.cuda()

    def cpu(self):
        self.X = self.X.cpu()
        self.Y = self.Y.cpu()

    def __getitem__(self, index):
        return self.X.__getitem__(index), \
            self.Y.__getitem__(index)

    def __len__(self):
        return len(self.X)


class JSONDatasetTriplet(JSONDataset):
    def __init__(self, filepath=''):
        super(JSONDatasetTriplet, self).__init__(filepath=filepath)
        self.classr_dict = {}
        for i in range(self.Y.size(0)):
            if(self.Y[i] not in self.classr_dict):
                self.classr_dict[self.Y[i]] = []
            self.classr_dict[self.Y[i]].append(i)

    def __getitem__(self, index):
        adv1 = randint(0, len(self.X)-1)
        adv2 = self.classr_dict[self.Y[index]][randint(0, len(self.classr_dict[self.Y[index]])-1)]
        return torch.cat((self.X.__getitem__(index).unsqueeze(2),
                          self.X.__getitem__(adv1).unsqueeze(2),
                          self.X.__getitem__(adv2).unsqueeze(2)), 2),\
            torch.LongTensor([self.Y.__getitem__(index)])

    def __len__(self):
        return (len(self.X))


class T7Dataset(Dataset):
    def __init__(self, filepath='', train=True):
        super(T7Dataset, self).__init__()
        data = torch.load(filepath)

        self.X = torch.Tensor(data['X'].squeeze())
        self.Y = data['Y'].long()

    def cuda(self):
        self.X = self.X.cuda()
        self.Y = self.Y.cuda()

    def cpu(self):
        self.X = self.X.cpu()
        self.Y = self.Y.cpu()

    def __getitem__(self, index):
        return self.X.__getitem__(index), \
            self.Y.__getitem__(index)

    def __len__(self):
        return len(self.X)


class T7DatasetTriplet(T7Dataset):
    def __init__(self, filepath=''):
        super(T7DatasetTriplet, self).__init__(filepath=filepath)
        self.classr_dict = {}
        for i in range(self.Y.size(0)):
            if(self.Y[i] not in self.classr_dict):
                self.classr_dict[self.Y[i]] = []
            self.classr_dict[self.Y[i]].append(i)

    def __getitem__(self, index):
        adv1 = randint(0, len(self.X)-1)
        adv2 = self.classr_dict[self.Y[index]][randint(0, len(self.classr_dict[self.Y[index]])-1)]

        return torch.cat((self.X.__getitem__(index).unsqueeze(1),
                          self.X.__getitem__(adv1).unsqueeze(1),
                          self.X.__getitem__(adv2).unsqueeze(1)), 1),\
            torch.LongTensor([self.Y.__getitem__(index)])

    def __len__(self):
        return (len(self.X))


class T7DatasetMF(T7Dataset):
    def __init__(self, folderpath='', train=True):
        super(T7Dataset, self).__init__()
        self.X = torch.Tensor()
        self.Y = torch.LongTensor()
        lsdir = os.listdir(folderpath)
        for f in lsdir:
            print(os.path.join(folderpath, f))
            if(f not in ['.', '..']):
                data = torch.load(os.path.join(folderpath, f))
                self.X = torch.cat([self.X, data['X'].squeeze()], 0)
                self.Y = torch.cat([self.Y, data['Y'].squeeze()], 0)


class T7DatasetMFTriplet(T7DatasetMF):
    def __init__(self, filepath=''):
        super(T7DatasetMFTriplet, self).__init__(folderpath=filepath)
        self.classr_dict = {}
        for i in range(self.Y.size(0)):
            if(self.Y[i] not in self.classr_dict):
                self.classr_dict[self.Y[i]] = []
            self.classr_dict[self.Y[i]].append(i)

    def __getitem__(self, index):
        adv1 = randint(0, len(self.X)-1)
        adv2 = self.classr_dict[self.Y[index]][randint(0, len(self.classr_dict[self.Y[index]])-1)]

        return torch.cat((self.X.__getitem__(index).unsqueeze(1),
                          self.X.__getitem__(adv1).unsqueeze(1),
                          self.X.__getitem__(adv2).unsqueeze(1)), 1),\
            torch.LongTensor([self.Y.__getitem__(index)])

    def __len__(self):
        return (len(self.X))


# specific dataset

config_file = './conf/.dataset.conf'
default_dataset_path = 'dataset/'


def get_config(key):
    if(not os.path.exists(config_file)):
        conf_folder = os.path.split(os.path.realpath(config_file))[0]
        print('creating the configuration folder "'+str(conf_folder)+'"')
        os.makedirs(conf_folder)
        torch.save({}, config_file)
    f = torch.load(config_file)
    if(key in f):
        return f[key]
    else:
        return None


def set_config(key, value):
    f = torch.load(config_file)
    f[key] = value
    torch.save(f, config_file)


def DMOZ_1K_data():
    cfg = get_config('DMOZ1K')
    # dataset not downloaded
    if(cfg is None):
        dataset_path = os.path.realpath(os.path.join(default_dataset_path,
                                        'DMOZ1K/'))
        try:
            shutil.rmtree(dataset_path)
        except Exception:
            pass
        os.makedirs(dataset_path)
        usual = DMOZ1KDownloader(dataset_path)['usual']
        set_config('DMOZ1K', {'path': dataset_path,
                   'zip': usual})

        cfg = get_config('DMOZ1K')
    # dataset not decompressed
    if('dataset_root_data_path' not in cfg):
        folder_path = cfg['path']
        # decompress usual
        if(not os.path.exists(os.path.join(folder_path, '1k_class'))):
            cp_path = os.path.join(cfg['path'], cfg['zip'])
            zip_ref = zipfile.ZipFile(cp_path, 'r')
            zip_ref.extractall(cfg['path'])
            zip_ref.close()
            cfg['dataset_root_data_path'] = os.path.join(folder_path,
                                                         '1k_class')
            print(cfg)
    set_config('DMOZ1K', cfg)
    print(cfg)
    list_file = []
    for f1 in os.listdir(cfg['dataset_root_data_path']):
        rpath = os.path.join(cfg['dataset_root_data_path'], f1)
        list_file.append({'train': os.path.join(rpath, 'train.txt'),
                          'test': os.path.join(rpath, 'test.txt'),
                          'validation': os.path.join(rpath, 'validation.txt')
                          })
    return list_file


def DMOZ_12K_data():
    cfg = get_config('DMOZ12K')
    # dataset not downloaded
    if(cfg is None):
        dataset_path = os.path.realpath(os.path.join(default_dataset_path,
                                        'DMOZ12K/'))
        try:
            shutil.rmtree(dataset_path)
        except Exception:
            pass
        os.makedirs(dataset_path)
        usual = DMOZ12KDownloader(dataset_path)['usual']
        set_config('DMOZ12K', {'path': dataset_path,
                   'zip': usual})

        cfg = get_config('DMOZ12K')
    # dataset not decompressed
    if('dataset_root_data_path' not in cfg):
        folder_path = cfg['path']
        # decompress usual
        if(not os.path.exists(os.path.join(folder_path, '12k_class'))):
            cp_path = os.path.join(cfg['path'], cfg['zip'])
            zip_ref = zipfile.ZipFile(cp_path, 'r')
            zip_ref.extractall(cfg['path'])
            zip_ref.close()
            cfg['dataset_root_data_path'] = os.path.join(folder_path,
                                                         '12k_class')
    set_config('DMOZ12K', cfg)
    print(cfg)
    list_file = []
    for f1 in os.listdir(cfg['dataset_root_data_path']):
        rpath = os.path.join(cfg['dataset_root_data_path'], f1)
        list_file.append({'train': os.path.join(rpath, 'train.txt'),
                          'test': os.path.join(rpath, 'test.txt'),
                          'validation': os.path.join(rpath, 'validation.txt')
                          })
    return list_file
