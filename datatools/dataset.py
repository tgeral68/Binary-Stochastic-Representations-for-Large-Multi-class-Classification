import zipfile
import torch
import shutil
import json
import copy
import tqdm
import bz2
import io
import os

from random import randint
from torch.utils.data import Dataset
from .downloader import DMOZ1KDownloader, DMOZ12KDownloader, ALOIDownloader
from .compression_tools import decompress
# A json datasets loader which
# take json containing two keys
# 'X' and 'Y'


class TrainEvalDataset(Dataset):
    def __init__(self):
        self.train = False

    def isTrain(self):
        self.train

    def isEval(self):
        return not self.train

    def setTrain(self):
        self.train = True

    def setEval(self):
        self.train = False

    def build(self):
        pass

    def clear(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class TXTDataset(TrainEvalDataset):
    # read a txt file to build the Dataset
    def __init__(self, filepath, sparse=False,
                 class_map=None, max_lenght=500,
                 word_index_map=None):
        super(TXTDataset, self).__init__()
        self.max_lenght = max_lenght
        self.class_map = {} if(class_map is None) else class_map
        self.word_index_map = {} if(word_index_map is None) else word_index_map
        # if is allowed to extend the class_map
        self.ecm = (class_map is None)
        self.ewm = (word_index_map is None)
        self.filepath = filepath
        self.sparse = sparse

    def build(self):
        self.X, X = [], []
        self.Y, Y = [], []

        local_vocabulary = []

        # reading data
        with open(self.filepath) as data_file:
            for line in data_file:
                # the first char correspond to the classe
                component = line.split()
                word = [int(component[i].split(':')[0])
                        for i in range(1, len(component))]
                value = [float(component[i].split(':')[1])
                         for i in range(1, len(component))]
                X.append([word, value])
                Y.append(int(component[0]))
                local_vocabulary += word

        # deleting
        for i in range(len(X)):
            if(len(X[i][0]) > self.max_lenght):
                continue
            elif(Y[i] not in self.class_map and not self.ecm):
                continue

            if(Y[i] not in self.class_map):
                self.class_map[Y[i]] = len(self.class_map)

            self.Y.append(self.class_map[Y[i]])

            # updating word_index_map
            if(self.ewm):
                for j in range(len(X[i][0])):
                    if(X[i][0][j] not in self.word_index_map):
                        self.word_index_map[X[i][0][j]] = \
                            len(self.word_index_map)

            # adding the new input
            self.X.append([[self.word_index_map[X[i][0][j]], X[i][1][j]]
                          for j in range(len(X[i][0]))
                          if(X[i][0][j] in self.word_index_map)])

        self.Y = torch.LongTensor(self.Y)
        # in case of a non sparse dataset
        if(not self.sparse):
            Xcomplete = torch.zeros(len(self.X),
                                    len(self.word_index_map))
            for x in range(len(self.X)):
                for i in range(len(self.X[x])):
                    Xcomplete[x][self.X[x][i][0]] = self.X[x][i][1]
            self.X = Xcomplete
        else:
            Xcomplete = torch.zeros(len(self.X),
                                    self.max_lenght,
                                    2)
            for xi in range(len(self.X)):
                for i in range(len(self.X[xi])):
                    Xcomplete[xi][i][0] = self.X[xi][i][0]
                    Xcomplete[xi][i][1] = self.X[xi][i][1]
            self.X = Xcomplete
        self.classr_dict = {}
        for i in range(self.Y.size(0)):
            if(self.Y[i] not in self.classr_dict):
                self.classr_dict[self.Y[i]] = []
            self.classr_dict[self.Y[i]].append(i)

    def remove_from_index(self, list_index):

        X = 0
        if(self.X.dim() == 3):
            X = self.X.new(self.X.size(0) - len(list_index), self.X.size(1),
                           self.X.size(2))
        else:
            X = self.X.new(self.X.size(0) - len(list_index), self.X.size(1))
        Y = self.Y.new(self.X.size(0) - len(list_index))

        list_index = (torch.Tensor(list_index)).sort()[0]
        print('removing '+str(len(list_index))+' data : ')
        cpt = 0
        cpt2 = 0
        for tr, i in enumerate(tqdm.trange(len(self))):

            if(i != int(list_index[cpt])):
                X[cpt2] = self.X[i]
                Y[cpt2] = self.Y[i]
                cpt2 += 1

            else:
                if(cpt + 1 < len(list_index)):
                    cpt += 1
        self.X = X
        self.Y = Y

    def __str__(self):
        rstr = 'TXTDataset : '
        rstr += '\n \t number of words      -> '+str(len(self.vocabulary_size))
        rstr += '\n \t number of categories -> '+str(len(self.class_map))
        rstr += '\n \t number of samples    -> '+str(len(self))

    def save(self, path):
        print('Saving the dataset at "'+path+'"')
        with open(path, 'w') as f:
            for i in tqdm.trange(self.X.size(0)):
                wstr = str(self.Y[i])+' '
                for j in range(self.X[i].size(0)):
                    if(self.X.dim() == 3):

                        wstr += str(self.X[i][j][0]) + \
                            ':'+str(self.X[i][j][1]) + ' '
                    else:
                        if(self.X[i][j] != 0):
                            wstr += str(j)+':'+str(self.X[i][j])+' '
                print(wstr, file=f)
                f.flush()

    def __getitem__(self, index):
        if(self.isEval()):
            return self.X.__getitem__(index), \
                self.Y.__getitem__(index)

        else:

            adv1 = randint(0, len(self.X)-1)
            adv2 = self.classr_dict[self.Y[index]]\
                [randint(0, len(self.classr_dict[self.Y[index]])-1)]
            return torch.cat((self.X.__getitem__(index).unsqueeze(2 if(self.sparse) else 1),
                              self.X.__getitem__(adv1).unsqueeze(2 if(self.sparse) else 1),
                              self.X.__getitem__(adv2).unsqueeze(2 if(self.sparse) else 1)), 2 if(self.sparse) else 1),\
                torch.LongTensor([self.Y.__getitem__(index)])

    def __len__(self):
        return len(self.X)


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
            decompress(os.path.join(cfg['path'], cfg['zip']), cfg['path'])
            cfg['dataset_root_data_path'] = os.path.join(folder_path,
                                                         '1k_class')

    set_config('DMOZ1K', cfg)

    list_file = []
    for f1 in os.listdir(cfg['dataset_root_data_path']):
        rpath = os.path.join(cfg['dataset_root_data_path'], f1)
        train = TXTDataset(os.path.join(rpath, 'train.txt'), sparse=True)
        train.setTrain()
        test = TXTDataset(os.path.join(rpath, 'test.txt'), sparse=True,
                          word_index_map=train.word_index_map,
                          class_map=train.class_map)
        validation = TXTDataset(os.path.join(rpath, 'validation.txt'),
                                sparse=True,
                                word_index_map=train.word_index_map,
                                class_map=train.class_map)
        list_file.append({'train': train,
                          'test': test,
                          'validation': validation
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
            decompress(os.path.join(cfg['path'], cfg['zip']), cfg['path'])
            cfg['dataset_root_data_path'] = os.path.join(folder_path,
                                                         '12k_class')
    set_config('DMOZ12K', cfg)
    print(cfg)

    rpath = cfg['dataset_root_data_path']
    train = TXTDataset(os.path.join(rpath, 'train.txt'), sparse=True)
    test = TXTDataset(os.path.join(rpath, 'test.txt'), sparse=True,
                      word_index_map=train.word_index_map,
                      class_map=train.class_map)
    validation = TXTDataset(os.path.join(rpath, 'validation.txt'), sparse=True,
                            word_index_map=train.word_index_map,
                            class_map=train.class_map)

    return [{'train': train, 'test': test, 'validation': validation}]



def ALOI_data():
    cfg = get_config('ALOI')
    # dataset not downloaded
    if(cfg is None):
        dataset_path = os.path.realpath(os.path.join(default_dataset_path,
                                        'ALOI/'))
        try:
            shutil.rmtree(dataset_path)
        except Exception:
            pass
        os.makedirs(dataset_path)
        usual = ALOIDownloader(dataset_path)['usual']
        set_config('ALOI', {'path': dataset_path,
                   'zip': usual})

        cfg = get_config('ALOI')
    # dataset not decompressed
    if('ucp_dataset_root_data_path' not in cfg):
        folder_path = cfg['path']
        # decompress usual
        if(not os.path.exists(os.path.join(folder_path, 'aloi.scale'))):
            decompress(os.path.join(cfg['path'], cfg['zip']), cfg['path'])
            cfg['ucp_dataset_root_data_path'] = os.path.join(folder_path,
                                                             'aloi.txt')
    set_config('ALOI', cfg)
    rpath = cfg['ucp_dataset_root_data_path']
    if('dataset_root_data_path' not in cfg):
        # we split aloi in train val test
        print('splitting the dataset')
        dataset = TXTDataset(rpath,
                             sparse=False, max_lenght=128)
        try:
            os.makedirs(os.path.join(cfg['path'], 'ALOI1K'))
        except:
            pass
        dataset.build()
        # default value splitting into 80 train 10--10 val--test
        perm = torch.randperm(len(dataset))
        dtrain = copy.deepcopy(dataset)
        dtest = copy.deepcopy(dataset)
        dvalidation = copy.deepcopy(dataset)
        dtrain.remove_from_index(perm[int(0.8*perm.size(0)):].tolist())
        dtrain.save(os.path.join(cfg['path'], 'ALOI1K', 'train.txt'))

        dtest.remove_from_index(perm[:int(0.9*perm.size(0))].tolist())
        dtest.save(os.path.join(cfg['path'], 'ALOI1K', 'test.txt'))

        dvalidation.remove_from_index(perm[0:int(0.8*perm.size(0)):].tolist() +
                                      perm[int(0.9*perm.size(0)): ].tolist())
        dvalidation.save(os.path.join(cfg['path'], 'ALOI1K', 'validation.txt'))
        cfg['dataset_root_data_path'] = os.path.join(cfg['path'], 'ALOI1K')
        set_config('ALOI', cfg)
    rpath = cfg['dataset_root_data_path']
    train = TXTDataset(os.path.join(rpath, 'train.txt'), sparse=False,
                       max_lenght=128)
    train.setTrain()
    test = TXTDataset(os.path.join(rpath, 'test.txt'), sparse=False,
                      word_index_map=train.word_index_map,
                      class_map=train.class_map,
                      max_lenght=128)
    validation = TXTDataset(os.path.join(rpath, 'validation.txt'),
                            sparse=False,
                            word_index_map=train.word_index_map,
                            class_map=train.class_map,
                            max_lenght=128)

    return [{'train': train, 'test': test, 'validation': validation}]
