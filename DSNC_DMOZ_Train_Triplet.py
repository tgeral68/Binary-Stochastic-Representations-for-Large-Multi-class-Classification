
import pandas as pd
import argparse
import tqdm
import time
import sys
import os


# pytorch import
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F

# local library

# sparse modules
from module.stochastic import STEECOCSparseLinearTriplet
from module.sparse import SparseLinear, EmbeddingSum

# data loader and tools
from datatools.dataset import TXTDataset, TXTDatasetTriplet, DMOZ_1K_data,\
    DMOZ_12K_data


# usefull functions to build the ecoc codes
from utils.ecoc_utils import MHamming_random_codes_fast
from utils.ecoc_utils import Hamming_distanceFast


# parsing the options
parser = argparse.ArgumentParser()
parser.add_argument('--code_size', nargs='*', default=[10],
                    help='Binary representation size')
parser.add_argument('--cuda', action='store_true',
                    help='Enable gpu processing')
parser.add_argument('--iteration', type=int, default=50,
                    help='number of training epochs')
parser.add_argument('--learning_rate', type=float,
                    help='Learning rate', default=1e-2)
parser.add_argument('--weight_decay', type=float,
                    help='Weight decay (l2 regularisation)', default=0.)
parser.add_argument('--prefix', type=str, default='',
                    help='A prefix name for saving the model')
parser.add_argument('--dataset', type=str, default='1K',
                    help='DMOZ "1K" or "12K"'
                    )
args = parser.parse_args()


# defining the seed
torch.manual_seed(25)
torch.cuda.manual_seed(25)


def fit(model, dataset_train, dataset_validation, optimizer, criterion,
        check_val=10, nb_iteration=50, batch_size=1000, counter_max=20,
        cuda=False, ef=1e-6):
    # define log variables :
    # -the accuracy train
    # -the accuracy validation
    # -the best validation obtained
    # -the values of losses
    # -the model associate to the best
    # validation encountered during training
    # -a counter to quit if the validation performances do not increase
    # during counter_max * check_val iteration
    at = []
    av = []
    bv = 0.
    losses = []
    losses_dist = []
    losses_rapr = []
    bmp = 'tmp/'+str(time.time())+'.pytorch'
    counter = 0
    elastica_factor = ef
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    # if tmp folder does not exist we create it
    try:
        os.makedirs('tmp')
    except:
        pass
    # we define the selection tensor to not
    # allocate a new tensor at each times
    # and allowing to select even with cuda
    selector = dataset_train.X.new().long()

    # surrounded by try and catch to save
    # even if there is memory error
    try:
        progress_bar = tqdm.trange(nb_iteration)
        for i, pb in enumerate(progress_bar):
            model.train()
            losses.append(.0)
            losses_dist.append(.0)
            losses_rapr.append(.0)
            at.append(.0)
            if(cuda):
                selector = selector.cuda()
            selector.resize_(len(dataset_train)).\
                copy_(torch.randperm(len(dataset_train)))
            randperm = torch.split(selector, batch_size)
            model.ste.slope = model.ste.slope * 1.0001
            for k, d in enumerate(dataloader):
                optimizer.zero_grad()
                x, y = Variable(d[0]), Variable(d[1])
                p, p2, p3 = model(x)

                loss = criterion(F.log_softmax(p),  y[:, 0])
                loss_c = (((model.cencoded1.sigmoid() - model.cencoded3.sigmoid()))**2).sum(1).sum()
                loss_c2 = (model.cencoded1.size(1) - (((model.cencoded1.sigmoid() - model.cencoded2.sigmoid()))**2).sum(1)).sum()
                tloss = loss\
                    +\
                    elastica_factor*loss_c +\
                    elastica_factor*loss_c2
                tloss.backward()
                optimizer.step()
                losses[-1] += loss.data[0]
                losses_dist[-1] += loss_c2.data[0]
                losses_rapr[-1] += loss_c.data[0]
                at[-1] += (y.data == p.data.max(1)[1]).sum()
            at[-1] /= len(dataset_train)
            at[-1] = ((at[-1] * 10000)//1)/100
            losses[-1] /= len(dataset_train)
            losses_dist[-1] /= len(dataset_train)
            losses_rapr[-1] /= len(dataset_train)
            # every 10 iteration we look at the global system performances on
            # the validation set and save it if best accuracy is reach
            if(i % check_val == 0):
                model.eval()
                # new entry for accuracy logs
                av.append(.0)
                selector = selector.cpu()
                selector.resize_(len(dataset_validation)).\
                    copy_(torch.randperm(len(dataset_validation)))
                randperm = torch.split(selector, batch_size)
                for k, d in enumerate(randperm):
                    key = Variable(dataset_validation.
                                   X[d])
                    if(cuda):
                        key = key.cuda()

                    p = model(key)
                    y_truth = dataset_validation.Y[d]
                    if(cuda):
                        y_truth = y_truth.cuda()
                    av[-1] += (y_truth == p.data.max(1)[1]).sum()
                av[-1] = (((float(av[-1])/len(dataset_validation))
                           * 10000)//1)/100
                # if we reach the best validation
                # we save thel model
                if(av[-1] > bv):
                    torch.save(model, bmp)
                    bv = av[-1]

                    elastica_factor *= 2.
                    counter = 0
                else:
                    elastica_factor /= 2.
                    counter += 1
            progress_bar.set_postfix(
                                     loss=losses[-1],
                                     at=str(at[-1])+'%',
                                     av=str(av[-1])+'%',
                                     bv=str(bv)+'%',
                                     ld=losses_dist[-1],
                                     lr=losses_rapr[-1],
                                     elas=elastica_factor
                                    )
            # if no increasing validation performances -> quit
            if(counter > counter_max):
                print("Quit : performances do not increase during " +
                      str(counter_max * check_val) + " iterations")
                training_logs = {'losses': losses,
                                 'loss_repulse': losses_dist,
                                 'loss_raproach': losses_rapr,
                                 'accuracies_train': at,
                                 'accuracy_validation': av,
                                 'best_val': bv,
                                 'best_model_path': bmp}
                return training_logs
    except Exception as ex:
        traceback.print_exc()
        print("An error occured, attempting to save logs and models")
        training_logs = {'losses': losses,
                         'loss_repulse': losses_dist,
                         'loss_raproach': losses_rapr,
                         'accuracies_train': at,
                         'accuracy_validation': av,
                         'best_val': bv,
                         'best_model_path': bmp}
        return training_logs
    training_logs = {'losses': losses,
                     'loss_repulse': losses_dist,
                     'loss_raproach': losses_rapr,
                     'accuracies_train': at,
                     'accuracy_validation': av,
                     'best_val': bv,
                     'best_model_path': bmp}
    return training_logs


def get_codes(dataset, model, batch_size=1000):
    predict = torch.LongTensor()
    code = torch.FloatTensor()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for data in dataloader:
        x, y = Variable(data[0]), Variable(data[1])
        p = model(x)
        # print(p.data.max(1)[1])
        # print(predict)
        predict = torch.cat((predict, p.data.max(1)[1]), 0)
        code = torch.cat((code, model.ccode1.data), 0)
    return code, predict
# iterate over models


def main(args):
    list_dataset_path = None
    if(args.dataset == '1K'):
        list_dataset_path = DMOZ_1K_data()
    elif(args.dataset == '12K'):
        list_dataset_path = DMOZ_12K_data()
    # log view at the end of the learning process
    accuracy = {}  # best validation accuracy
    accuracy_kd = {}  # best validation accuracy by knn
    st = 0.  # time variable
    # for each set
    for dmoz_set in list_dataset_path:
        print("Loading data...", end=" ")
        sys.stdout.flush()
        st = time.time()
        dtrain = TXTDatasetTriplet(dmoz_set['train'], class_map={})
        dvalidation = TXTDataset(dmoz_set['validation'],
                                 word_index_map=dtrain.word_index_map,
                                 class_map=dtrain.class_map)
        dtest = TXTDataset(dmoz_set['test'],
                           word_index_map=dtrain.word_index_map,
                           class_map=dtrain.class_map)
        print(str(time.time() - st)[0:5]+'s')

        for code_size_str in args.code_size:
            code_size = int(code_size_str)

            # building the model
            model = STEECOCSparseLinearTriplet(code_size,
                                               len(dtrain.word_index_map)+1,
                                               len(dtrain.class_map))

            # define the optimization method
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay
                                   )
            if(args.cuda):
                model.cuda()
                dtrain.X = dtrain.X.cuda()
                dtrain.Y = dtrain.Y.cuda()
            # start learning process
            logs = fit(model,
                       dtrain,
                       dvalidation,
                       optimizer,
                       nn.NLLLoss(size_average=True),
                       nb_iteration=args.iteration
                       )
            # -------------------------------------------------------------- #
            # evaluate the model
            best_model = torch.load(logs['best_model_path'])
            best_model.cpu()
            dataloader = DataLoader(dtest, batch_size=1000)
            best_model.eval()
            acc_test = 0
            best_model.eval()
            for k, d in enumerate(dataloader):
                optimizer.zero_grad()
                x, y = Variable(d[0]), Variable(d[1])
                p = best_model(x)

                acc_test += (y.data == p.data.max(1)[1]).sum()
            acc_test /= len(dtest)
            acc_test = ((acc_test * 10000)//1)/100
            accuracy[code_size].append(acc_test)

            # evaluation on the k-nearest neigbhour methods
            # obtain train codes and their classe value
            dtrain.X = dtrain.X.cpu()
            dtrain.Y = dtrain.Y.cpu()
            ground_truth_train = dtrain.Y
            ground_truth_test = dtest.Y
            train_codes, predicted_classes_train = get_codes(dtrain,
                                                             best_model)
            test_codes, predicted_classes_test = get_codes(dtest, best_model)

            # computing knn accuracy
            knn_index = [Hamming_distanceFast(code, train_codes).min(0)[1]
                         for code in test_codes]
            accuracy_predicted_codes = \
                ((predicted_classes_train.index_select(0,
                 torch.cat(knn_index, 0).squeeze())
                 == ground_truth_test).sum())/len(dtest)
            accuracy_truth_codes = \
                ((ground_truth_train.index_select(0,
                 torch.cat(knn_index, 0).squeeze())
                 == ground_truth_test).sum())/len(dtest)
            accuracy_kd[code_size].append((((accuracy_predicted_codes * 10000)//1)
                                          / 100, ((accuracy_truth_codes * 10000)//1)
                                                                        / 100 ))
            # saving the logs and the models
            print(accuracy_kd[code_size])
            print(accuracy[code_size])
            print(best_model)
            logs['accuracy_test'] = (accuracy[code_size])
            logs['accuracy_test_kd'] = (accuracy_kd[code_size])

            try:
                os.makedirs('./log')
            except Exception:
                pass
            torch.save({"model": best_model, "logs": logs},
                       './logs/'+args.prefix+'_\
    ste_DMOZ_triplet_'+str(code_size)+'_'+str(dfolder)+'.pytorch')

    print(pd.DataFrame(accuracy))
    print(pd.DataFrame(accuracy_kd))


if __name__ == "__main__":
    main(args)
