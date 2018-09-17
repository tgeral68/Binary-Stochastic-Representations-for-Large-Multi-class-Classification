# Binary Stochastic Representations for Large Multi class Classification

Implementation of the paper Binary-Stochastic-Representations-for-Large-Multi-class-Classification in pytorch.


## Requirement ##
* pytorch
* tqdm
* urllib
* numpy
* sklearn
* pandas
* argparse
* zipfile
* shutil


To install all the dependency package you can use pip as following:


```
pip install torch torchvision
pip install tqdm urllib numpy scikit-learn pandas argparse zipfile shutil
```

We recommend to use python 3+ since the code have been develloped for this version.

## Example ##

### Training a model ###
For the DMOZ Dataset you can start the training step by: 

```
python train.py --code_size 36 --dataset 1K --iteration 100 --folder ./tmp
```


In this particular case we learn a model with a latent space using 36 dimenssion (bits) on the DMOZ-1K dataset
with a maximum of 100 iteration. The best models in term of validation will be saved in ./tmp folder (in the case of 
DMOZ-1K we got 5 split thus 5 models is trained). Moreover an automatic evaluation will be performed at the end of the 
learning step and result are stored in the "log" folder  (dictionary pytorch format).



For DMOZ the datasets used in the paper will be automatically downloaded in dataset folder, for Imagenet you must Download the Imagenet 2012 Dataset and get the last layer of RESNET-152 and then create the dataset (if you save the features in txt format you can load it with TXTDataset object). 

For Aloi you must Download The aloi Dataset from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html

### Evaluate a model ###
```
python eval.py  --dataset 1K --folder ./tmp
```

Evaluate models in ./tmp on the DMOZ-1K dataset: mean accuracy using standart KNN on the embeddings.



### Option ###
* --dataset - The dataset to train the model (you can add yours see dataset to know how)
  * 1k, 12K, ALOI
* --code_size The number of bits to use in the representation space
* --iteration Maximum number of iterations (default: 50)
* --folder the path were will be saved the models (default: tmp/)
* --learning_rate the initial learning rate (default: 1e-2)
* --weight_decay l2 regularization (default: 0)
* --cuda use gpu (currently broken)
* -h show all the options description

