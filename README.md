# Binary-Stochastic-Representations-for-Large-Multi-class-Classification

Implementation of the paper Binary-Stochastic-Representations-for-Large-Multi-class-Classification in pytorch.


## Dependency ##
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
pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl 
pip install torchvision
pip install tqdm urllib numpy scikit-learn pandas argparse zipfile shutil
```
## Example ##

For the DMOZ Dataset you can start the training step by: 

```
python DSNC_DMOZ_Train_Triplet.py --code_size 36 --dataset 1K --iteration 100
```


For DMOZ the dataset used will be automatically downloaded, for Imagenet you must Download the Imagenet 2012 Dataset and get the last layer of RESNET-152 and then create the dataset (if you save the features in txt format you can load it with TXTDataset object). For Aloi you must Download The aloi Dataset from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html

### Option ###
* --dataset - The dataset to train the model (you can add yours see dataset to know how)
 ** 1k, 12K, ALOI
* --code_size The number of bits to use in the representation space
