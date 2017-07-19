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

## Example ##

For the DMOZ Dataset you can start the training step by: 

```
python DSNC_DMOZ_Train_Triplet.py --code-size 36 --dataset 1K
```
For DMOZ the dataset used will be automatically downloaded, for Imagenet you must Download the Imagenet 2012 Dataset and get the last layer of RESNET-152. For Aloi you must Download The aloi Dataset from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
