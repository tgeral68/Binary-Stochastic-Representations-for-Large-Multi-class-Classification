import numpy as np
import urllib.request as req
from tqdm import tqdm
from os.path import join


def download_progress_bar(url, filepath):
    file_name = url.split('/')[-1]
    u = req.urlopen(url)
    f = open(filepath, 'wb')
    meta = u.info()

    file_size = int(meta.get("Content-Length"))
    print("\n\nDownloading: "+file_name+" Bytes: "+str(file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % \
            (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print(status, end='\t')

    f.close()


def DMOZ1KDownloader(folder):
    root = 'http://webia.lip6.fr/~gerald/data/'
    download_progress_bar(root+'DMOZ1K.zip',
                          join(folder, 'DMOZ1K.zip'))
    return {'usual': 'DMOZ1K.zip'}


def DMOZ12KDownloader(folder):
    root = 'http://webia.lip6.fr/~gerald/data/'
    download_progress_bar(root+'DMOZ12K.zip',
                          join(folder, 'DMOZ12K.zip'))
    return {'usual': 'DMOZ12K.zip'}


def ALOIDownloader(folder):
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/aloi.scale.bz2'
    download_progress_bar(url,
                          join(folder, 'aloi.scale.bz2'))
    return {'usual': 'aloi.scale.bz2'}
