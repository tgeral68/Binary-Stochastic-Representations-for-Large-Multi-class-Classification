from os import path
import bz2
import zipfile as zip
import mimetypes


def decompress(input_path, output_path):
    print(input_path)
    ext = mimetypes.guess_type(input_path)
    ext = str(ext[0]) + str(ext[1])
    if(ext is None):
        # we try to read the mime types inside the head of the file
        ext = mimetypes.read_mime_types(input_path)

    if(ext is None):
        raise UnknowMimeType(path.split(input_path)[1])

    if('bzip2' in ext):
        with open(input_path, 'rb') as zipfile:

            data = bz2.decompress(zipfile.read())
            print(output_path+(path.split(input_path)[1].split('.')[0]))
            f = open(path.join(output_path,
                     (path.split(input_path)[1].split('.')[0])+'.txt'), 'wb')
            f.write(data)
    elif('zip' in ext):
        zip_ref = zip.ZipFile(input_path, 'r')
        zip_ref.extractall(output_path)
        zip_ref.close()

    else:
        raise NotImplementedError()
###############################################
# Exception


class UnknowMimeType(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'MimeType is undefined for the file "'+self.value()+'"'
