from os import walk
from PIL import Image
import os, shutil, csv, random, magic, imghdr, sys, struct

def img_size_type(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    im = Image.open(fname)
    img_type = imghdr.what(base_dir + source_dir + j + '/' + name)
    return im.size, img_type


if len(sys.argv) == 1 or sys.argv[1] == 'clean':

    # example dir: /data/private/hymen_test/train/bee/1092977343_cb42b38d62.jpg
    # example dir: /data/private/clean_data/AD/fd3b93d9d53bba352bd1be3c245bc02c
    # num_file = 1000
    base_dir = '/data/private'
    source_dir = '/clean_data/'
    data_dir = '/learn'
    class_lst = ['ADULT', 'AD', 'ILLEGALITY', 'NORMAL', 'SEMI_ADULT', 'HATRED']


elif sys.argv[1] == 'catdog':
    num_file = 500
    base_dir = '/data/private'
    source_dir = '/catdog_data/'
    data_dir = '/hymen_test'
    class_lst = ['Cat', 'Dog']

for i in class_lst:
    # remove current dataset
    if os.path.isdir(base_dir + data_dir + '/train/' + i):
        print('Working on \ttrain ' + i)
        os.system('for i in ' + base_dir + data_dir + '/train/' + i+ '/*; do echo "Processing $i"; exiftool -all= "$i"; done')

    if os.path.isdir(base_dir + data_dir + '/val/' + i):
        print('Working on \tval ' + i)
        os.system('for i in ' + base_dir + data_dir + '/val/' + i+ '/*; do echo "Processing $i"; exiftool -all= "$i"; done')

print('\nDirectories created \n')
