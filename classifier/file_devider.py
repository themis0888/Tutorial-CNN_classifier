"""
This program distribute every images in each categories to train data and validation data.

you have 6 category of images in below directory
/base_dir/source_dir
 ├── AD
 ├── ADULT
 ├── HATRED
 ├── ILLEGALITY
 ├── NORMAL
 └── SEMI_ADULT

after you run 'python file_devider.py', you will get 
/base_dir/data_dir
├── train
│   ├── AD
│   ├── ADULT
│   ├── HATRED
│   ├── ILLEGALITY
│   ├── NORMAL
│   └── SEMI_ADULT
└── val
    ├── AD
    ├── ADULT
    ├── HATRED
    ├── ILLEGALITY
    ├── NORMAL
    └── SEMI_ADULT


And I'm pretty sure that there must be much more faster method in pytorch 
which is doing exactly same thing. :D 
"""

from os import walk
from PIL import Image
import os, shutil, csv, random, magic, imghdr, sys, struct

def img_size_type(fname, src_dir, dest_dir):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    im = Image.open(src_dir + fname)
    im.load()
    im.convert('RGB').save(dest_dir + fname + '.jpg', 'JPEG')
    return im.size, im.format


if len(sys.argv) == 1 or sys.argv[1] == 'clean':

    # example dir: /data/private/hymen_test/train/bee/1092977343_cb42b38d62.jpg
    # example dir: /data/private/clean_data/AD/fd3b93d9d53bba352bd1be3c245bc02c
    # num_file = 10000
    base_dir = '/data/private'
    source_dir = '/clean_data/'
    data_dir = '/learn'
    # 'ADULT', 'AD', 'ILLEGALITY', 'NORMAL', 'SEMI_ADULT', 'HATRED'
    class_lst = ['AD', 'ILLEGALITY']


elif sys.argv[1] == 'catdog':
    num_file = 500
    base_dir = '/data/private'
    source_dir = '/catdog_data/'
    data_dir = '/hymen_test'
    class_lst = ['Cat', 'Dog']


print('From \t %s' % (base_dir + source_dir))
print('To \t %s' % (base_dir + data_dir))

# make the dataset directory 
# If something goes wrong, do 're -r *' at the /learn/ dir and go through this again

for i in class_lst:
    # remove current dataset
    if os.path.isdir(base_dir + data_dir + '/train/' + i):
        shutil.rmtree(base_dir + data_dir + '/train/' + i)
    os.makedirs(base_dir + data_dir + '/train/' + i)

    if os.path.isdir(base_dir + data_dir + '/val/' + i):
        shutil.rmtree(base_dir + data_dir + '/val/' + i)
    os.makedirs(base_dir + data_dir + '/val/' + i)

print('\nDirectories created \n')

test_idx = 0
# put num_file files in /clean_data/ to the /learn/
for j in class_lst:

    print('Working on \t ' + j)
    
    format_dic = dict()

    # randomly shffle the files and divide 
    dir_lst = os.listdir(base_dir + source_dir + j + '/')
    # random.shuffle(dir_lst)
    # num_file = 50 
    num_file = len(dir_lst)

    for name in dir_lst[:num_file // 2]:

        try:
            image_size, img_format = img_size_type(name, 
                base_dir + source_dir + j + '/',
                base_dir + data_dir + '/train/' + j + '/')
        except IOError:
            # print('excepted', end = ' ')
            continue
        else:
            # Pass when the file type isn't supported by PIL 
            # Pass when image is smaller than input size (255, 255)
            if (not ((256 < image_size[0]) and (256 < image_size[1]))):
                continue 
        print('Working on \ttrain\t ' + j)   



    for name in dir_lst[num_file // 2 + 1 : num_file]:
        
        try:
            image_size, img_format = img_size_type(name, 
                base_dir + source_dir + j + '/',
                base_dir + data_dir + '/val/' + j + '/')
        except IOError:
            # print('excepted', end = ' ')
            continue
        else:
            # Pass when the file type isn't supported by PIL 
            # Pass when image is smaller than input size (255, 255)
            if (not ((256 < image_size[0]) and (256 < image_size[1]))):
                continue 
        print('Working on \tval\t ' + j)  







