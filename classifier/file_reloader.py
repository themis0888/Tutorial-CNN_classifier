from os import walk
from PIL import Image
import os, shutil, csv, random, magic, imghdr, sys, struct

if len(sys.argv) == 1 or sys.argv[1] == 'clean':

    # example dir: /data/private/hymen_test/train/bee/1092977343_cb42b38d62.jpg
    # example dir: /data/private/clean_data/AD/fd3b93d9d53bba352bd1be3c245bc02c
    # num_file = 10000
    base_dir = '/data/private'
    source_dir = '/learn/val/'
    data_dir = '/learn'
    # 'ADULT', 'AD', 'ILLEGALITY', 'NORMAL', 'SEMI_ADULT', 'HATRED'
    class_lst = ['SEMI_ADULT']
    tv_ratio = 3/5


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

for j in class_lst:

    print('Working on \t ' + j)
    
    format_dic = dict()

    # randomly shffle the files and divide 
    dir_lst = os.listdir(base_dir + source_dir + j + '/')
    # random.shuffle(dir_lst)
    # num_file = 50 
    num_file = len(dir_lst)
    idx = 0
    for name in dir_lst[:round(num_file * tv_ratio)]:
        os.rename(base_dir + data_dir + '/val/' + j + '/' + name,
            base_dir + data_dir + '/train/' + j + '/' + name)
        
        if idx % 1000 == 0:
            print('{:.4f}% \tDone'.format(idx/(num_file*tv_ratio)))

        idx += 1









