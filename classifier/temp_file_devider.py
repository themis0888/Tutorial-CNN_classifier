from os import walk
from PIL import Image
import os, shutil, csv, random, magic, imghdr, sys, struct

def img_size_type(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    im = Image.open(fname)
    img_type = imghdr.what(base_dir + source_dir + j + '/' + name)
    return im.size, img_type, im.format


if len(sys.argv) == 1 or sys.argv[1] == 'clean':

    # example dir: /data/private/hymen_test/train/bee/1092977343_cb42b38d62.jpg
    # example dir: /data/private/clean_data/AD/fd3b93d9d53bba352bd1be3c245bc02c
    # num_file = 10000
    base_dir = '/data/private'
    source_dir = '/learn/'
    data_dir = '/learn'
    # 'ADULT', 'AD', 'ILLEGALITY', 'NORMAL', 'SEMI_ADULT', 'HATRED'
    class_lst = ['ADULT', 'AD', 'ILLEGALITY', 'NORMAL', 'SEMI_ADULT', 'HATRED']


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
    
    # randomly shffle the files and divide 
    dir_lst = os.listdir(base_dir + source_dir + '/train/' + j + '/')

    for name in dir_lst:
        """
        try:
            image_size, image_type, img_format = img_size_type(base_dir + source_dir + j + '/' + name)
        except:
            print('excepted', end = ' ')
            continue
        else:
            # Pass when the file type isn't supported by PIL 
            # Pass when image is smaller than input size (255, 255)
            if ((image_type not in ['jpg', 'jpeg']) 
                or (not ((256 < image_size[0]) and (256 < image_size[1])))
                or img_format not in ['JPEG', 'JPG']):
                continue 

        # Put the image in the directory
        shutil.copyfile(base_dir + source_dir + j + '/' + name, 
            base_dir + data_dir + '/train/'+ j + '/' + name) # + '.' + file_type)
        print('Working on \t ' + j)   
        """
        os.rename(base_dir + data_dir + '/train/'+ j + '/' + name, 
            base_dir + data_dir + '/train/'+ j + '/' + name + '.jpg')


    dir_lst = os.listdir(base_dir + source_dir + '/val/' + j + '/')

    for name in dir_lst:
        """
        try:
            image_size, image_type = img_size_type(base_dir + source_dir + j + '/' + name)
        except:
            # print('excepted', end = ' ')
            continue
        else:
            # Pass when the file type isn't supported by PIL 
            # Pass when image is smaller than input size (255, 255)
            if ((image_type not in ['jpg', 'jpeg']) 
                or (not ((256 < image_size[0]) and (256 < image_size[1])))):
                continue 
        
        shutil.move(base_dir + source_dir + j + '/' + name, 
            base_dir + data_dir + '/val/' + j + '/') # + '.' + file_type)
        print('Working on \t ' + j)
        """
        os.rename(base_dir + data_dir + '/val/'+ j + '/' + name, 
            base_dir + data_dir + '/val/'+ j + '/' + name + '.jpg')









