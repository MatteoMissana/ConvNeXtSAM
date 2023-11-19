import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import imghdr
from pathlib import Path

def list_directories(path):
    # Get a list of all items in the directory
    all_items = os.listdir(path)

    # Filter out only directories
    directories = sorted([item for item in all_items if os.path.isdir(os.path.join(path, item))])

    return directories

def extract_images_from_directory(directory_path):
    # Ensure the directory path is absolute
    directory_path = os.path.abspath(directory_path)

    # Create a Path object for the directory
    directory_path = Path(directory_path)

    # Initialize a list to store image file paths
    image_paths = []

    # Iterate over files in the directory
    for file_path in directory_path.iterdir():
        # Check if the file is a regular file and is an image
        if file_path.is_file() and imghdr.what(file_path):
            image_paths.append(file_path)
    return sorted(image_paths)

def create_directory(directory_path):
    # Ensure the directory path is absolute
    directory_path = os.path.abspath(directory_path)

    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create the directory and any necessary parent directories
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")


def save_difs_mean(path_dir_sam, path_dir_CneXt, path_save):

    #create path_save directory
    create_directory(path_save)

    #extract the directories containing the numpy arrays
    dir_sam=list_directories(path_dir_sam)
    dir_cnext=list_directories(path_dir_CneXt)

    #extract image paths
    list_sam_imgs=extract_images_from_directory(path_dir_sam)
    print(list_sam_imgs[0])
    list_cnext_imgs=extract_images_from_directory(path_dir_CneXt)

    for j, d in enumerate(dir_sam):
        create_directory(os.path.join(path_save, d))
        for i in ['26','30','34']:

            npy_sam_paths=[]
            npy_cnext_paths=[]

            for s in os.listdir(os.path.join(path_dir_sam,d)):
                npy_sam_paths.append(os.path.join(path_dir_sam, dir_sam[j], s))
                npy_cnext_paths.append(os.path.join(path_dir_CneXt, dir_cnext[j], s))

            print(npy_sam_paths[0])
            #extract numpy array and make the difference
            npy_sam = [item for item in npy_sam_paths if 'stage' + i in item]
            npy_cnext = [item for item in npy_cnext_paths if 'stage' + i in item]


            diff= np.load(npy_sam[0])-np.load(npy_cnext[0])
            mean= np.mean(diff, axis=0)

            #save as image the mean
            fig, ax = plt.subplots()
            im = ax.imshow(mean, cmap='jet')
            plt.colorbar(im, ax=ax)
            create_directory(os.path.join(path_save, d, 'dif_stage{}'.format(i)))
            plt.savefig(os.path.join(path_save, d, 'dif_stage{}/mean.png'.format(i)), bbox_inches='tight',
                        pad_inches=0.1)
            plt.close()

        #transfer the prediction from sam and conv directories to save
        if str(list_sam_imgs[j]).endswith('.png'):
            predsam_path= os.path.join(path_save,d,'SAM_prediction.png')
            predcnext_path = os.path.join(path_save, d, 'ConvNeXt_prediction.png')
            flag=True

        elif str(list_sam_imgs[j]).endswith('.jpg'):
            predsam_path= os.path.join(path_save,d,'SAM_prediction.jpg')
            predcnext_path = os.path.join(path_save, d, 'ConvNeXt_prediction.jpg')
            flag=True
        else:
            flag=False

        if flag:
            shutil.copy(list_sam_imgs[j], predsam_path)
            shutil.copy(list_cnext_imgs[j], predcnext_path)
        else:
            print('warning: image type not supported, cant copy prediction in the new directory')


def save_difs(path_dir_sam, path_dir_CneXt, path_save):

    #create path_save directory
    create_directory(path_save)

    #extract the directories containing the numpy arrays
    dir_sam=list_directories(path_dir_sam)
    dir_cnext=list_directories(path_dir_CneXt)

    #extract image paths
    list_sam_imgs=extract_images_from_directory(path_dir_sam)
    list_cnext_imgs=extract_images_from_directory(path_dir_CneXt)

    for j, d in enumerate(dir_sam):
        create_directory(os.path.join(path_save, d))
        for i in ['26','30','34']:

            npy_sam_paths=[]
            npy_cnext_paths=[]

            for s in os.listdir(os.path.join(path_dir_sam,d)):
                npy_sam_paths.append(os.path.join(path_dir_sam, dir_sam[j], s))
                npy_cnext_paths.append(os.path.join(path_dir_CneXt, dir_cnext[j], s))

            print(npy_sam_paths[0])
            #extract numpy array and make the difference
            npy_sam = [item for item in npy_sam_paths if 'stage' + i in item]
            npy_cnext = [item for item in npy_cnext_paths if 'stage' + i in item]


            diff= np.load(npy_sam[0])-np.load(npy_cnext[0])
            mean= np.mean(diff, axis=0)

            #save all single maps as images
            for k, map in enumerate(diff):
                if not os.path.isfile(os.path.join(path_save, d, 'dif_stage{}/map_{}.png'.format(i, k))):
                    fig, ax = plt.subplots()
                    im = ax.imshow(map, cmap='jet')
                    plt.colorbar(im, ax=ax)
                    create_directory(os.path.join(path_save, d, 'dif_stage{}'.format(i)))
                    plt.savefig(os.path.join(path_save, d, 'dif_stage{}/map_{}.png'.format(i, k)), bbox_inches='tight',
                                pad_inches=0.1)
                    plt.close()

            #save as image the mean
            fig, ax = plt.subplots()
            im = ax.imshow(mean, cmap='jet')
            plt.colorbar(im, ax=ax)
            create_directory(os.path.join(path_save, d, 'dif_stage{}/mean'.format(i)))
            plt.savefig(os.path.join(path_save, d, 'dif_stage{}/mean/mean.png'.format(i)), bbox_inches='tight',
                        pad_inches=0.1)
            plt.close()

        #transfer the prediction from sam and conv directories to save
        if str(list_sam_imgs[j]).endswith('.png'):
            predsam_path= os.path.join(path_save,d,'SAM_prediction.png')
            predcnext_path = os.path.join(path_save, d, 'ConvNeXt_prediction.png')
            flag=True

        elif str(list_sam_imgs[j]).endswith('.jpg'):
            predsam_path= os.path.join(path_save,d,'SAM_prediction.jpg')
            predcnext_path = os.path.join(path_save, d, 'ConvNeXt_prediction.jpg')
            flag=True
        else:
            flag=False

        if flag:
            shutil.copy(list_sam_imgs[j], predsam_path)
            shutil.copy(list_cnext_imgs[j], predcnext_path)
        else:
            print('warning: image type not supported, cant copy prediction in the new directory')







