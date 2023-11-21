import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def save_difs(path_dir_sam, path_dir_CneXt, path_save):

    if not os.path.isdir(path_save):
        os.mkdir(path_save)

    ckpt_path = os.path.join(path_save, 'checkpoint.txt')

    folder_sam_list = sorted(os.listdir(path_dir_sam))
    folder_cnext_list = sorted(os.listdir(path_dir_CneXt))
    sep = ','

    if not os.path.isfile(ckpt_path):
        f = open(ckpt_path, 'w')
        dim_ckpt = "26"
        img_ckpt = folder_sam_list[0]
        map_ckpt = "0"
        f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
        f.close()

    else:
        f = open(ckpt_path, 'r')
        ckpt = f.read().split(',')
        dim_ckpt = ckpt[0]
        img_ckpt = ckpt[1]
        map_ckpt = ckpt[2]
        f.close()

    for num in ["26", "30", "34"]:
        print('stage', num)
        if num == dim_ckpt:
            for indx,folder_sam in enumerate(folder_sam_list):
                print('i 1',indx)

                '''
                if indx >= 6:
                    #print(indx,dim_ckpt,img_ckpt,map_ckpt)
                    img_ckpt = folder_sam_list[0]
                    f = open(ckpt_path, 'w')
                    f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
                    f.close()
                    break
                '''
                if folder_sam == img_ckpt:
                    print('folder sam',folder_sam)
                    for indx2,folder_Cnext in enumerate(folder_cnext_list):
                        print('i 2', indx2)
                        if folder_Cnext == folder_sam and (('.png' or '.jpg')not in folder_sam):
                            print('folder convnext', folder_Cnext)
                            for files_sam in sorted(os.listdir(os.path.join(path_dir_sam, folder_sam))):
                                if "stage{}".format(num) == files_sam.split('_')[0]:
                                    print('sam_file',files_sam)
                                    for files_Cnext in sorted(os.listdir(os.path.join(path_dir_CneXt, folder_Cnext))):
                                        if "stage{}".format(num) == files_Cnext.split('_')[0]:
                                            print('Cnext_file', files_Cnext)

                                            if num == "26":
                                                s = 'l'
                                            elif num == "30":
                                                s = 'm'
                                            elif num == "34":
                                                s = 's'

                                            name = folder_sam

                                            if not os.path.isdir(os.path.join(path_save,name)):
                                                os.mkdir(os.path.join(path_save,name))

                                            if not os.path.isfile(os.path.join(path_save,name,name+'_convnextsam'+'.png')):
                                                shutil.copy(os.path.join(path_dir_sam,name+'.png'),os.path.join(path_save,name,name+'_convnextsam'+'.png'))

                                            if not os.path.isfile(os.path.join(path_save,name,name+'_convnext'+'.png')):
                                                shutil.copy(os.path.join(path_dir_CneXt,name+'.png'),os.path.join(path_save,name,name+'_convnext'+'.png'))

                                            for i in ('s', 'm', 'l'):
                                                if not os.path.isdir(os.path.join(path_save,name,'dif_{}'.format(i))):
                                                    os.mkdir(os.path.join(path_save,name,'dif_{}'.format(i)))
                                                if not os.path.isdir(os.path.join(path_save,name,'dif_{}/mean'.format(i))):
                                                    os.mkdir(os.path.join(path_save,name,'dif_{}/mean'.format(i)))

                                            sam = np.load(os.path.join(path_dir_sam, folder_sam, files_sam))
                                            Cnext = np.load(os.path.join(path_dir_CneXt, folder_Cnext, files_Cnext))

                                            dif = sam - Cnext

                                            for i, map in enumerate(dif):
                                                if i >= int(map_ckpt):
                                                    map_ckpt = str(i)
                                                    f = open(ckpt_path,'w')
                                                    f.write(sep.join([dim_ckpt,img_ckpt,map_ckpt]))
                                                    f.close()

                                                    if not os.path.isfile(os.path.join(path_save,name,'dif_{}/map_{}.png'.format(s,i))):
                                                        # plt.imsave('dif_s/map_{}.png'.format(i),map)
                                                        fig, ax = plt.subplots()
                                                        im = ax.imshow(map, cmap='jet')
                                                        plt.colorbar(im, ax=ax)
                                                        plt.savefig(os.path.join(path_save,name,'dif_{}/map_{}.png'.format(s,i)), bbox_inches='tight', pad_inches=0.1)
                                                        plt.close()

                                                    if not os.path.isfile(os.path.join(path_save,name,'dif_{}/mean/mean.png'.format(s))):
                                                        mean = np.mean(dif, axis=0)
                                                        # plt.imsave('dif_s/mean/mean.png',mean)
                                                        fig, ax = plt.subplots()
                                                        im = ax.imshow(mean, cmap='jet')
                                                        plt.colorbar(im, ax=ax)
                                                        plt.savefig(os.path.join(path_save,name,'dif_{}/mean/mean.png'.format(s)), bbox_inches='tight', pad_inches=0.1)
                                                        plt.close()

                                            map_ckpt = "0"
                                            f = open(ckpt_path, 'w')
                                            f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
                                            f.close()

                    if indx == len(folder_sam_list)-1:
                        img_ckpt = folder_sam_list[0]
                    else:
                        img_ckpt = folder_sam_list[indx+1]

                    f = open(ckpt_path, 'w')
                    f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
                    f.close()

            img_ckpt = folder_sam_list[0]
            dim_ckpt = "{}".format(int(dim_ckpt) + 4)
            f = open(ckpt_path, 'w')
            f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
            f.close()

    dim_ckpt = str(26)
    f = open(ckpt_path, 'w')
    f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
    f.close()

    os.remove(ckpt_path)
    return

def save_difs_mean(path_dir_sam, path_dir_CneXt, path_save):
    if not os.path.isdir(path_save):
        os.mkdir(path_save)

    ckpt_path = os.path.join(path_save, 'checkpoint.txt')

    print(path_dir_sam)
    folder_sam_list = sorted(os.listdir(path_dir_sam))
    '''folder_cnext_list = sorted(os.listdir(path_dir_CneXt))'''
    folder_cnext_list = sorted(os.listdir(path_dir_CneXt))
    sep = ','

    if not os.path.isfile(ckpt_path):
        f = open(ckpt_path, 'w')
        dim_ckpt = "26"
        img_ckpt = folder_sam_list[0]
        map_ckpt = "0"
        f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
        f.close()

    else:
        f = open(ckpt_path, 'r')
        ckpt = f.read().split(',')
        dim_ckpt = ckpt[0]
        img_ckpt = ckpt[1]
        map_ckpt = ckpt[2]
        f.close()

    for num in ["26", "30", "34"]:
        print('stage', num)
        if num == dim_ckpt:
            for indx, folder_sam in enumerate(folder_sam_list):
                if not (os.path.isfile((os.path.join(path_dir_sam, folder_sam)))) and (folder_sam == img_ckpt):
                    print('folder sam', folder_sam)
                    for indx2, folder_Cnext in enumerate(folder_cnext_list):
                        print('i 2', indx2)
                        if not (os.path.isfile((os.path.join(path_dir_CneXt, folder_sam)))) and folder_Cnext == folder_sam:
                            print('folder convnext', folder_Cnext)
                            for files_sam in sorted(os.listdir(os.path.join(path_dir_sam, folder_sam))):
                                print('files sam : {}'.format(files_sam))
                                if "stage{}".format(num) == files_sam.split('_')[0]:
                                    print('sam_file', files_sam)
                                    for files_Cnext in sorted(os.listdir(os.path.join(path_dir_CneXt, folder_Cnext))):
                                        if "stage{}".format(num) == files_Cnext.split('_')[0]:
                                            print('Cnext_file', files_Cnext)

                                            if num == "26":
                                                s = 'l'
                                            elif num == "30":
                                                s = 'm'
                                            elif num == "34":
                                                s = 's'

                                            name = folder_sam

                                            if not os.path.isdir(os.path.join(path_save, name)):
                                                os.mkdir(os.path.join(path_save, name))

                                            if os.path.isfile(os.path.join(path_dir_sam, name + '.png')) and \
                                                    os.path.isfile(os.path.join(path_dir_CneXt, name + '.png')):

                                                if not os.path.isfile(
                                                        os.path.join(path_save, name, name + '_convnextsam' + '.png')):
                                                    shutil.copy(os.path.join(path_dir_sam, name + '.png'),
                                                                os.path.join(path_save, name,
                                                                             name + '_convnextsam' + '.png'))

                                                if not os.path.isfile(
                                                        os.path.join(path_save, name, name + '_convnext' + '.png')):
                                                    shutil.copy(os.path.join(path_dir_CneXt, name + '.png'),
                                                                os.path.join(path_save, name, name + '_convnext' + '.png'))

                                            if os.path.isfile(
                                                    os.path.join(path_dir_sam, name + '.jpg')) and os.path.isfile(
                                                    os.path.join(path_dir_CneXt, name + '.jpg')):

                                                if not os.path.isfile(
                                                        os.path.join(path_save, name,
                                                                     name + '_convnextsam' + '.jpg')):
                                                    shutil.copy(os.path.join(path_dir_sam, name + '.jpg'),
                                                                os.path.join(path_save, name,
                                                                             name + '_convnextsam' + '.jpg'))

                                                if not os.path.isfile(
                                                        os.path.join(path_save, name, name + '_convnext' + '.jpg')):
                                                    shutil.copy(os.path.join(path_dir_CneXt, name + '.jpg'),
                                                                os.path.join(path_save, name,
                                                                             name + '_convnext' + '.jpg'))



                                            for i in ('s', 'm', 'l'):
                                                if not os.path.isdir(os.path.join(path_save, name, 'dif_{}'.format(i))):
                                                    os.mkdir(os.path.join(path_save, name, 'dif_{}'.format(i)))
                                                if not os.path.isdir(
                                                        os.path.join(path_save, name, 'dif_{}/mean'.format(i))):
                                                    os.mkdir(os.path.join(path_save, name, 'dif_{}/mean'.format(i)))

                                            sam = np.load(os.path.join(path_dir_sam, folder_sam, files_sam))
                                            Cnext = np.load(os.path.join(path_dir_CneXt, folder_Cnext, files_Cnext))

                                            dif = Cnext

                                            for i, map in enumerate(dif):
                                                if i >= int(map_ckpt):
                                                    map_ckpt = str(i)
                                                    f = open(ckpt_path, 'w')
                                                    f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
                                                    f.close()

                                                    '''if not os.path.isfile(os.path.join(path_save, name,
                                                                                       'dif_{}/map_{}.png'.format(s,
                                                                                                                  i))):
                                                        # plt.imsave('dif_s/map_{}.png'.format(i),map)
                                                        fig, ax = plt.subplots()
                                                        im = ax.imshow(map, cmap='jet')
                                                        plt.colorbar(im, ax=ax)
                                                        plt.savefig(os.path.join(path_save, name,
                                                                                 'dif_{}/map_{}.png'.format(s, i)),
                                                                    bbox_inches='tight', pad_inches=0.1)
                                                        plt.close()'''

                                                    if not os.path.isfile(os.path.join(path_save, name,'dif_{}/mean/mean.png'.format(s))):
                                                        mean = np.mean(dif, axis=0)
                                                        # plt.imsave('dif_s/mean/mean.png',mean)
                                                        fig, ax = plt.subplots()
                                                        im = ax.imshow(mean, cmap='jet')
                                                        plt.colorbar(im, ax=ax)
                                                        plt.savefig(os.path.join(path_save, name,
                                                                                 'dif_{}/mean/mean.png'.format(s)),
                                                                    bbox_inches='tight', pad_inches=0.1)
                                                        plt.close()

                                            map_ckpt = "0"
                                            f = open(ckpt_path, 'w')
                                            f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
                                            f.close()

                    if indx == len(folder_sam_list) - 1:
                        img_ckpt = folder_sam_list[0]
                    else:
                        img_ckpt = folder_sam_list[indx + 1]

                    f = open(ckpt_path, 'w')
                    f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
                    f.close()

            img_ckpt = folder_sam_list[0]
            dim_ckpt = "{}".format(int(dim_ckpt) + 4)
            f = open(ckpt_path, 'w')
            f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
            f.close()

    dim_ckpt = str(26)
    f = open(ckpt_path, 'w')
    f.write(sep.join([dim_ckpt, img_ckpt, map_ckpt]))
    f.close()

    os.remove(ckpt_path)
    return