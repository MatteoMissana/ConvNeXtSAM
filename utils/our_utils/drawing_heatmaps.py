import os
import torch
import cv2 as cv
import numpy as np
from PIL import Image
import requests
from utils.dataloaders import exif_transpose
from utils.general import increment_path

def draw_bbox(pred, img):
    #given image and prediction list returns annotated image
    for bbox in pred:
        obj_class = int(bbox[-1])
        conf = float(bbox[-2])

        p1 = (int(bbox[0]),int(bbox[1]))
        p2 = (int(bbox[2]),int(bbox[3]))

        img = cv.rectangle(img, p1, p2, (255, 0, 0), thickness=2)
        centro=[(int(bbox[1])+int(bbox[3]))//2, (int(bbox[0])+int(bbox[2]))//2]
        img[centro[0] - 10:centro[0] + 11, centro[1] - 10:centro[1] + 11, :] = [255,0,0]
    return img

def draw_maxpoint(out, save_dir='runs/maxpixel', name='exp', thresh=False, medie=False, exist_ok=False):
    save_path = os.path.join(save_dir,name)
    f = increment_path(save_path, exist_ok=exist_ok, mkdir=True)

    if thresh:
        coord_t = out.max_per_box(thresh=True)

    if medie:
        coord_m = out.max_per_box(medie=True)

    preds = out.pred

    for i, im in enumerate(out.ims):
        if isinstance(im, str):
            name = im.split('\\')[-1]
            im = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im)
            im = np.asarray(exif_transpose(im))
        else:
            name = f"image_{i}.jpg"
        im_copy = im.copy()
        if preds[i].shape[0]:
            im_copy = draw_bbox(pred=preds[i], img=im_copy)
        del im

        # draw rect on pixel extracted with threshold
        if thresh:
            if coord_t[i]:  # if not empty list
                for box in coord_t[i]:
                    im_copy[box[0] - 10:box[0] + 11, box[1] - 10:box[1] + 11, :] = [0, 255, 0]  # green

        # draw rect on pixel extracted with weighted mean
        if medie:
            if coord_m[i]:  # if not empty list
                for box in coord_m[i]:
                    im_copy[box[0] - 10:box[0] + 11, box[1] - 10:box[1] + 11, :] = [0, 0, 255]  # blue

        # save image
        Image.fromarray(im_copy).save(os.path.join(f, f'{name}'), quality=95, subsampling=0)

    # print results
    print('results saved in {}'.format(f))
    print('green rectangle stands for threshold method')
    print('blue one for weighted mean')
