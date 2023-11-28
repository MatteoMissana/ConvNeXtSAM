import os

import torch
# from utils.general import non_max_suppression
import cv2 as cv
from matplotlib import  pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image
import requests

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------     IMPORTANTE    ----------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#
# i pesi che vuoi usare devono essere nel main branch della tua repo locale ( NON VA AGGIUNTO A GIT PERCHE' PESA TROPPO)

def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90}.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info['exif'] = exif.tobytes()
    return image


def draw_bbox(pred, img):
    for bbox in pred:
        obj_class = int(bbox[-1])
        conf = "{:.{}f}".format(float(bbox[-2]), 2)

        p1 = (int(bbox[0]),int(bbox[1]))
        p2 = (int(bbox[2]),int(bbox[3]))

        c = ((int(bbox[0])+int(bbox[2]))//2,(int(bbox[1])+int(bbox[3]))//2)

        img = cv.rectangle(img, p1, p2, (255, 0, 0), thickness=2)
        img[c[1]-10:c[1]+11,c[0]-10:c[0]+11,:] = [255,0,0]

        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 3
        text_size = cv.getTextSize(conf, font, font_scale, font_thickness)[0]
        text_position = (int(int(bbox[0]) + (int(bbox[2])-int(bbox[0]) - text_size[0]) / 2), int(int(bbox[1]) - 5))

        cv.putText(img, conf, text_position, font, font_scale, (255, 0, 0), font_thickness, cv.LINE_AA)

    return img



# qui sotto ti metto come salvare il modello per esportarlo ed usarlo dove vuoi
# torch.save(model, 'your_path/complete_model.pth')

# img_path = 'dataset/micro/images/test/image_000001_png.rf.581bf4b408683cfe08dca8b231c0fd29.jpg'
# img = cv.imread(img_path)
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
model = torch.hub.load('', 'custom', 'runs/train/Micro/ConvNeXTtSAM_pretrained_hyp_finetune_60epcs/weights/best.pt', source='local')

out = model('dataset/sample_002_2d_needle_manipulation 2.mp4')  # out è un oggetto Detections dichiarato in common.py righa 1950

out.save(save_frames=True)
# si possono fare un sacco di cose simpatiche con questo oggetto...

# out.save(save_dir='runs/detect/exp') ti salva le immagini con la bbox sopra in save_dir (quello è il default)
# out.crop(save_dir='runs/detect/exp') ti salva i crop delle sole bbox
# guarda dove lo dichiara per altri metodi che non ho esplorato...

# out.pred da fuori le coordinate
# out.heat_maps da fuori le heatmap mediate delle teste


pred = out.heat_maps

# pred == list(tensor(x_up_left, y_up_left, x_bot_right, y_bot_right, conf, class); len(pred) == num_imgs; tensor.size() == (n_preds,6)

#
# cords = out.max_per_box(thresh=True)
#
# cords2 = out.max_per_box(medie=True)
#
# for i,im_path in enumerate(out.ims):
#
#     name = im_path.split('\\')[-1]
#     im = Image.open(requests.get(im_path, stream=True).raw if str(im_path).startswith('http') else im_path)
#     im = np.asarray(exif_transpose(im))
#     im_copy = im.copy()
#     del im
#     f = 'runs/detect/SAM_only_s'
#     if not os.path.isdir(f):
#         os.mkdir(f)
#
#     if cords[i]:   # per tenere conto delle immagini senza bbox da in output un lista vuota
#         for box in cords[i]:
#             im_copy[box[0]-10:box[0]+11, box[1]-10:box[1]+11, :] = [0,0,255]
#         for box2 in cords2[i]:
#             im_copy[box2[0]-10:box2[0]+11, box2[1]-10:box2[1]+11, :] = [0, 255, 0]
#
#     if out.pred[i].shape[0]:
#         im_copy = draw_bbox(out.pred[i], im_copy)
#
#     Image.fromarray(im_copy).save(os.path.join(f,f'{name}'), quality=95, subsampling=0)
#
#
# #
#
#
#
#
