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


def draw_bbox(preds, img):
    for pred in preds:
        for bbox in pred:
            obj_class = int(bbox[-1])
            conf = float(bbox[-2])

            p1 = (int(bbox[0]),int(bbox[1]))
            p2 = (int(bbox[2]),int(bbox[3]))

            img = cv.rectangle(img, p1, p2, (255, 0, 0), thickness=2)

    return img


model = torch.hub.load('', 'custom', r"C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\model weights\ConvNeXtSAM_finetuning.pt", source='local')



# qui sotto ti metto come salvare il modello per esportarlo ed usarlo dove vuoi
# torch.save(model, 'your_path/complete_model.pth')

img_path = r"C:\Users\User\OneDrive - Politecnico di Milano\matteo onedrive\OneDrive - Politecnico di Milano\tesi_3ennale\dataset_MMI\images\test"

#img = cv.imread(img_path)
#img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

out = model(img_path)  # out è un oggetto Detections dichiarato in common.py righa 1950

# si possono fare un sacco di cose simpatiche con questo oggetto...

# out.save(save_dir='runs/detect/exp') ti salva le immagini con la bbox sopra in save_dir (quello è il default)
# out.crop(save_dir='runs/detect/exp') ti salva i crop delle sole bbox
# guarda dove lo dichiara per altri metodi che non ho esplorato...

# out.pred da fuori le coordinate
# out.heat_maps da fuori le heatmap mediate delle teste


pred = out.heat_maps
# pred == list(tensor(x_up_left, y_up_left, width, height, conf, class); len(pred) == num_imgs; tensor.size() == (n_preds,6)


cords = out.max_per_box(medie=True)

for i,im_path in enumerate(out.ims):

    name = im_path.split('\\')[-1]
    im = Image.open(requests.get(im_path, stream=True).raw if str(im_path).startswith('http') else im_path)
    im = np.asarray(exif_transpose(im))
    im_copy = im.copy()
    del im
    f = 'runs/detect/centri_medie_convnextSAM_MMItest'
    if not os.path.isdir(f):
        os.mkdir(f)

    if cords[i]:   # per tenere conto delle immagini senza bbox da in output un lista vuota
        for box in cords[i]:
            im_copy[box[0]-10:box[0]+11, box[1]-10:box[1]+11, 2] = 255
            im_copy[box[0] - 10:box[0] + 11, box[1] - 10:box[1] + 11, 1] = 0
            im_copy[box[0] - 10:box[0] + 11, box[1] - 10:box[1] + 11, 0] = 0

    Image.fromarray(im_copy).save(os.path.join(f,f'{name}'), quality=95, subsampling=0)







