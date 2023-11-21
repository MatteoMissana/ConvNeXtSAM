import torch
# from utils.general import non_max_suppression
import cv2 as cv
from matplotlib import  pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------     IMPORTANTE    ----------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#
# i pesi che vuoi usare devono essere nel main branch della tua repo locale ( NON VA AGGIUNTO A GIT PERCHE' PESA TROPPO)



def draw_bbox(preds, img):
    for pred in preds:
        for bbox in pred:
            obj_class = int(bbox[-1])
            conf = float(bbox[-2])

            p1 = (int(bbox[0]),int(bbox[1]))
            p2 = (int(bbox[2]),int(bbox[3]))

            img = cv.rectangle(img, p1, p2, (255, 0, 0), thickness=2)

    return img


model = torch.hub.load('', 'custom', 'ConvNext.pt',source='local')



# qui sotto ti metto come salvare il modello per esportarlo ed usarlo dove vuoi
# torch.save(model, 'your_path/complete_model.pth')

img_path = 'dataset/micro/images/test/image_000003_png.rf.c57270494b2f08b77196e026070d2097.jpg'

img = cv.imread(img_path)
img= cv.cvtColor(img, cv.COLOR_BGR2RGB)

out = model(img)  # out è un oggetto Detections dichiarato in common.py righa 1950

# si possono fare un sacco di cose simpatiche con questo oggetto...

# out.save(save_dir='runs/detect/exp') ti salva le immagini con la bbox sopra in save_dir (quello è il default)
# out.crop(save_dir='runs/detect/exp') ti salva i crop delle sole bbox
# guarda dove lo dichiara per altri metodi che non ho esplorato...

# out.pred da fuori le coordinate
# out.heat_maps da fuori le heatmap mediate delle teste


pred = out.heat_maps
# pred == list(tensor(x_up_left, y_up_left, width, height, conf, class); len(pred) == num_imgs; tensor.size() == (n_preds,6)


out.max_per_box()


# sta roba è il modo più comodo che ho trovato per farle a colori...
# cm = plt.get_cmap('jet')
#
# for i in range(3):
#     pred[0][i][pred[0][i] < 0] = 0
#
#     my_img = cv.resize(pred[0][i], (img.shape[1], img.shape[0]), interpolation=cv.INTER_LINEAR)
#     norm = plt.Normalize(vmin=np.min(my_img), vmax=np.max(my_img))
#     colored_image = cm(norm(my_img))
#     Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(f'runs/prova_reshp{i}.png')






