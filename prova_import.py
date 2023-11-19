import torch
# from utils.general import non_max_suppression
import cv2 as cv
from matplotlib import  pyplot as plt

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


model = torch.hub.load('MatteoMissana/ConvNeXtSAM', 'custom', 'ConvNext.pt', trust_repo=True)



# qui sotto ti metto come salvare il modello per esportarlo ed usarlo dove vuoi
# torch.save(model, 'your_path/complete_model.pth')


# print(model)
# summary(model, (3, 640, 640))

img = plt.imread('dataset/micro/images/test/image_000002_png.rf.a9ecda8477278f9747c4d47ffb8038bb.jpg')

out = model(img)  # out è un oggetto Detections dichiarato in common.py righa 1950

# si possono fare un sacco di cose simpatiche con questo oggetto...

# out.save(save_dir='runs/detect/exp') ti salva le immagini con la bbox sopra in save_dir (quello è il default)
# out.crop(save_dir='runs/detect/exp') ti salva i crop delle sole bbox
# guarda dove lo dichiara per altri metodi che non ho esplorato...

# out.save()
pred = out.pred
# pred == list(tensor(x_up_left, y_up_left, width, height, conf, class); len(pred) == num_imgs; tensor.size() == (n_preds,6)




print(pred)


# out_img = draw_bbox(pred, img)
#
# fig, ax = plt.subplots()
# ax.imshow(out_img)
#
# plt.savefig('runs/detect/exp/img.jpg')
# plt.close()


