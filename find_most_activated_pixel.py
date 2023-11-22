import torch
from utils.our_utils.drawing_heatmaps import draw_maxpoint

# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------     IMPORTANTE    ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
#
# i pesi che vuoi usare devono essere nel main branch della tua repo locale ( NON VA AGGIUNTO A GIT PERCHE' PESA TROPPO)

# model load
model = torch.hub.load('', 'custom', 'INSERT HERE THE MODEL WEIGHTS', source='local')

img_path = 'INSERT HERE THE PATH TO YOUR IMAGES'

out = model(img_path)  # out è un oggetto Detections dichiarato in common.py riga 1950

# si possono fare un sacco di cose simpatiche con questo oggetto...

# out.save(save_dir='runs/detect/exp') ti salva le immagini con la bbox sopra in save_dir (quello è il default)
# out.crop(save_dir='runs/detect/exp') ti salva i crop delle sole bbox
# guarda dove lo dichiara per altri metodi che non ho esplorato...
# out.pred da fuori le coordinate
# out.heat_maps da fuori le heatmap mediate delle teste

draw_maxpoint(out, thresh=True, medie=True)







