# ConvNeXtSAM
ConvNeXtSAM, a YOLO based approach for tool-tip detection and localization in microsurgery

Our two main models are "ConvNeXtSAM", that contains the spatial attention model (SAM), and "ConvNeXt", that doesn't contain it. They were trained on a hand labeled subset of Cholect50 and fine tuned on a microsurgery dataset. You can find the weights of our models here: https://polimi365-my.sharepoint.com/:f:/g/personal/10767238_polimi_it/Ej4ki1wHAZtPr48_lxQEdPoBna0UPR26Cc5civmf0qnWCQ?e=VXw9X5

to do an inference on external data:
```
detect.py --weights (insert here the weights of your model) --source (your image or dataset path)
```

You can also extract more informations from your detections by doing this:
```
model = torch.hub.load('', 'custom', 'INSERT HERE THE MODEL WEIGHTS', source='local')

img_path = 'INSERT HERE THE PATH TO YOUR IMAGES'

out = model(img_path)
```
In this case out is an object with various methods:
```
out.save(save_dir='runs/detect/exp') 
```
saves the images with respective bounding box in save_dir
```
# out.crop(save_dir='runs/detect/exp') 
```
only saves the crops of your images that are contained in the bounding boxes.
In order to get your predictions do:
```
preds = out.pred 
```
and to retrieve the mean heatmaps that are fed to the prediction heads:
```
out.heat_maps
```

If instead you want to have an idea of the most activated pixel based from these heatmaps:
```
draw_maxpoint(out, savepath='your path (default runs/maxpixel/exp)' thresh=True, medie=True)
```
with this function you will save your images with some dots that represent
the most activated pixel. Dots will be green, red and blue. Green one
is derived with a threshold method that derives the pixel from a determined 
heatmap based on the dimension of the bounding box. You can 
utilize this method by setting the 'thresh' flag True.
The blue one is instead derived from a heatmap obtained 
as a weighted mean of the 3 heatmaps, based
on bounding box size. To utilize this method set the
'medie' flag True. Red one is the center of the bounding
box.

To evaluate your model:
```
python val.py --weights (insert here the weights of your model) --data (insert the name of the yaml file codifying your dataset) 
```

in order to train your model:
```
train.py --epochs (number of epochs) --batch  --img (image resolution) --data (name of the yaml file codifying your dataset) --cfg (model yaml file in yolov5 format) --weights (eventual weights if the model is pretrained) --hyp (yaml file containing your hyperparameters in yolo format) 
```

For more complete information refer to Yolov5 repo, to which we based our models. You can find it here:
https://github.com/ultralytics/yolov5
