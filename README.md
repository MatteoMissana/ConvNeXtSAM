# ConvNeXtSAM
ConvNeXtSAM, a YOLO based approach for tool-tip detection and localization in microsurgery

Our two main models are "ConvNeXtSAM", that contains the spatial attention model (SAM), and "ConvNeXt", that doesn't contain it. They were trained on a hand labeled subset of Cholect50 and fine tuned on a microsurgery dataset. You can find the weights of our models here: https://polimi365-my.sharepoint.com/:f:/g/personal/10767238_polimi_it/Ej4ki1wHAZtPr48_lxQEdPoBna0UPR26Cc5civmf0qnWCQ?e=VXw9X5

to do an inference on external data:
```
detect.py --weights (insert here the weights of your model) --source (your image or dataset path)
```

to evaluate your model:
```
python val.py --weights (insert here the weights of your model) --data (insert the name of the yaml file codifying your dataset) 
```

in order to train your model:
```
train.py --epochs (number of epochs) --batch  --img (image resolution) --data (name of the yaml file codifying your dataset) --cfg (model yaml file in yolov5 format) --weights (eventual weights if the model is pretrained) --hyp (yaml file containing your hyperparameters in yolo format) 
```

For more complete information refer to Yolov5 repo, to which we based our models. You can find it here:
https://github.com/ultralytics/yolov5
