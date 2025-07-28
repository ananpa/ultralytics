import ultralytics
from ultralytics import YOLO
import torch.nn as nn
import torch

#ultralytics.checks()
#print('\n')

#print(ultralytics.__file__)
#print('\n')


model = YOLO('yolov8-fuse2.yaml')
print(model.model.info())
print()
print(model.model) 


dummy = torch.zeros(1, 3, 640, 640)
_ = model.model(dummy)             # ต้องรันผ่านได้ ไม่ error

'''
# ตรวจสอบค่า w_i หลัง train
from ultralytics import YOLO
from ultralytics.nn.modules.learnable_fusion import LearnableFusion
m = YOLO('runs/detect/exp/weights/best.pt')
for n, module in m.model.named_modules():
    if isinstance(module, LearnableFusion):
        print(n, torch.softmax(module.w, dim=0))

'''