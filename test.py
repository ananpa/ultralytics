import ultralytics
from ultralytics import YOLO
import torch.nn as nn
from ultralytics.nn.modules.activation import PELU

ultralytics.checks()
print('\n')

print(ultralytics.__file__)
print('\n')

ACTIVATION_NAME = 'pelu'
BASE_MODEL = 'yolov8s'

model = YOLO(f'{BASE_MODEL}-{ACTIVATION_NAME}.yaml')
#print(model.model)

torch_model = model.model


# Recursive function to find all PELU activations
def print_pelu_parameters(module, prefix=''):
    for name, child in module.named_children():
        if isinstance(child, nn.Module):
            if isinstance(child, PELU):
                print(f"{prefix}.{name}: a = {child.a.item():.6f}, b = {child.b.item():.6f}")
            else:
                print_pelu_parameters(child, prefix + '.' + name if prefix else name)

# Print all PELU a and b values
print("🔍 PELU Parameters in YOLOv8 model:\n")
print_pelu_parameters(torch_model)