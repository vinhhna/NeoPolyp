import os
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = smp.UnetPlusPlus(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=3     
)
model.to(device)

checkpoint = torch.load('model_best_checkpoint.pth', weights_only=True)
model.load_state_dict(checkpoint['model'])

model_path = Path('test_prediction')
model_path.mkdir(parents=True, exist_ok=True)

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

color_dict= {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}
def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))
    for k in color_dict.keys():
        output[mask==k] = color_dict[k]
    return np.uint8(output)    

trainsize = 256
model.eval()
for i in os.listdir("test_sample"):
    image_path = os.path.join("test_sample", i)
    ori_img = cv2.imread(image_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(ori_img, (trainsize, trainsize))
    transformed = val_transform(image=img)
    input_img = transformed["image"]
    input_img = input_img.unsqueeze(0).to(device)
    with torch.inference_mode():
        output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
    mask = cv2.resize(output_mask, (ori_img.shape[1], ori_img.shape[0]))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask, color_dict)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_prediction/{}".format(i), mask_rgb) 