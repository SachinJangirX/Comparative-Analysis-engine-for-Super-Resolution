import cv2 
import numpy as np
import torch 

def load_image(path: str):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess(img: np.ndarray):
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    return tensor

def postprocess(tensor: torch.Tensor):
    img = tensor.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = (img*255.0).clip(0,255).astype(np.uint8)
    return img