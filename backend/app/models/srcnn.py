import cv2 
import torch 
import torch.nn as nn

from app.models.base import SuperResolutionModel
from app.utils.image_utils import preprocess, postprocess


class SRCNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    
class SRCNN(SuperResolutionModel):
    def __init__(self, scale: int = 4):
        self.scale = scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SRCNNNet().to(self.device)
        self.model.eval()


    def run(self, image):
        h, w = image.shape[:2]

        upscaled = cv2.resize(
            image,
            (w*self.scale, h*self.scale),
            interpolation=cv2.INTER_CUBIC,
        )

        tensor = preprocess(upscaled).to(self.device)
        with torch.no_grad():
            refined = self.model(tensor)

        out = tensor + 0.1*refined

        return postprocess(out)