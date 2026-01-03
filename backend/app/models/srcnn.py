import torch
import torch.nn as nn

from app.models.base import SuperResolutionModel
from app.utils.image_utils import preprocess, postprocess

class SRCNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x 
    
class SRCNN(SuperResolutionModel):
    def __init__(self):
        self.model = SRCNNNet()
        self.model.eval()

    def run(self, image):
        tensor = preprocess(image)
        with torch.no_grad():
            out = self.model(tensor)
        return postprocess(out)