import torch 
import torch.nn as nn
import numpy as np

from app.models.base import SuperResolutionModel
from app.utils.image_utils import preprocess, postprocess


# RRDBNet arhitecture (Simplified ESRGAN)
class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3,1,1)
        self.conv2 = nn.Conv2d(channels, channels, 3,1,1)
        self.conv3 = nn.Conv2d(channels, channels, 3,1,1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return x + out * 0.2
    

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=5):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3,1,1)
        self.blocks = nn.Sequential(*[ResidualDenseBlock(nf) for _ in range(nb)])
        self.conv_last = nn.Conv2d(nf, out_nc, 3,1,1)

    def forward(self, x):
        fea = self.conv_first(x)
        out = self.blocks(fea)
        out = self.conv_last(out)
        return out
    


# ESRGAN wrapper

class ESRGAN(SuperResolutionModel):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = RRDBNet().to(self.device)
        weights_path = "app/models/weights/RRDB_ESRGAN_x4.pth"

        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device),
            strict=False
        )

        self.model.eval()

    def run(self, image: np.ndarray):
        tensor = preprocess(image).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        return postprocess(out)