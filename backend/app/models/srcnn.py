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
    
    def sharpen(img, strength=1.2):
        blur = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
        return cv2.addWeighted(img, strength, blur, -(strength-1), 0)
    
class SRCNN(SuperResolutionModel):
    def __init__(self, scale: int = 4):
        self.scale = scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SRCNNNet().to(self.device)
        self.model.eval()


    def run(self, image):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        h, w = y.shape
        y_up = cv2.resize(
            y,
            (w*self.scale, h*self.scale),
            interpolation=cv2.INTER_CUBIC,
        )

        y_tensor = torch.from_numpy(y_up/255.0).float().unsqueeze(0).unsqueeze(0)
        y_tensor = y_tensor.to(self.device)

        with torch.no_grad():
            y_sr = self.model(y_tensor)

        y_sr = y_sr.mean(dim=1, keepdim=False)
        y_sr = y_sr.squeeze(0).cpu().numpy()
        y_sr = np.clip(y_sr*255.0, 0, 255).astype(np.uint8)

        cr_up = cv2.resize(cr, (y_sr.shape[1], y_sr.shape[0]), interpolation=cv2.INTER_CUBIC)
        cb_up = cv2.resize(cb, (y_sr.shape[1], y_sr.shape[0]), interpolation=cv2.INTER_CUBIC)

        out = cv2.merge([y_sr, cr_up, cb_up])
        return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)
    
        out = sharpen(out, strength=1.2)

        return out