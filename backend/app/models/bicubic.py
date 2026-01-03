import cv2 
from app.models.base import SuperResolutionModel

class Bicubic(SuperResolutionModel):
    def __init__(self, scale=4):
        self.scale = scale

    def run(self, image):
        h, w = image.shape[:2]
        return cv2.resize(
            image,
            (w*self.scale, h*self.scale),
            interpolation=cv2.INTER_CUBIC
        )
    