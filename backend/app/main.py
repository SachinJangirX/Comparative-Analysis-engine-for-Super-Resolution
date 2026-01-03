import cv2
from app.models.srcnn import SRCNN
from app.utils.image_utils import load_image
from app.models.bicubic import Bicubic
from app.engine.comparator import ComparatorEngine

img = load_image("test.jpg")

models = {
    "bicubic": Bicubic(),
    "srcnn": SRCNN(),
}

engine = ComparatorEngine(models)
results = engine.run_all(img)

for name, out in results.items():
    cv2.imwrite(f"{name}_out.jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

print("Comparison complete.")