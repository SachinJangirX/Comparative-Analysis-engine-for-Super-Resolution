from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

from app.models.srcnn import SRCNN
from app.models.bicubic import Bicubic
from app.engine.comparator import ComparatorEngine


app = FastAPI(title="Super Resolution Comparison API")

models = {
    "bicubic": Bicubic(),
    "srcnn": SRCNN()
}

engine = ComparatorEngine(models)

@app.post("/compare")
async def compare_models(file: UploadFile = File(...)):
    data = await file.read()
    np_img = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = engine.run_all(img)

    output_paths = {}

    for name, out in results.items():
        path = f"{name}_api_out.jpg"
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        output_paths[name] = path

    return {"outputs": output_paths}

@app.get("/")
def health_check():
    return {"status": "API is running"}