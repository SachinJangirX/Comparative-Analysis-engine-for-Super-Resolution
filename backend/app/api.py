from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

from fastapi.staticfiles import StaticFiles
from app.models.srcnn import SRCNN
from app.models.bicubic import Bicubic
from app.engine.comparator import ComparatorEngine
from fastapi.middleware.cors import CORSMiddleware
# from app.models.esrgan import ESRGAN


app = FastAPI(title="Super Resolution Comparison API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory=".", html=True), name="static")

models = {
    "bicubic": Bicubic(),
    "srcnn": SRCNN(),
    # "esrgan": ESRGAN()
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