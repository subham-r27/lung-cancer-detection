import os
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Disable oneDNN warnings (Windows)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI(title="Lung Cancer Detection")

# Paths
MODEL_PATH = "../dev/models/cnn_model.keras"
UPLOAD_FOLDER = "static/uploads"

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
model = load_model(MODEL_PATH)

# Class labels (must match training order)
CLASS_LABELS = ["Benign", "Malignant", "Normal"]

# Templates & static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Preprocess image
    img = image.load_img(file_path, target_size=(128, 128))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    result = model.predict(img)
    predicted_index = np.argmax(result)
    prediction = CLASS_LABELS[predicted_index]
    confidence = round(float(result[0][predicted_index]) * 100, 2)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "confidence": confidence,
            "image_path": file_path
        }
    )
