from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

endpoint = "http://localhost:8502/v1/models/models:predict"
CLASS_NAMES = ['Early_Blight', 'Late_Blight', 'Healthy']

app = FastAPI()


@app.get("/ping")
async def ping():
    return 'working...'


def read_file_as_image(data) -> np.ndarray: #data is 'file.read()', which is the bytes of the file
    image = np.array(Image.open(BytesIO(data))) #basically i'm opening the image and converting it into numpy array
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read()) #file.read() reads the image and converts it into bytes
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8502)
