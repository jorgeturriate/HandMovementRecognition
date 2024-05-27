from flask import Flask, request, jsonify
import cv2 as cv
from flask_cors import CORS
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from cnnModelHands import model, device
from frameProcessing import process_frames


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(Image.fromarray(image))


@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('frames[]')
    frames = [cv.imdecode(np.frombuffer(file.read(), np.uint8), cv.IMREAD_COLOR) for file in files]

    real_image, optical_image = process_frames(frames)
    
    predicted_class= False

    if real_image is False or optical_image is False:
        return jsonify({'class': False}), 200

    real_tensor = preprocess_image(real_image).unsqueeze(0).to(device)
    optical_tensor = preprocess_image(optical_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(real_tensor, optical_tensor)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = predicted.item()

    return jsonify({'class': predicted_class})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)