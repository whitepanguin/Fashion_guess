# FastAPI backend
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F

class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.Conv2d(28, 28, kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(28, 56, kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.Conv2d(56, 56, kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(56 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # y는 10
        )

    def forward(self, x):
        x = self.classifier(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
model = ConvNeuralNetwork()
state_dict = torch.load('./model_weights.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

CLASSES = [
  'T-shirt',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Boot'
]

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.RandomInvert(1),
        transforms.Normalize((0.5), (0.5))
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    return transform(image).unsqueeze(0)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        print(f"Received file: {file.filename}, size: {len(image_bytes)} bytes")
        
        input_tensor = preprocess_image(image_bytes)
        print(f"input tensor shape: {input_tensor.shape}")
        
        with torch.no_grad():
            outputs = model(input_tensor)
            print(f"Model outputs: {outputs}")
            
            _, predicted = torch.max(outputs, 1)
            label = CLASSES[predicted.item()]
            print(f"Predicted label: {label}")
        
        return JSONResponse(content={"label": label})
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)