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
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # output: (32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(2)                   # output: (32, 13, 13)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * 13 * 13, 128),     # 32*13*13 = 5408
            nn.ReLU(),
            nn.Linear(128, 10)                # Final output layer
        )

    def forward(self, x):
        x = self.conv(x)
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