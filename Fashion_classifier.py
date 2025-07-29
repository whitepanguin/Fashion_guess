import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets


def csv_to_imagefolder(csv_path, output_dir):
    # CSV 파일 불러오기
    df = pd.read_csv(csv_path)

    # 이미지 데이터와 레이블 분리
    labels = df['label']
    images = df.drop('label', axis=1).values

    # 출력 폴더 생성
    for label in labels.unique():
        os.makedirs(os.path.join(output_dir, str(label)), exist_ok=True)

    # 이미지 저장
    for idx, (image, label) in tqdm(enumerate(zip(images, labels)), total=len(labels)):
        img_array = image.reshape(28, 28).astype(np.uint8)  # 28x28 흑백 이미지
        img = Image.fromarray(img_array, mode='L')          # PIL 이미지 생성 (L = grayscale)

        # 저장 경로
        img_path = os.path.join(output_dir, str(label), f"{idx}.png")
        img.save(img_path)

def prepare_imagefolder(train_csv, test_csv, train_output, test_output):
    # train 폴더가 없으면 변환 수행
    if not os.path.exists(train_output) or not os.listdir(train_output):
        print("🔄 변환 중: train CSV → 이미지 폴더")
        csv_to_imagefolder(train_csv, train_output)
    else:
        print("✅ train 이미지 폴더 이미 존재함, 변환 생략")

    if not os.path.exists(test_output) or not os.listdir(test_output):
        print("🔄 변환 중: test CSV → 이미지 폴더")
        csv_to_imagefolder(test_csv, test_output)
    else:
        print("✅ test 이미지 폴더 이미 존재함, 변환 생략")

train_csv  = '/content/drive/MyDrive/AI활용 소프트웨어 개발/12_딥러닝/data/fashion-mnist_train.csv'
test_csv  = '/content/drive/MyDrive/AI활용 소프트웨어 개발/12_딥러닝/data/fashion-mnist_test.csv'

train_output = '/content/drive/MyDrive/AI활용 소프트웨어 개발/12_딥러닝/data/train'
test_output = '/content/drive/MyDrive/AI활용 소프트웨어 개발/12_딥러닝/data/test'
prepare_imagefolder(train_csv, test_csv, train_output, test_output)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.RandomInvert(p=1.0),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.ImageFolder(root=train_output, transform=transform)
testset = datasets.ImageFolder(root=test_output, transform=transform)

len(trainset), len(testset)

print(trainset.__getitem__(10))

print(trainset.classes, testset.classes)

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random

# DataLoader로 섞기
train_loader = DataLoader(trainset, batch_size=256, shuffle=True)

# 첫 배치에서 모든 클래스 하나씩만 뽑기
images, labels = next(iter(train_loader))

# 클래스별 하나씩 고르기 위한 상태
shown = [False] * 10
selected_images = [None] * 10

# 각 클래스 하나씩 저장
for img, label in zip(images, labels):
    label_int = int(label)
    if not shown[label_int]:
        selected_images[label_int] = img
        shown[label_int] = True
    if all(shown):
        break

# 시각화
for i in range(10):
    plt.imshow(selected_images[i].squeeze(0), cmap='gray')
    plt.title(f"Label: {i}")
    plt.axis('off')
    plt.show()

class_map = {
    0: 'T-shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Boot'
}

loader = DataLoader(
    dataset=trainset,
    batch_size=64,
    shuffle= True,
)

imgs, labels = next(iter(loader))
print(imgs.shape, labels.shape)

fig, axes = plt.subplots(8, 8, figsize=(16, 16))

for ax, img, label in zip(axes.flatten(), imgs, labels):
    ax.imshow(img.reshape(28, 28), cmap='gray')
    ax.set_title(class_map[label.item()])
    ax.axis('off')

    # 장치 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

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
    
model = ConvNeuralNetwork().to(device)
print(model)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

def train_loop(train_loader, model, loss_fn, optimizer):
    sum_losses = 0
    sum_accs = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_losses = sum_losses + loss

        y_prob = nn.Softmax(1)(y_pred)
        y_pred_index = torch.argmax(y_prob, axis=1)
        acc = (y_batch == y_pred_index).float().sum() / len(y_batch) * 100
        sum_accs = sum_accs + acc

    avg_loss = sum_losses / len(train_loader)
    avg_acc = sum_accs / len(train_loader)
    return avg_loss, avg_acc

epochs = 50

for i in range(epochs):
    print(f"------------------------------------------------")
    avg_loss, avg_acc = train_loop(loader, model, loss, optimizer)
    print(f'Epoch {i:4d}/{epochs} Loss: {avg_loss:.6f} Accuracy: {avg_acc:.2f}%')
print("Done!")

# 테스트 데이터 로드
test_loader = DataLoader(
    dataset=testset,
    batch_size=32,
    shuffle=True
)

imgs, labels = next(iter(test_loader))
fig, axes = plt.subplots(4, 8, figsize=(16, 8))

for ax, img, label in zip(axes.flatten(), imgs, labels):
    ax.imshow(img.reshape(28, 28), cmap='gray')
    ax.set_title(class_map[label.item()])
    ax.axis('off')

def test(model, loader):
    model.eval()

    sum_accs = 0

    img_list = torch.Tensor().to(device)
    y_pred_list = torch.Tensor().to(device)
    y_true_list = torch.Tensor().to(device)

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        y_prob = nn.Softmax(1)(y_pred)
        y_pred_index = torch.argmax(y_prob, axis=1)
        y_pred_list = torch.cat((y_pred_list, y_pred_index), dim=0) # 예측한 리스트 행으로 붙음
        y_true_list = torch.cat((y_true_list, y_batch), dim=0)
        img_list = torch.cat((img_list, x_batch), dim=0)
        acc = (y_batch == y_pred_index).float().sum() / len(y_batch) * 100
        sum_accs += acc

    avg_acc = sum_accs / len(loader)
    return y_pred_list, y_true_list, img_list, avg_acc

y_pred_list, y_true_list, img_list, avg_acc = test(model, test_loader)
print(f'테스트 정확도는 {avg_acc:.2f}% 입니다.')

fig, axes = plt.subplots(4, 8, figsize=(16, 8))

img_list_cpu = img_list.cpu()
y_pred_list_cpu = y_pred_list.cpu()
y_true_list_cpu = y_true_list.cpu()

for ax, img, y_pred, y_true in zip(axes.flatten(), img_list_cpu, y_pred_list_cpu, y_true_list_cpu):
  ax.imshow(img.reshape(28, 28), cmap='gray')
  ax.set_title(f'pred: {class_map[y_pred.item()]}, true: {class_map[y_true.item()]}')
  ax.axis('off')

plt.show()

# 모델의 가중치와 매개변수만 저장
# 모델의 구조가 저장되지 않으므로 모델 클래스 정의가 없으면 복원할 수 없음
torch.save(model.state_dict(), 'model_weights.pth')
torch.save(model, 'model.pt')