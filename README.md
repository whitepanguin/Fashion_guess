<img width="1915" height="1001" alt="12" src="https://github.com/user-attachments/assets/2076357b-56b5-45c8-ad2a-53cd4e672d60" />

<img width="1053" height="246" alt="21" src="https://github.com/user-attachments/assets/f4532fb0-e932-4832-aff8-88fc675250c1" />

1차 시도

            class ConvNeuralNetwork(nn.Module):
            def **init**(self):
            super(ConvNeuralNetwork, self).**init**()
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

정확도 75%너무 성능이 안나오고 무겁다

2차 시도

            class ConvNeuralNetwork(nn.Module):
            def **init**(self):
            super(ConvNeuralNetwork, self).**init**()
            self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3), # -> (64, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (64, 13, 13)
            
                        nn.Conv2d(64, 64, kernel_size=3),    # -> (64, 11, 11)
                        nn.ReLU(),
                        nn.MaxPool2d(2)                      # -> (64, 5, 5)
                    )
                    self.flatten = nn.Flatten()
                    self.fc_layers = nn.Sequential(
                        nn.Linear(64 * 5 * 5, 128),          # 64 filters * 5x5 feature maps
                        nn.ReLU(),
                        nn.Linear(128, 10)                   # Final classification layer
                    )
            
                def forward(self, x):
                    x = self.conv_layers(x)
                    x = self.flatten(x)
                    x = self.fc_layers(x)
                    return x

정확도 85% 가볍고 성능도 나오지만 아직 뭔가 부족해 보인다

-> https://airsbigdata.tistory.com/219 참고

3차 시도

            class ConvNeuralNetwork(nn.Module):
            def **init**(self):
            super(ConvNeuralNetwork, self).**init**()
            self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel*size=3), # output: (32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(2) # output: (32, 13, 13)
            )
            self.flatten = nn.Flatten()
            self.fc = nn.Sequential(
            nn.Linear(32 * 13 \_ 13, 128), # 32*13*13 = 5408
            nn.ReLU(),
            nn.Linear(128, 10) # Final output layer
            )
            
                def forward(self, x):
                    x = self.conv(x)
                    x = self.flatten(x)
                    x = self.fc(x)
                    return x

torch.Size([64, 1, 28, 28]) torch.Size([64])
cpu
ConvNeuralNetwork(
(conv): Sequential(
(0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
(1): ReLU()
(2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
(flatten): Flatten(start_dim=1, end_dim=-1)
(fc): Sequential(
(0): Linear(in_features=5408, out_features=128, bias=True)
(1): ReLU()
(2): Linear(in_features=128, out_features=10, bias=True)
)
)

---

## Epoch 0/20 Loss: 0.447115 Accuracy: 84.89%

## Epoch 1/20 Loss: 0.295797 Accuracy: 89.42%

## Epoch 2/20 Loss: 0.263250 Accuracy: 90.57%

## Epoch 3/20 Loss: 0.241937 Accuracy: 91.22%

## Epoch 4/20 Loss: 0.212615 Accuracy: 92.32%

## Epoch 5/20 Loss: 0.198813 Accuracy: 92.95%

## Epoch 6/20 Loss: 0.182557 Accuracy: 93.43%

## Epoch 7/20 Loss: 0.178742 Accuracy: 93.64%

## Epoch 8/20 Loss: 0.164814 Accuracy: 94.14%

## Epoch 9/20 Loss: 0.156708 Accuracy: 94.71%

## Epoch 10/20 Loss: 0.153109 Accuracy: 94.77%

## Epoch 11/20 Loss: 0.142522 Accuracy: 95.15%

## Epoch 12/20 Loss: 0.136206 Accuracy: 95.25%

## Epoch 13/20 Loss: 0.132448 Accuracy: 95.49%

## Epoch 14/20 Loss: 0.130007 Accuracy: 95.61%

## Epoch 15/20 Loss: 0.130977 Accuracy: 95.75%

## Epoch 16/20 Loss: 0.116767 Accuracy: 96.16%

## Epoch 17/20 Loss: 0.117509 Accuracy: 96.12%

## Epoch 18/20 Loss: 0.108685 Accuracy: 96.50%

Epoch 19/20 Loss: 0.115697 Accuracy: 96.28%
Done!

결과적으로 96% 효율을 내는 패션 예측 모델을 만들수 있었다.


