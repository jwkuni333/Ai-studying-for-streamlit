import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===============================
# 1. Hyperparameters
# ===============================
batch_size = 64
lr = 0.001
epochs = 5

# ===============================
# 2. MNIST Dataset & Loader
# ===============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # mean 0.5, std 0.5
])

train_data = datasets.MNIST(
    root=".",
    train=True,
    download=True,
    transform=transform
)

# test_data는 훈련 스크립트에서는 사용하지 않지만, Notebook의 원본 코드를 보존하기 위해 남겨둡니다.
# test_data = datasets.MNIST(
#     root=".",
#     train=False,
#     download=True,
#     transform=transform
# )

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=batch_size)


# ===============================
# 3. DNN Model
# ===============================
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # 28*28 (784) 입력 -> 256 -> ReLU -> 128 -> ReLU -> 10 출력
        self.net = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = DNN()

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ===============================
# 4. Training
# ===============================
print("Starting training...")
for epoch in range(epochs):
    total_loss = 0
    # enumerate(train_loader)를 사용하면 진행 상황을 더 잘 볼 수 있지만, 원본 코드를 유지했습니다.
    for images, labels in train_loader:
        # 이미지를 (배치 크기, 784)로 평탄화 (Flatten)
        images = images.view(-1, 28*28)

        # 역전파를 위한 기울기 초기화
        optimizer.zero_grad()
        
        # 순전파
        outputs = model(images)
        
        # 손실 계산
        loss = criterion(outputs, labels)

        # 역전파
        loss.backward()
        
        # 가중치 업데이트
        optimizer.step()

        total_loss += loss.item()

    # 평균 손실 대신 총 손실을 출력 (원본 코드와 동일)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

# ===============================
# 5. Save model
# ===============================
torch.save(model.state_dict(), "mnist_dnn.pth")
print("Model saved: mnist_dnn.pth")
