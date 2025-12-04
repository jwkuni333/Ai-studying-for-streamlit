import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------------\
# 1. 모델 정의 (훈련 시 사용한 모델 구조와 동일해야 함)
# -----------------------------\
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------\
# 2. 모델 불러오기
# -----------------------------\
model = DNN()
# 저장된 가중치(state_dict)를 로드
model.load_state_dict(torch.load("mnist_dnn.pth", map_location="cpu"))
# 추론 모드로 설정
model.eval()
print("Model loaded!")


# -------------------------\
# 3. MNIST 테스트 세트 불러오기
# -------------------------\
test_dataset = datasets.MNIST(
    root=".", # 데이터 다운로드 경로를 현재 디렉토리(.)로 통일
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)


# -------------------------\
# 4. 이미지 1개 가져오기 (테스트 세트의 첫 번째 이미지)
# -------------------------\
img, label = test_dataset[0]
print("True Label:", label)

# 이미지 표시
# img는 (1, 28, 28) 형태이므로, matplotlib을 위해 .squeeze()로 (28, 28)로 변환
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"True Label: {label}")
plt.show() # 이미지를 화면에 표시

# DNN input 형태로 변환: (1, 28, 28) -> (1, 784)
img_flat = img.view(1, 28*28)

# -------------------------\
# 5. 추론
# -------------------------\
with torch.no_grad():
    # 모델에 입력하여 예측값(output) 계산
    output = model(img_flat)
    # 가장 높은 확률을 가진 클래스(인덱스)를 예측값으로 선택
    pred = torch.argmax(output, dim=1).item()

print("Predicted:", pred)
