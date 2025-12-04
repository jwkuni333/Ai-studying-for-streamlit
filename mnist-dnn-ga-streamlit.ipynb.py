{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4bbaea98-14e2-4dc0-b129-e7fa7bb646f7",
   "metadata": {},
   "source": [
    "to train mnist dataset;"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "575d6001-8fdd-4ea4-b14e-757a54beb393",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100.0%\n",
      "100.0%\n",
      "100.0%\n",
      "100.0%\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [1/5] Loss: 313.8684\n",
      "Epoch [2/5] Loss: 143.4159\n",
      "Epoch [3/5] Loss: 105.3388\n",
      "Epoch [4/5] Loss: 84.1116\n",
      "Epoch [5/5] Loss: 74.0134\n",
      "Model saved: mnist_dnn.pth\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "from torchvision import datasets, transforms\n",
    "from torch.utils.data import DataLoader\n",
    "\n",
    "# ===============================\n",
    "# 1. Hyperparameters\n",
    "# ===============================\n",
    "batch_size = 64\n",
    "lr = 0.001\n",
    "epochs = 5\n",
    "\n",
    "# ===============================\n",
    "# 2. MNIST Dataset & Loader\n",
    "# ===============================\n",
    "transform = transforms.Compose([\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize((0.5,), (0.5,))  # mean 0.5, std 0.5\n",
    "])\n",
    "\n",
    "train_data = datasets.MNIST(\n",
    "    root=\".\",\n",
    "    train=True,\n",
    "    download=True,\n",
    "    transform=transform\n",
    ")\n",
    "\n",
    "test_data = datasets.MNIST(\n",
    "    root=\".\",\n",
    "    train=False,\n",
    "    download=True,\n",
    "    transform=transform\n",
    ")\n",
    "\n",
    "train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)\n",
    "test_loader = DataLoader(test_data, batch_size=batch_size)\n",
    "\n",
    "# ===============================\n",
    "# 3. DNN Model\n",
    "# ===============================\n",
    "class DNN(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(DNN, self).__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(28*28, 256),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(256, 128),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(128, 10)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        return self.net(x)\n",
    "\n",
    "model = DNN()\n",
    "\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n",
    "\n",
    "# ===============================\n",
    "# 4. Training\n",
    "# ===============================\n",
    "for epoch in range(epochs):\n",
    "    total_loss = 0\n",
    "    for images, labels in train_loader:\n",
    "        images = images.view(-1, 28*28)  # Flatten\n",
    "\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(images)\n",
    "        loss = criterion(outputs, labels)\n",
    "\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        total_loss += loss.item()\n",
    "\n",
    "    print(f\"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}\")\n",
    "\n",
    "# ===============================\n",
    "# 5. Save model\n",
    "# ===============================\n",
    "torch.save(model.state_dict(), \"mnist_dnn.pth\")\n",
    "print(\"Model saved: mnist_dnn.pth\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73b66152-d249-4b6c-af36-c635422dee51",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "341b7de2-8433-4c21-9314-fbbe324ed6e0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model loaded!\n",
      "True Label: 7\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted: 7\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "from torchvision import transforms\n",
    "from PIL import Image\n",
    "import matplotlib.pyplot as plt\n",
    "from torchvision import datasets, transforms\n",
    "\n",
    "\n",
    "# -----------------------------\n",
    "# 1. 모델 정의\n",
    "# -----------------------------\n",
    "class DNN(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(DNN, self).__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(28*28, 256),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(256, 128),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(128, 10)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        return self.net(x)\n",
    "\n",
    "# -----------------------------\n",
    "# 2. 모델 불러오기\n",
    "# -----------------------------\n",
    "model = DNN()\n",
    "model.load_state_dict(torch.load(\"mnist_dnn.pth\", map_location=\"cpu\"))\n",
    "model.eval()\n",
    "print(\"Model loaded!\")\n",
    "\n",
    "\n",
    "\n",
    "# -------------------------\n",
    "# 3. MNIST 테스트 세트 불러오기\n",
    "# -------------------------\n",
    "test_dataset = datasets.MNIST(\n",
    "    root=\"./data\",\n",
    "    train=False,\n",
    "    download=True,\n",
    "    transform=transforms.Compose([\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize((0.5,), (0.5,))\n",
    "    ])\n",
    ")\n",
    "\n",
    "\n",
    "\n",
    "# -------------------------\n",
    "# 4. 이미지 1개 가져오기\n",
    "# -------------------------\n",
    "img, label = test_dataset[0]\n",
    "print(\"True Label:\", label)\n",
    "\n",
    "\n",
    "# 이미지 표시\n",
    "plt.imshow(img.squeeze(), cmap=\"gray\")\n",
    "plt.title(f\"Label: {label}\")\n",
    "plt.show()\n",
    "\n",
    "\n",
    "\n",
    "# DNN input 형태로 변환 (1, 784)\n",
    "img_flat = img.view(1, 28*28)\n",
    "\n",
    "# -------------------------\n",
    "# 5. 추론\n",
    "# -------------------------\n",
    "with torch.no_grad():\n",
    "    output = model(img_flat)\n",
    "    pred = torch.argmax(output, dim=1).item()\n",
    "\n",
    "print(\"Predicted:\", pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "99823e59-87ce-45b5-b8f4-d94f93fde0b6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
