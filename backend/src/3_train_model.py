import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time

print("💾 [V3 Pro] 장애물 데이터(12 Features)를 불러오는 중...")
# 1. 방금 전처리한 V3 데이터 로드
base_dir = './data/processed_v3'
X_train = np.load(f'{base_dir}/X_train.npy').astype(np.float32)
y_train = np.load(f'{base_dir}/y_train.npy').astype(np.float32)
X_test = np.load(f'{base_dir}/X_test.npy').astype(np.float32)
y_test = np.load(f'{base_dir}/y_test.npy').astype(np.float32)

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# 2. 강력한 Residual 모델 구조 (입력 12차원)
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.activation = nn.SiLU()
    def forward(self, x): return self.activation(x + self.block(x))

class SoundScapeNetPro(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.res_blocks = nn.Sequential(ResidualBlock(256), ResidualBlock(256), ResidualBlock(256))
        self.output_layer = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, x): return self.output_layer(self.res_blocks(self.input_layer(x)))

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoundScapeNetPro(input_dim=12).to(device) # 🔥 입력 차원이 12개로 늘어났습니다!

criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 훈련 루프
epochs = 50
print(f"🚀 {device}에서 [SoundScape-Pro V3] 모델 학습 시작...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_X, val_y in test_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            val_loss += criterion(model(val_X), val_y).item()
            
    print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {train_loss/len(train_loader):.5f} | Val Loss: {val_loss/len(test_loader):.5f}")

elapsed_time = time.time() - start_time
print(f"⏱️ 총 학습 시간: {elapsed_time/60:.2f}분")

# 4. 모델 저장 (V3 전용 이름)
save_path = './data/models/soundscape_model_pro_v3.pth'
torch.save(model.state_dict(), save_path)
print(f"✨ V3 Pro 모델이 안전하게 저장되었습니다: {save_path}")