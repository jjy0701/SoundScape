import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------------------------
# 1. Dataset: 3D 히트맵용 데이터 로더 (입력 9개 -> 출력 1개)
# -----------------------------------------
class HeatmapDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"📂 데이터 로드 중: {os.path.basename(parquet_path)}")
        df = pd.read_parquet(parquet_path)
        
        # 배열로 추출
        features = np.stack(df['scaled_features'].apply(lambda x: x['values']).values)
        
        # 💡 [핵심 변경] 
        # X (입력 데이터, 9개): n_vertices, height, spk_x, spk_y, mic_x, mic_y, dist_x, dist_y, distance
        self.X = torch.tensor(features[:, :9], dtype=torch.float32)
        
        # Y (정답 데이터, 1개): dB (소리 크기)
        # 형태를 [데이터 수, 1] 로 맞추기 위해 인덱싱을 9:10으로 설정
        self.y = torch.tensor(features[:, 9:10], dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------------------
# 2. Model: 히트맵 시뮬레이터 신경망 (Forward MLP)
# -----------------------------------------
class SoundscapeForwardNet(nn.Module):
    def __init__(self):
        super(SoundscapeForwardNet, self).__init__()
        # 💡 뇌세포 다이어트: 256->128->64 에서 64->32 로 대폭 축소! 층(Layer)도 하나 뺐습니다.
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),     # 모델이 작아졌으니 뇌세포 끄는 비율도 10%로 살짝 낮춰줍니다.
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)     # 군더더기 없이 바로 1개의 정답(dB) 도출!
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------
# 3. Training Loop: 4080 Ti 풀가동!
# -----------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 3D 히트맵 시뮬레이터 AI 학습 시작! 사용 장치: {device}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'processed_soundscape')
    
    # 데이터 세팅
    train_dataset = HeatmapDataset(os.path.join(DATA_DIR, 'train.parquet'))
    test_dataset = HeatmapDataset(os.path.join(DATA_DIR, 'test.parquet'))
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    model = SoundscapeForwardNet().to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", leave=False)
            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(test_loader)
        print(f"📈 Epoch {epoch+1} 완료 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
        
        # 정방향 모델(Forward Model)의 최고 가중치 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 이름이 헷갈리지 않게 'best_forward_model.pth'로 저장합니다.
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "best_forward_model.pth"))

    print(f"✅ 학습 완료! 최고 성능(Val Loss): {best_val_loss:.4f}")
    print(f"💾 베스트 모델 저장 완료: {os.path.join(BASE_DIR, 'best_forward_model.pth')}")

if __name__ == "__main__":
    train()