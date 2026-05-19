import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm  # 진행률을 예쁘게 보여주는 라이브러리

# -----------------------------------------
# 1. Dataset: PySpark Parquet 데이터 로더
# -----------------------------------------
class RoomAcousticsDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"📂 데이터 로드 중: {os.path.basename(parquet_path)}")
        df = pd.read_parquet(parquet_path)
        
        # PySpark Vector의 'values' 배열만 추출하여 NumPy 행렬로 변환
        features = np.stack(df['scaled_features'].apply(lambda x: x['values']).values)
        
        # 인덱스 구조: 0:n_vertices, 1:height, 2:rt60, 3:dist_x, 4:dist_y, 5:distance, 6:dB
        # X (입력 데이터): 3번~6번 (소리 및 거리 정보 4개)
        self.X = torch.tensor(features[:, 3:], dtype=torch.float32)
        
        # Y (정답 데이터): 1번~2번 (천장 높이, 잔향 시간 2개)
        self.y = torch.tensor(features[:, 1:3], dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------------------
# 2. Model: 다중 출력 회귀 인공신경망 (MLP)
# -----------------------------------------
class SoundscapeNet(nn.Module):
    def __init__(self):
        super(SoundscapeNet, self).__init__()
        # GPU 성능에 맞춰 은닉층을 넉넉하고 깊게 구성
        self.net = nn.Sequential(
            nn.Linear(4, 256),
            nn.BatchNorm1d(256), # 배치 정규화 (학습 속도/안정성 대폭 향상)
            nn.ReLU(),
            nn.Dropout(0.2),     # 뇌세포 20%를 랜덤으로 꺼서 과적합(암기) 방지
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 2)     # 최종 출력 2개: height, rt60
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------
# 3. Training Loop: AI 학습 엔진
# -----------------------------------------
def train():
    # 🚀 학습 장치 설정 (자동으로 4080 Ti를 잡아냅니다)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 학습을 시작합니다! 사용 장치: {device}")
    if torch.cuda.is_available():
        print(f"💻 GPU: {torch.cuda.get_device_name(0)}")

    # 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'processed_soundscape')
    
    # 데이터 로더 (Batch Size 2048: 한 번에 2048개씩 묶어서 GPU에 투척)
    train_dataset = RoomAcousticsDataset(os.path.join(DATA_DIR, 'train.parquet'))
    test_dataset = RoomAcousticsDataset(os.path.join(DATA_DIR, 'test.parquet'))
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    model = SoundscapeNet().to(device)
    criterion = nn.MSELoss() # 평균 제곱 오차
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    best_val_loss = float('inf') # 최고 성능 저장을 위한 변수

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 훈련 단계 (tqdm 프로그래스 바 적용)
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
        
        # 검증 단계 (테스트 데이터로 평가)
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
        
        # 모델 저장 로직 (이전 에포크보다 에러율(Val Loss)이 낮아지면 저장!)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "best_model.pth"))

    print(f"✅ 학습 완료! 최고 성능(Val Loss): {best_val_loss:.4f}")
    print(f"💾 베스트 모델 저장 완료: {os.path.join(BASE_DIR, 'best_model.pth')}")

if __name__ == "__main__":
    train()