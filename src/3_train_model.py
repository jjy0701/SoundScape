import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------------------------
# 1. Dataset: Parquet 데이터를 PyTorch가 읽을 수 있게 변환
# -----------------------------------------
class AcousticDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"📂 데이터 로드 중: {os.path.basename(parquet_path)}")
        df = pd.read_parquet(parquet_path)
        
        # DataFrame의 리스트 형태 데이터를 NumPy 배열로 변환
        features = np.stack(df['scaled_features'].apply(lambda x: x['values']).values)
        
        # 💡 입력(X) 7개, 정답(y) 1개로 완벽 분리
        self.X = torch.tensor(features[:, :7], dtype=torch.float32)
        self.y = torch.tensor(features[:, 7:8], dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------------------
# 2. Model: 대리 모델 (Surrogate MLP) 구조
# -----------------------------------------
class SoundscapeMLP(nn.Module):
    def __init__(self):
        super(SoundscapeMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),      # 7개의 특징 입력
            nn.BatchNorm1d(64),    # 배치 정규화 (학습 안정화)
            nn.ReLU(),             # 활성화 함수
            nn.Dropout(0.1),       # 과적합 방지 (10% 비활성화)
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)       # 1개의 최종 예측값(0~1 스케일된 dB) 출력
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------
# 3. Training Loop: AI 학습 파이프라인
# -----------------------------------------
def train():
    # GPU(CUDA)가 있으면 무조건 사용! 없으면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 딥러닝 학습 시작! 사용 장치: {device}")
    if torch.cuda.is_available():
        print(f" 💡 그래픽카드 이름: {torch.cuda.get_device_name(0)}")

    # 경로 설정 (현재 파일 위치 기준)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 폴더 구조가 src 폴더 안에 있다면 한 단계 위로 올라가서 data 폴더를 찾음
    if os.path.basename(BASE_DIR) == 'src':
        DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'processed_soundscape')
        MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'models')
    else:
        DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed_soundscape')
        MODEL_DIR = os.path.join(BASE_DIR, 'data', 'models')
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 데이터 로더 준비 (배치 사이즈 2048)
    train_dataset = AcousticDataset(os.path.join(DATA_DIR, 'train.parquet'))
    test_dataset = AcousticDataset(os.path.join(DATA_DIR, 'test.parquet'))
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    # 모델, 손실 함수, 최적화 알고리즘 세팅
    model = SoundscapeMLP().to(device)
    criterion = nn.MSELoss()  # 평균 제곱 오차 (기획서 명시)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50  # 50번 반복 학습
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --- [학습 모드] ---
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()       # 기울기 초기화
            outputs = model(batch_X)    # 예측
            loss = criterion(outputs, batch_y) # 오차 계산
            loss.backward()             # 역전파
            optimizer.step()            # 가중치 업데이트
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- [검증 모드] ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad(): # 검증할 때는 학습을 하지 않음
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(test_loader)
        print(f"📈 Epoch {epoch+1} 완료 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 최고 기록 갱신 시 모델 가중치 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(MODEL_DIR, "best_forward_model.pth")
            torch.save(model.state_dict(), save_path)

    print(f"\n✅ 50 Epoch 학습 완료! 최고 성능(Val Loss): {best_val_loss:.4f}")
    print(f"💾 베스트 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    train()