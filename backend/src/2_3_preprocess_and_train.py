import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------
# 1. 모델 아키텍처 (Output 6개로 확장!)
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim))
        self.activation = nn.SiLU()
    def forward(self, x): return self.activation(x + self.block(x))

class SoundScapeNetPro_MultiBand(nn.Module):
    def __init__(self, input_dim=12, output_dim=6):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.res_blocks = nn.Sequential(ResidualBlock(256), ResidualBlock(256), ResidualBlock(256))
        # 🔥 출력층을 1개에서 6개(주파수 대역별)로 확장
        self.output_layer = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, output_dim))
        
    def forward(self, x): 
        return self.output_layer(self.res_blocks(self.input_layer(x)))

# ---------------------------------------------------------
# 2. 데이터 로딩 및 전처리 파이프라인
# ---------------------------------------------------------
def prepare_data(data_dir='data_multiband_v4'):
    print("📥 Parquet 데이터 파일 로딩 중...")
    files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"✅ 총 {len(df)}개의 데이터 로드 완료!")

    # X (입력 특성 12개)
    feature_cols = ['room_w', 'room_l', 'spk_x', 'spk_y', 'mic_x', 'mic_y', 
                    'dist', 'mean_absorption', 'obs_x', 'obs_y', 'obs_w', 'obs_l']
    
    # Y (출력 타겟 6개 - 주파수별 dB)
    target_cols = ['db_125Hz', 'db_250Hz', 'db_500Hz', 'db_1000Hz', 'db_2000Hz', 'db_4000Hz']

    X = df[feature_cols].values
    y = df[target_cols].values

    print("📊 데이터 정규화(Scaling) 진행 중...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler() # Multi-output 스케일링도 지원합니다.
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Train / Test 분할 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # 모델 및 스케일러 저장용 폴더 생성
    os.makedirs('models_v4', exist_ok=True)
    joblib.dump(scaler_X, 'models_v4/scaler_X_multi.pkl')
    joblib.dump(scaler_y, 'models_v4/scaler_y_multi.pkl')
    print("💾 스케일러 저장 완료 (models_v4/)")

    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------
# 3. 모델 훈련 (Train)
# ---------------------------------------------------------
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 학습 시작 (Using {device})")

    X_train, X_test, y_train, y_test = prepare_data()

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    model = SoundScapeNetPro_MultiBand(input_dim=12, output_dim=6).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50 # 논문용이면 50~100 권장
    train_losses, test_losses = [], []

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        test_batch_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                test_batch_losses.append(criterion(pred, y_batch).item())
        
        test_loss = np.mean(test_batch_losses)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    print(f"✅ 학습 종료! (총 소요 시간: {time.time() - start_time:.1f}초)")
    torch.save(model.state_dict(), 'models_v4/soundscape_model_multiband.pth')
    print("💾 모델 저장 완료 (models_v4/soundscape_model_multiband.pth)")

    # 논문용 Loss 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss (Validation)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Multi-Band Model Training Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('models_v4/training_loss_curve.png', dpi=300)
    print("📈 Loss 그래프 저장 완료 (models_v4/training_loss_curve.png)")

if __name__ == "__main__":
    train_model()