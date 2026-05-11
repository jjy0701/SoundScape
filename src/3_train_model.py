import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

class AcousticDataset(Dataset):
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        features = np.stack(df['scaled_features'].apply(lambda x: x['values']).values)
        
        # 💡 [핵심 변경] 0~7번(총 8개)을 X로, 8번을 y로 분리
        self.X = torch.tensor(features[:, :8], dtype=torch.float32)
        self.y = torch.tensor(features[:, 8:9], dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SoundscapeMLP(nn.Module):
    def __init__(self):
        super(SoundscapeMLP, self).__init__()
        self.net = nn.Sequential(
            # 💡 [핵심 변경] 입력 차원을 7에서 8로 변경!
            nn.Linear(8, 64),      
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 [Level 1] 업그레이드 모델 학습 시작! 사용 장치: {device}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(BASE_DIR) == 'src':
        DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'processed_soundscape')
        MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'models')
    else:
        DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed_soundscape')
        MODEL_DIR = os.path.join(BASE_DIR, 'data', 'models')
    
    train_dataset = AcousticDataset(os.path.join(DATA_DIR, 'train.parquet'))
    test_dataset = AcousticDataset(os.path.join(DATA_DIR, 'test.parquet'))
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    model = SoundscapeMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50 
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
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(test_loader)
        print(f"📈 Epoch {epoch+1} 완료 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(MODEL_DIR, "best_forward_model.pth")
            torch.save(model.state_dict(), save_path)

    print(f"\n✅ 50 Epoch 학습 완료! 최종 최고 성능(Val Loss): {best_val_loss:.4f}")

if __name__ == "__main__":
    train()