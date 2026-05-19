import torch
import torch.nn as nn
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# 1. 방금 학습한 모델 아키텍처 불러오기
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim))
        self.activation = nn.SiLU()
    def forward(self, x): return self.activation(x + self.block(x))

class SoundScapeNetPro_MultiBand(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(12, 256), nn.LayerNorm(256), nn.SiLU())
        self.res_blocks = nn.Sequential(ResidualBlock(256), ResidualBlock(256), ResidualBlock(256))
        self.output_layer = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 6))
    def forward(self, x): return self.output_layer(self.res_blocks(self.input_layer(x)))

def run_test():
    print("🤖 AI 모델 및 스케일러 로딩 중...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SoundScapeNetPro_MultiBand().to(device)
    model.load_state_dict(torch.load('models_v4/soundscape_model_multiband.pth', map_location=device))
    model.eval()

    scaler_X = joblib.load('models_v4/scaler_X_multi.pkl')
    scaler_y = joblib.load('models_v4/scaler_y_multi.pkl')

    # 2. 가상의 테스트 환경 세팅 (방 크기: 20x20m, 흡음률: 0.1)
    # 특성: [room_w, room_l, spk_x, spk_y, mic_x, mic_y, dist, mean_abs, obs_x, obs_y, obs_w, obs_l]
    
    # [Case A] 장애물이 시야를 가리지 않는 상황 (구석에 박혀있음)
    case_A = np.array([[20.0, 20.0, 5.0, 10.0, 15.0, 10.0, 10.0, 0.1, 1.0, 1.0, 1.0, 1.0]])
    
    # [Case B] 거대한 장애물이 스피커와 마이크 사이를 완벽히 가로막은 상황
    case_B = np.array([[20.0, 20.0, 5.0, 10.0, 15.0, 10.0, 10.0, 0.1, 9.0, 8.0, 2.0, 4.0]])

    # 3. 예측 수행
    with torch.no_grad():
        pred_A_scaled = model(torch.tensor(scaler_X.transform(case_A), dtype=torch.float32).to(device))
        pred_B_scaled = model(torch.tensor(scaler_X.transform(case_B), dtype=torch.float32).to(device))

    # 스케일링 복구 (원래 dB 수치로 변환)
    db_A = scaler_y.inverse_transform(pred_A_scaled.cpu().numpy())[0]
    db_B = scaler_y.inverse_transform(pred_B_scaled.cpu().numpy())[0]

    # 4. 결과 출력
    freqs = ['125Hz', '250Hz', '500Hz', '1kHz', '2kHz', '4kHz']
    
    print("\n🎤 [실전 추론 결과: 장애물 통과 시 주파수별 음압(dB) 변화]")
    print("-" * 55)
    print(f"{'주파수':<10} | {'Case A (장애물 없음)':<18} | {'Case B (막힘)':<15} | {'감쇠량(Drop)'}")
    print("-" * 55)
    
    drops = []
    for i, f in enumerate(freqs):
        drop = db_A[i] - db_B[i]
        drops.append(drop)
        print(f"{f:<10} | {db_A[i]:>12.2f} dB | {db_B[i]:>12.2f} dB | 📉 -{drop:.2f} dB")
    print("-" * 55)

    # 5. 논문용 막대 그래프 생성
    x = np.arange(len(freqs))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, db_A, width, label='Free Field (No Obstacle)', color='#3498db')
    plt.bar(x + width/2, db_B, width, label='Occluded (Behind Obstacle)', color='#e74c3c')
    plt.ylabel('Sound Pressure Level (dB)')
    plt.title('Frequency-Dependent Acoustic Attenuation by Obstacle')
    plt.xticks(x, freqs)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs('results_v4', exist_ok=True)
    plt.savefig('results_v4/frequency_attenuation_bar.png', dpi=300)
    print("📊 논문용 결과 그래프 저장 완료! (results_v4/frequency_attenuation_bar.png)")

if __name__ == "__main__":
    run_test()