import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ------------------------------------------------
# 1. AI 아키텍처 정의
# ------------------------------------------------
class SoundscapeMLP(nn.Module):
    def __init__(self):
        super(SoundscapeMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
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

# ------------------------------------------------
# 2. 리소스 로드 (캐싱)
# ------------------------------------------------
@st.cache_resource
def load_models_and_scalers():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'data', 'models', 'best_forward_model.pth')
    scaler_x_path = os.path.join(base_dir, 'data', 'models', 'scaler_X.pkl')
    scaler_y_path = os.path.join(base_dir, 'data', 'models', 'scaler_y.pkl')

    model = SoundscapeMLP()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval() 

    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    return model, scaler_X, scaler_y

# ------------------------------------------------
# 3. Streamlit UI 및 렌더링 로직
# ------------------------------------------------
def main():
    st.set_page_config(page_title="무한 스피커 시뮬레이터", layout="wide")
    st.title("🎧 대리 모델 기반 무한 스피커 음향 대시보드")
    st.markdown("원하는 만큼 스피커를 추가하고 배치하여 복합적인 음압 분포를 실시간으로 설계하세요.")

    try:
        model, scaler_X, scaler_y = load_models_and_scalers()
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return

    # --- 💡 세션 상태(Session State) 초기화 ---
    # 사용자가 스피커를 추가/삭제해도 정보가 날아가지 않도록 기억 공간을 만듭니다.
    if 'speakers' not in st.session_state:
        st.session_state.speakers = [{'id': 1, 'x': 5.0, 'y': 5.0}] # 기본 스피커 1개
    if 'next_id' not in st.session_state:
        st.session_state.next_id = 2 # 다음 스피커에 부여할 고유 번호

    # 스피커 추가 함수
    def add_speaker():
        st.session_state.speakers.append({
            'id': st.session_state.next_id, 
            'x': room_width / 2, 
            'y': room_length / 2
        })
        st.session_state.next_id += 1

    # 스피커 삭제 함수
    def remove_speaker(spk_id):
        st.session_state.speakers = [s for s in st.session_state.speakers if s['id'] != spk_id]

    # --- 사이드바 UI ---
    with st.sidebar:
        st.header("⚙️ 공간 설정")
        room_width = st.slider("방 가로 길이 (m)", 5.0, 20.0, 15.0, 0.5)
        room_length = st.slider("방 세로 길이 (m)", 5.0, 20.0, 10.0, 0.5)
        
        st.markdown("---")
        st.header("🔊 스피커 관리")
        
        # 스피커 추가 버튼
        if st.button("➕ 새 스피커 추가", use_container_width=True):
            add_speaker()

        st.markdown("---")
        
        # 동적으로 생성된 스피커 목록 UI 출력
        for i, spk in enumerate(st.session_state.speakers):
            with st.expander(f"스피커 {spk['id']} 설정", expanded=True):
                # 방 크기가 줄어들면 스피커 위치도 방 안에 맞게 자동 조정되도록 최대값 제한
                max_x = max(0.5, float(room_width - 0.5))
                max_y = max(0.5, float(room_length - 0.5))
                
                # 고유 Key를 부여하여 슬라이더 충돌 방지
                spk['x'] = st.slider("X 좌표", 0.5, max_x, min(spk['x'], max_x), 0.1, key=f"x_{spk['id']}")
                spk['y'] = st.slider("Y 좌표", 0.5, max_y, min(spk['y'], max_y), 0.1, key=f"y_{spk['id']}")
                
                # 삭제 버튼
                if st.button("🗑️ 이 스피커 삭제", key=f"del_{spk['id']}"):
                    remove_speaker(spk['id'])
                    st.rerun() # 삭제 후 즉시 화면 새로고침

    # --- 메인 화면: 추론 및 시각화 ---
    resolution = 0.5 
    x_coords = np.arange(0, room_width, resolution)
    y_coords = np.arange(0, room_length, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

    flatten_x = X_grid.flatten()
    flatten_y = Y_grid.flatten()
    
    # 총 에너지를 누적할 빈 배열 생성 (스피커가 0개면 소리도 0)
    total_energy = np.zeros_like(flatten_x, dtype=np.float64)

    # 생성된 모든 스피커에 대해 각각 AI 추론 실행 후 에너지 합산
    with torch.no_grad():
        for spk in st.session_state.speakers:
            dist = np.sqrt((spk['x'] - flatten_x)**2 + (spk['y'] - flatten_y)**2)
            feat = np.column_stack((np.full_like(flatten_x, room_width), np.full_like(flatten_x, room_length),
                                    np.full_like(flatten_x, spk['x']), np.full_like(flatten_x, spk['y']),
                                    flatten_x, flatten_y, dist))
            
            # 1. 스케일링 -> 2. 모델 예측 -> 3. dB로 복구
            tensor_feat = torch.tensor(scaler_X.transform(feat), dtype=torch.float32)
            pred_scaled = model(tensor_feat).numpy()
            pred_db = scaler_y.inverse_transform(pred_scaled).flatten()
            
            # 4. 물리 법칙: dB를 순수 선형 에너지로 변환하여 누적 합산
            energy = 10 ** (pred_db / 10.0)
            total_energy += energy

    # 시각화 로직
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 스피커가 하나라도 있을 때만 히트맵 계산
    if len(st.session_state.speakers) > 0:
        # 합산된 총 에너지를 다시 사람이 듣는 dB 단위로 변환
        final_db = 10 * np.log10(total_energy + 1e-12)
        heatmap_data = final_db.reshape(X_grid.shape)
        
        sns.heatmap(heatmap_data, ax=ax, cmap="magma", 
                    xticklabels=np.round(x_coords, 1), 
                    yticklabels=np.round(y_coords, 1))
    else:
        # 스피커가 없으면 검은색 빈 화면 출력
        ax.set_facecolor('black')
        ax.set_xlim(0, len(x_coords))
        ax.set_ylim(0, len(y_coords))
        ax.text(len(x_coords)/2, len(y_coords)/2, "No Speakers Active", 
                color="white", ha='center', va='center', fontsize=20)
    
    ax.invert_yaxis() 
    ax.set_title(f"Dynamic SPL (dB) Distribution\nRoom: {room_width}m x {room_length}m | Active Speakers: {len(st.session_state.speakers)}", fontsize=16)
    ax.set_xlabel("Width (X) [m]")
    ax.set_ylabel("Length (Y) [m]")

    # 화면에 스피커 위치(마커) 찍기
    for spk in st.session_state.speakers:
        # 마커의 색상을 다르게 할 수도 있지만 가독성을 위해 통일하고 번호를 매김
        ax.plot(spk['x'] / resolution, spk['y'] / resolution, marker='*', color='cyan', markersize=15)
        ax.text(spk['x'] / resolution, (spk['y'] / resolution) + 0.5, f"S{spk['id']}", color='cyan', fontsize=12, ha='center')

    st.pyplot(fig)

if __name__ == "__main__":
    main()