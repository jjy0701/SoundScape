import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import joblib
import os
import librosa
import platform
import pandas as pd
import io
import soundfile as sf

# ------------------------------------------------
# 1. 기본 설정 및 특징 정의
# ------------------------------------------------
# 한글 폰트 설정 (깨짐 방지)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# AI가 학습한 12개 특징 이름
FEATURE_NAMES = [
    'room_w', 'room_l', 'spk_x', 'spk_y', 'mic_x', 'mic_y', 
    'dist', 'mean_absorption', 'obs_x', 'obs_y', 'obs_w', 'obs_l'
]

# ------------------------------------------------
# 2. AI 모델 구조 정의 (SoundScape-Pro V3)
# ------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.activation = nn.SiLU()
    def forward(self, x): 
        return self.activation(x + self.block(x))

class SoundScapeNetPro(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.res_blocks = nn.Sequential(ResidualBlock(256), ResidualBlock(256), ResidualBlock(256))
        self.output_layer = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, x): 
        return self.output_layer(self.res_blocks(self.input_layer(x)))

# ------------------------------------------------
# 3. 리소스 로드 및 오디오 물리 엔진
# ------------------------------------------------
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m_path = os.path.join(base_dir, 'data', 'models', 'soundscape_model_pro_v3.pth')
    sx_path = os.path.join(base_dir, 'data', 'models', 'scaler_X_v3.pkl')
    sy_path = os.path.join(base_dir, 'data', 'models', 'scaler_y_v3.pkl')
    
    model = SoundScapeNetPro(input_dim=12)
    model.load_state_dict(torch.load(m_path, map_location='cpu', weights_only=True))
    model.eval()
    
    return model, joblib.load(sx_path), joblib.load(sy_path)

def apply_room_acoustics(audio, predicted_db, max_db_in_room):
    """예측된 dB를 바탕으로 오디오의 실제 물리적 진폭을 조절합니다."""
    audio_float = np.array(audio, dtype=np.float32)
    
    # 방 안에서 가장 큰 소리(스피커 앞)를 기준으로 상대적인 감쇠량 계산
    relative_db = predicted_db - max_db_in_room 
    gain = 10 ** (relative_db / 20.0)
    
    processed = audio_float * gain
    return processed.astype(np.float32)

# ------------------------------------------------
# 4. 메인 대시보드 UI
# ------------------------------------------------
def main():
    st.set_page_config(page_title="SoundScape V3 Dashboard", layout="wide")
    st.title("🎧 SoundScape V3: 장애물 차폐 실시간 시뮬레이터")
    st.markdown("100만 개의 정밀 물리 데이터(PRA+알고리즘)를 학습한 AI가 장애물 뒤의 회절 소음을 예측합니다.")

    try:
        model, scaler_X, scaler_y = load_assets()
    except Exception as e:
        st.error(f"모델 로드 실패: {e}\n'data/models/' 폴더에 v3 파일이 있는지 확인하세요.")
        return

    # --- [사이드바] 공간 및 객체 설정 ---
    with st.sidebar:
        st.header("🏢 공간 설정")
        rw = st.slider("방 가로 길이 (m)", 8.0, 20.0, 15.0)
        rl = st.slider("방 세로 길이 (m)", 8.0, 20.0, 10.0)
        abs_v = st.slider("벽면 흡음률", 0.02, 0.8, 0.2)

        st.markdown("---")
        st.header("📦 장애물(Obstacle) 설정")
        ob_w = st.slider("장애물 폭", 1.0, 6.0, 3.0)
        ob_l = st.slider("장애물 길이", 1.0, 6.0, 2.0)
        ob_x = st.slider("장애물 X 좌표", 0.0, rw - ob_w, rw/2 - ob_w/2)
        ob_y = st.slider("장애물 Y 좌표", 0.0, rl - ob_l, rl/2 - ob_l/2)

        st.markdown("---")
        st.header("🔊 스피커 위치")
        spk_x = st.slider("스피커 X", 0.5, rw-0.5, 3.0)
        spk_y = st.slider("스피커 Y", 0.5, rl-0.5, 3.0)

        st.markdown("---")
        st.header("👂 청취자 위치")
        list_x = st.slider("청취자 X", 0.0, rw, rw-2.0)
        list_y = st.slider("청취자 Y", 0.0, rl, rl-2.0)

    # --- [AI 추론] 공간 전체 히트맵 연산 ---
    res = 0.5
    X_grid, Y_grid = np.meshgrid(np.arange(0, rw, res), np.arange(0, rl, res))
    fx, fy = X_grid.flatten(), Y_grid.flatten()
    dist = np.sqrt((spk_x - fx)**2 + (spk_y - fy)**2)

    features = np.column_stack([
        np.full_like(fx, rw), np.full_like(fx, rl),
        np.full_like(fx, spk_x), np.full_like(fx, spk_y),
        fx, fy, dist, np.full_like(fx, abs_v),
        np.full_like(fx, ob_x), np.full_like(fx, ob_y),
        np.full_like(fx, ob_w), np.full_like(fx, ob_l)
    ])

    with torch.no_grad():
        features_df = pd.DataFrame(features, columns=FEATURE_NAMES)
        # .values를 사용하여 터미널 경고(Warning) 제거
        in_tensor = torch.tensor(scaler_X.transform(features_df.values), dtype=torch.float32)
        pred_scaled = model(in_tensor).numpy()
        db_map = scaler_y.inverse_transform(pred_scaled).reshape(X_grid.shape)
        
        # 방 안에서 가장 시끄러운 위치의 dB 값을 오디오 정규화 기준으로 사용
        max_db_in_room = np.max(db_map)

    # --- [시각화] 히트맵 그리기 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(db_map, ax=ax, cmap="magma", cbar_kws={'label': 'SPL (dB)'})
    
    # 장애물 박스
    rect = patches.Rectangle((ob_x/res, ob_y/res), ob_w/res, ob_l/res, 
                             linewidth=2, edgecolor='white', facecolor='gray', alpha=0.8)
    ax.add_patch(rect)
    
    # 스피커와 청취자 위치
    ax.plot(spk_x/res, spk_y/res, 'w*', markersize=15, label="Speaker")
    ax.plot(list_x/res, list_y/res, 'co', markersize=12, label="Listener")
    
    ax.invert_yaxis()
    ax.legend(loc='upper right')
    ax.set_title("V3 AI 음향 지도 (장애물 회절/차폐 반영)", fontsize=15)
    st.pyplot(fig)

    # --- [오디오] 실시간 청취 시뮬레이션 영역 ---
    st.markdown("---")
    st.header("🎙️ 실시간 청취 시뮬레이션")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        audio_file = st.file_uploader("목소리 파일(.wav) 업로드", type=["wav"])
    
    if audio_file:
        # 오디오 파일 로드
        y, sr = librosa.load(audio_file, sr=None)
        
        # 청취자 단일 위치에 대한 AI 추론
        d_val = np.sqrt((spk_x - list_x)**2 + (spk_y - list_y)**2)
        f_val = pd.DataFrame([[rw, rl, spk_x, spk_y, list_x, list_y, d_val, abs_v, ob_x, ob_y, ob_w, ob_l]], columns=FEATURE_NAMES)
        
        with torch.no_grad():
            cur_db = scaler_y.inverse_transform(model(torch.tensor(scaler_X.transform(f_val.values), dtype=torch.float32)).numpy())[0][0]
        
        with col2:
            st.write(f"📍 현재 청취자 위치의 예측 음압: **{cur_db:.2f} dB**")
            
            # [자동 합성 및 재생] 슬라이더 조작 시 즉시 실행
            with st.spinner("AI가 공간 음향을 실시간 합성하는 중..."):
                # 1. 볼륨 조절 적용
                out_audio = apply_room_acoustics(y, cur_db, max_db_in_room)
                
                # 2. 파형 그래프 그리기 (100배 압축하여 전체 흐름 파악)
                fig_check, ax_check = plt.subplots(figsize=(10, 2))
                ax_check.plot(out_audio[::100])
                ax_check.set_ylim([-1, 1]) # 기준선을 고정하여 진폭 변화를 직관적으로 보여줌
                st.pyplot(fig_check)
                
                # 3. Streamlit의 강제 볼륨 뻥튀기 방지 (메모리 WAV로 굽기)
                out_audio_locked = np.int16(np.clip(out_audio, -1.0, 1.0) * 32767)
                buffer = io.BytesIO()
                sf.write(buffer, out_audio_locked, sr, format='WAV', subtype='PCM_16')
                buffer.seek(0)
                
                # 4. 즉시 자동 재생 (autoplay=True)
                st.audio(buffer, format='audio/wav', autoplay=True)
                
                st.success("✨ 마우스로 슬라이더를 옮기면 소리가 즉시 자동 재생됩니다!")

if __name__ == "__main__":
    main()