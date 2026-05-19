from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import math
import os

app = FastAPI(title="SoundScape V4 Pro API")

# CORS 설정 (React 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 아키텍처
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim))
        self.activation = nn.SiLU()
    def forward(self, x): return self.activation(x + self.block(x))

class SoundScapeNetPro(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.res_blocks = nn.Sequential(ResidualBlock(256), ResidualBlock(256), ResidualBlock(256))
        self.output_layer = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, x): return self.output_layer(self.res_blocks(self.input_layer(x)))

# 로드 영역
model, scaler_X, scaler_y = None, None, None
FEATURE_NAMES = ['room_w', 'room_l', 'spk_x', 'spk_y', 'mic_x', 'mic_y', 'dist', 'mean_absorption', 'obs_x', 'obs_y', 'obs_w', 'obs_l']

@app.on_event("startup")
def load_assets():
    global model, scaler_X, scaler_y
    base_dir = os.path.dirname(os.path.abspath(__file__))
    m_path = os.path.join(base_dir, 'data', 'models', 'soundscape_model_pro_v3.pth')
    sx_path = os.path.join(base_dir, 'data', 'models', 'scaler_X_v3.pkl')
    sy_path = os.path.join(base_dir, 'data', 'models', 'scaler_y_v3.pkl')
    model = SoundScapeNetPro(input_dim=12)
    model.load_state_dict(torch.load(m_path, map_location='cpu', weights_only=True))
    model.eval()
    scaler_X, scaler_y = joblib.load(sx_path), joblib.load(sy_path)

# 데이터 규격
class Speaker(BaseModel): 
    x: float
    y: float
    angle: float = 0  # 스피커 방향 (도 단위)
class Obstacle(BaseModel): x: float; y: float; w: float; l: float
class AcousticRequest(BaseModel):
    room_w: float; room_l: float; list_x: float; list_y: float; abs_v: float
    obstacles: List[Obstacle]; speakers: List[Speaker]

# 핵심 로직: 단일 리스너 위치 예측
def get_prediction(req, spk, obs_list):
    dist = math.sqrt((spk.x - req.list_x)**2 + (spk.y - req.list_y)**2)
    
    # 1. 장애물 없는 기준값
    base_feat = pd.DataFrame([[req.room_w, req.room_l, spk.x, spk.y, req.list_x, req.list_y, dist, req.abs_v, 0,0,0,0]], columns=FEATURE_NAMES)
    base_db = scaler_y.inverse_transform(model(torch.tensor(scaler_X.transform(base_feat.values), dtype=torch.float32)).detach().numpy())[0][0]
    
    # 1.5 지향성(Directivity) 패턴 적용
    listener_angle = math.degrees(math.atan2(req.list_y - spk.y, req.list_x - spk.x))
    angle_diff = abs(listener_angle - spk.angle)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    # 정면(0도)=1.0, 측면(90도)=0.5, 후면(180도)=0.1
    directivity_factor = max(0.1, 1.0 - (angle_diff / 180) * 0.9)
    base_db += 20 * math.log10(directivity_factor)  # 지향성 감쇠 적용
    
    # 2. 누적 패널티 계산
    total_penalty = 0.0
    for ob in obs_list:
        # 각 장애물이 있을 때의 dB 계산
        obs_feat = pd.DataFrame([[req.room_w, req.room_l, spk.x, spk.y, req.list_x, req.list_y, dist, req.abs_v, ob.x, ob.y, ob.w, ob.l]], columns=FEATURE_NAMES)
        obs_db = scaler_y.inverse_transform(model(torch.tensor(scaler_X.transform(obs_feat.values), dtype=torch.float32)).detach().numpy())[0][0]
        total_penalty += max(0, base_db - obs_db)
    
    return base_db - total_penalty

@app.post("/predict")
def predict_endpoint(req: AcousticRequest):
    total_energy = 0.0
    for spk in req.speakers:
        db = get_prediction(req, spk, req.obstacles)
        total_energy += 10 ** (db / 10.0)
    return {"total_db": round(10 * math.log10(max(total_energy, 1e-10)), 2)}

@app.post("/predict_map")
def predict_map(req: AcousticRequest):
    # 히트맵용 격자 데이터 생성 (성능을 위해 0.5m 단위)
    res = 0.5
    grid = []
    for y in np.arange(0, req.room_l, res):
        row = []
        for x in np.arange(0, req.room_w, res):
            tmp_req = req.copy(update={"list_x": x, "list_y": y})
            total_energy = 0.0
            for spk in req.speakers:
                db = get_prediction(tmp_req, spk, req.obstacles)
                total_energy += 10 ** (db / 10.0)
            row.append(round(10 * math.log10(max(total_energy, 1e-10)), 1))
        grid.append(row)
    return {"map": grid}