import pyroomacoustics as pra
import numpy as np
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import os

# 1. 자재 및 흡음률 DB (공식 표준)
MATERIAL_DB = {
    "concrete_bare": 0.02, "glass_window": 0.04, "brickwork": 0.05,
    "plasterboard": 0.10, "wood_16mm": 0.15, "carpet_cotton": 0.50, "acoustic_tiles": 0.80
}
MATERIAL_NAMES = list(MATERIAL_DB.keys())

# --- 장애물 차폐 판별 알고리즘 (CCW) ---
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def is_blocked(sx, sy, mx, my, ox, oy, ow, ol):
    """스피커와 청취자 사이의 직선이 장애물에 막혔는지 검사"""
    spk, mic = (sx, sy), (mx, my)
    c1, c2 = (ox, oy), (ox+ow, oy)
    c3, c4 = (ox+ow, oy+ol), (ox, oy+ol)
    
    def intersect(A, B, C, D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    if (intersect(spk, mic, c1, c2) or intersect(spk, mic, c2, c3) or 
        intersect(spk, mic, c3, c4) or intersect(spk, mic, c4, c1)):
        return True
    return False

def is_inside_l_shape(x, y, w, l, cut_x, cut_y, room_type):
    if room_type == "shoebox": return True
    if x >= cut_x and y >= cut_y: return False
    return True

# --- 단일 공간 시뮬레이션 (코어 로직) ---
def simulate_single_room(seed_offset):
    np.random.seed(seed_offset)
    random.seed(seed_offset)
    
    room_w, room_l = random.uniform(8.0, 30.0), random.uniform(8.0, 30.0)
    room_type = random.choice(["shoebox", "l_shape"])
    mat_abs = MATERIAL_DB[random.choice(MATERIAL_NAMES)]
    
    # 방 구조 세팅
    if room_type == "shoebox":
        room_early = pra.ShoeBox([room_w, room_l, 3.0], fs=16000, materials=pra.Material(mat_abs), max_order=3)
        corners = np.array([[0, 0], [room_w, 0], [room_w, room_l], [0, room_l]]).T
        cut_x, cut_y = room_w, room_l
    else:
        room_early = pra.ShoeBox([room_w, room_l, 3.0], fs=16000, materials=pra.Material(mat_abs), max_order=3)
        cut_x = random.uniform(room_w*0.4, room_w*0.7)
        cut_y = random.uniform(room_l*0.4, room_l*0.7)
        corners = np.array([[0, 0], [room_w, 0], [room_w, cut_y], [cut_x, cut_y], [cut_x, room_l], [0, room_l]]).T

    # 좌표 세팅
    while True:
        spk_x, spk_y = random.uniform(0.5, room_w-0.5), random.uniform(0.5, room_l-0.5)
        if is_inside_l_shape(spk_x, spk_y, room_w, room_l, cut_x, cut_y, room_type): break
    while True:
        mic_x, mic_y = random.uniform(0.5, room_w-0.5), random.uniform(0.5, room_l-0.5)
        if is_inside_l_shape(mic_x, mic_y, room_w, room_l, cut_x, cut_y, room_type): break

    # 장애물 위치 세팅
    obs_w = random.uniform(1.0, 6.0)
    obs_l = random.uniform(1.0, 6.0)
    obs_x = random.uniform(0.0, room_w - obs_w)
    obs_y = random.uniform(0.0, room_l - obs_l)

    # 에너지 계산 (지향성 보정 + Ray Tracing)
    dir_vec = pra.directivities.DirectionVector(azimuth=0, colatitude=90, degrees=True)
    dir_obj = pra.directivities.CardioidFamily(orientation=dir_vec, p=0.5)

    room_early.add_source([spk_x, spk_y, 1.5], directivity=dir_obj)
    room_early.add_microphone([mic_x, mic_y, 1.5])
    room_early.compute_rir()
    energy_dir = np.sum(room_early.rir[0][0] ** 2)

    room_early_omni = pra.ShoeBox([room_w, room_l, 3.0], fs=16000, materials=pra.Material(mat_abs), max_order=3)
    room_early_omni.add_source([spk_x, spk_y, 1.5])
    room_early_omni.add_microphone([mic_x, mic_y, 1.5])
    room_early_omni.compute_rir()
    energy_omni = np.sum(room_early_omni.rir[0][0] ** 2)
    
    dir_factor = energy_dir / max(energy_omni, 1e-10)

    room_final = pra.Room.from_corners(corners, fs=16000, materials=pra.Material(mat_abs), 
                                       max_order=3, ray_tracing=True, air_absorption=True)
    room_final.extrude(3.0)
    room_final.add_source([spk_x, spk_y, 1.5]) 
    room_final.add_microphone([mic_x, mic_y, 1.5])
    room_final.compute_rir()
    
    final_energy = np.sum(room_final.rir[0][0] ** 2) * dir_factor
    base_db = 10 * np.log10(max(final_energy, 1e-10))

    # 🔥 [핵심] 주파수 대역별 장애물 차폐(회절) 감쇠 적용
    freqs = [125, 250, 500, 1000, 2000, 4000]
    db_levels = {}

    if is_blocked(spk_x, spk_y, mic_x, mic_y, obs_x, obs_y, obs_w, obs_l):
        thickness_factor = np.sqrt(obs_w * obs_l)
        for f in freqs:
            # 주파수가 높을수록 감쇠(패널티)가 심해지는 물리적 근사 공식
            freq_penalty = 5.0 + (thickness_factor * 1.5 * np.log10(f / 50.0))
            db_levels[f'db_{f}Hz'] = base_db - freq_penalty
    else:
        # 차폐되지 않았을 때는 모든 주파수 대역이 베이스 dB를 따름
        for f in freqs:
            db_levels[f'db_{f}Hz'] = base_db

    return {
        'room_w': room_w, 'room_l': room_l,
        'spk_x': spk_x, 'spk_y': spk_y,
        'mic_x': mic_x, 'mic_y': mic_y,
        'dist': np.sqrt((spk_x - mic_x)**2 + (spk_y - mic_y)**2),
        'mean_absorption': mat_abs,
        'obs_x': obs_x, 'obs_y': obs_y, 'obs_w': obs_w, 'obs_l': obs_l,
        # 6개 라벨값으로 확장
        'db_125Hz': db_levels['db_125Hz'],
        'db_250Hz': db_levels['db_250Hz'],
        'db_500Hz': db_levels['db_500Hz'],
        'db_1000Hz': db_levels['db_1000Hz'],
        'db_2000Hz': db_levels['db_2000Hz'],
        'db_4000Hz': db_levels['db_4000Hz']
    }

# --- 병렬 처리 파이프라인 ---
def generate_data_final_run(total_samples=1000000, batch_size=100000):
    num_cores = multiprocessing.cpu_count()
    num_batches = total_samples // batch_size
    
    print(f"🔥 [Multi-Band 딥 시뮬레이션] 데이터셋 구축 시작")
    print(f"목표 샘플: {total_samples}개 | 배치 크기: {batch_size}개 | 사용 코어: {num_cores}개")
    
    # 새로운 폴더에 저장
    output_dir = 'data_multiband_v4'
    os.makedirs(output_dir, exist_ok=True)
    
    for b in range(num_batches):
        print(f"\n📦 배치 {b+1}/{num_batches} 생성 중...")
        data_records = []
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(simulate_single_room, i + (b * batch_size)) for i in range(batch_size)]
            
            for future in tqdm(as_completed(futures), total=batch_size, desc=f"Batch {b+1}"):
                data_records.append(future.result())
        
        df = pd.DataFrame(data_records)
        filename = f"{output_dir}/multiband_data_part_{b+1}.parquet"
        df.to_parquet(filename)
        print(f"💾 저장 완료: {filename} ({len(df)}개 샘플)")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # 100만 개 데이터를 10만 개씩 10번에 걸쳐 저장합니다.
    generate_data_final_run(total_samples=1000000, batch_size=100000)