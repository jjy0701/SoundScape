import numpy as np
import pandas as pd
import pyroomacoustics as pra
import os
import random
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Point, Polygon
import functools

# 터미널에 로그가 즉시 출력되도록 설정
print = functools.partial(print, flush=True)

def generate_infinite_polygon():
    """3~8각형의 무작위 다각형 좌표 생성"""
    num_vertices = random.randint(3, 8)
    angles = sorted([random.uniform(0, 2 * np.pi) for _ in range(num_vertices)])
    distances = [random.uniform(5, 15) for _ in range(num_vertices)]
    corners = [[dist * np.cos(angle), dist * np.sin(angle)] for angle, dist in zip(angles, distances)]
    return np.array(corners).T, round(random.uniform(3.0, 5.0), 1)

def simulate_infinite_room(room_id):
    """단일 방 시뮬레이션 및 데이터 추출 (에러 발생 시 빈 리스트 반환하여 스킵)"""
    try:
        FS, GRID_STEP, MIC_HEIGHT = 16000, 0.7, 1.2
        corners, height = generate_infinite_polygon()
        poly = Polygon(corners.T)
        rt60 = round(random.uniform(0.3, 1.2), 2)
        
        # 물리 엔진 설정
        material = pra.Material(0.15)
        room = pra.Room.from_corners(corners, fs=FS, materials=material, max_order=8)
        room.extrude(height)
        
        # 🔧 스피커 배치를 위한 안전 구역(Margin) 설정
        margin = 0.5
        poly_buffered = poly.buffer(-margin)
        
        # 방이 너무 좁아서 마진을 줬더니 사라지면 원래 다각형 사용
        spk_poly = poly if poly_buffered.is_empty else poly_buffered
        spk_bounds = spk_poly.bounds
        min_x, min_y, max_x, max_y = float(spk_bounds[0]), float(spk_bounds[1]), float(spk_bounds[2]), float(spk_bounds[3])
        
        # ✅ 강력한 스피커 배치 검증 (Shapely + pyroomacoustics 2중 체크)
        max_attempts = 100
        valid_spk = False
        spk = [0.0, 0.0, 0.0]
        
        for _ in range(max_attempts):
            spk = [float(random.uniform(min_x, max_x)), 
                   float(random.uniform(min_y, max_y)), 
                   float(random.uniform(1.0, height-1.0))]
            
            # 1차 검문 (수학적 다각형 내부인가?)
            if spk_poly.contains(Point(spk[0], spk[1])): 
                # 2차 검문 (물리 엔진 상 내부인가?)
                if room.is_inside(spk):
                    try:
                        room.add_source(spk, signal=np.random.randn(FS))
                        valid_spk = True
                        break # 성공 시 루프 탈출
                    except Exception:
                        pass # 알 수 없는 충돌 시 무시하고 다음 좌표 시도
                        
        if not valid_spk:
            return [] # 100번 시도해도 안 들어가면 쿨하게 이 방은 포기함
        
        # 🔧 마이크 그리드 생성 (마이크도 2중 체크 도입)
        room_bounds = poly.bounds
        x_pts = np.arange(float(room_bounds[0]), float(room_bounds[2]), GRID_STEP)
        y_pts = np.arange(float(room_bounds[1]), float(room_bounds[3]), GRID_STEP)
        xx, yy = np.meshgrid(x_pts, y_pts)
        
        valid_mics = []
        for mx, my in zip(xx.flatten(), yy.flatten()):
            mx, my = float(mx), float(my)
            # 마이크 역시 Shapely와 엔진 양쪽에서 모두 OK 한 곳만 살림
            if poly.contains(Point(mx, my)) and room.is_inside([mx, my, MIC_HEIGHT]):
                valid_mics.append([mx, my, float(MIC_HEIGHT)])
                
        if not valid_mics: 
            return []
        
        room.add_microphone_array(np.array(valid_mics).T)
        
        # 연산 수행 (여기서 터져도 방을 스킵하도록 예외 처리)
        try:
            room.compute_rir()
        except (ValueError, RuntimeError):
            return []

        # 데이터 추출
        res = []
        n_vertices = int(len(corners[0]))
        height_val, rt60_val = float(height), float(rt60)
        
        for i in range(len(valid_mics)):
            # 가끔 반사음 배열이 비어있는 극단적 케이스 방어
            if len(room.rir[i][0]) == 0:
                continue
            energy = np.sum(np.square(room.rir[i][0]))
            db = float(10 * np.log10(energy + 1e-10))
            res.append([f"ROOM_{room_id:07d}", n_vertices, height_val, rt60_val, spk[0], spk[1], valid_mics[i][0], valid_mics[i][1], db])
            
        return res
    except Exception as e: # 🔧 Exception 뒤에 as e 추가
        # 🔧 어떤 에러인지 화면에 빨간불 켜주기
        print(f"❌ 방 {room_id} 내부 에러 발생: {repr(e)}") 
        return []

def run_infinite_gen(target_rows=1000000):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

    OUT_FILE = os.path.join(DATA_DIR, 'soundscape_infinite_1m.csv')
    print(f"🚀 [13900K Full Power] 데이터 생성 시작 (목표: {target_rows:,}행)")
    
    cols = ['room_id', 'n_vertices', 'height', 'rt60', 'spk_x', 'spk_y', 'mic_x', 'mic_y', 'dB']
    if not os.path.exists(OUT_FILE):
        pd.DataFrame(columns=cols).to_csv(OUT_FILE, index=False)

    current_rows = 0
    room_idx = 1
    skipped_rooms = 0 # 꼬여서 포기한 방의 개수 기록
    max_workers = max(1, os.cpu_count() - 2)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while current_rows < target_rows:
            futures = [executor.submit(simulate_infinite_room, room_idx + j) for j in range(50)]
            batch_data = []
            
            for f in futures:
                res = f.result() # 에러가 나더라도 무조건 처리 완료될 때까지 대기
                if isinstance(res, list) and len(res) > 0:
                    batch_data.extend(res)
                else:
                    skipped_rooms += 1 # 기하학적으로 꼬인 방은 스킵 카운트
            
            if batch_data:
                pd.DataFrame(batch_data).to_csv(OUT_FILE, mode='a', header=False, index=False)
                current_rows += len(batch_data)
                room_idx += 50
                print(f"📊 적재 완료: {current_rows:,} / {target_rows:,} 행...")
            else:
                # 🔧 방이 다 실패해서 스킵될 때도 터미널에 알려주기
                print(f"⚠️ 방 {room_idx} ~ {room_idx+4} 전부 실패! (누적 스킵: {skipped_rooms}개)")
                room_idx += 50
    
    print(f"✅ 대장정 완료! 총 {current_rows:,}행 생성됨 (저장 경로: {OUT_FILE})")

if __name__ == "__main__":
   
    # 🔧 2단계: 위 테스트가 성공하면, 윗줄을 지우고 아랫줄의 주석(#)을 풀어서 진짜로 돌리세요!
    run_infinite_gen(1000000)