import numpy as np
import pandas as pd
import pyroomacoustics as pra
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor # 병렬 처리 핵심 라이브러리

# 1. 하나의 방을 시뮬레이션하는 단일 함수 (요리사 1명의 작업)
def simulate_single_room(room_id):
    GRID_STEP = 0.5
    MIC_HEIGHT = 1.2
    FS = 16000
    
    # 랜덤 변수 생성
    width = round(random.uniform(5.0, 20.0), 1)
    length = round(random.uniform(5.0, 15.0), 1)
    height = round(random.uniform(3.0, 6.0), 1)
    room_dim = [width, length, height]
    rt60 = round(random.uniform(0.4, 1.5), 2)
    
    spk_pos = [round(random.uniform(1.0, width-1.0), 2), 
               round(random.uniform(1.0, length-1.0), 2), 
               round(random.uniform(1.0, height-1.0), 2)]

    # 물리 엔진 실행
    try:
        e_absorption, max_order = pra.inverse_sabine(rt60=rt60, room_dim=room_dim)
        room = pra.ShoeBox(room_dim, fs=FS, materials=pra.Material(e_absorption), max_order=max_order)
        room.add_source(spk_pos, signal=np.random.randn(FS))
        
        x_coords = np.arange(GRID_STEP, width, GRID_STEP)
        y_coords = np.arange(GRID_STEP, length, GRID_STEP)
        xx, yy = np.meshgrid(x_coords, y_coords)
        mic_locs = np.vstack((xx.flatten(), yy.flatten(), np.full_like(xx, MIC_HEIGHT).flatten()))
        room.add_microphone_array(mic_locs)
        
        room.compute_rir()

        room_results = []
        for i in range(mic_locs.shape[1]):
            ir = room.rir[i][0]
            db_level = 10 * np.log10(np.sum(ir**2) + 1e-10)
            
            # 의도적 에러 삽입
            if random.random() < 0.02: db_level = np.nan
            elif random.random() < 0.01: db_level = 300.0
                
            room_results.append([f"ROOM_{room_id:05d}", width, length, height, rt60, 
                                 spk_pos[0], spk_pos[1], mic_locs[0, i], mic_locs[1, i], db_level])
        return room_results
    except:
        return []

# 2. 메인 실행부 (병렬 처리 지휘)
def run_parallel_simulation(total_rooms=10000):
    print(f"🔥 13900K 가동 시작: 총 {total_rooms}개 방 병렬 시뮬레이션...")
    start_time = time.time()
    
    # CPU 코어 개수 확인 (13900K는 24코어지만 효율을 위해 20개 정도 사용 권장)
    max_workers = os.cpu_count() - 4 
    
    all_data = []
    # 멀티프로세싱 엔진 가동
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 1부터 total_rooms까지의 작업을 코어들에게 분산 할당
        results = list(executor.map(simulate_single_room, range(1, total_rooms + 1)))
    
    # 결과 합치기
    for r in results:
        all_data.extend(r)
        
    # 데이터프레임 생성 및 저장
    columns = ['room_id', 'width', 'length', 'height', 'rt60', 'spk_x', 'spk_y', 'mic_x', 'mic_y', 'dB']
    df = pd.DataFrame(all_data, columns=columns)
    
    os.makedirs('./data', exist_ok=True)
    # 10,000개면 CSV보다 Parquet가 낫지만 일단 확인용으로 CSV 저장
    df.to_csv('./data/soundscape_massive_10k.csv', index=False)
    
    print(f"✅ 완료! 소요시간: {time.time() - start_time:.2f}초")
    print(f"📊 총 생성된 데이터 로우(Row) 수: {len(df):,}개")

if __name__ == "__main__":
    run_parallel_simulation(10000)