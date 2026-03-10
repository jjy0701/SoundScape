import numpy as np
import pandas as pd
import pyroomacoustics as pra
import os
import random
import time

def run_mass_simulation(num_rooms=10): # 일단 테스트로 10개의 공간만 만들어봅니다.
    print(f"=== SoundScape 대규모 음향 데이터 생성 시작 (총 {num_rooms}개 공간) ===")
    
    OUTPUT_DIR = './data'
    OUTPUT_FILE = 'soundscape_mass_data.csv'
    GRID_STEP = 0.5
    MIC_HEIGHT = 1.2
    FS = 16000
    
    all_results = []
    start_time = time.time()

    for room_id in range(1, num_rooms + 1):
        # ---------------------------------------------------------
        # 1. 무작위(Random) 공간 및 환경 설정
        # ---------------------------------------------------------
        # 방 크기 무작위 생성 (가로 5~20m, 세로 5~15m, 높이 3~6m 사이)
        width = round(random.uniform(5.0, 20.0), 1)
        length = round(random.uniform(5.0, 15.0), 1)
        height = round(random.uniform(3.0, 6.0), 1)
        room_dim = [width, length, height]
        
        # 잔향 시간(RT60) 무작위 생성 (0.4초 ~ 1.5초) -> 공간의 울림 정도가 매번 다름
        rt60 = round(random.uniform(0.4, 1.5), 2)
        
        # 스피커 위치 무작위 생성 (벽에 너무 붙지 않게 방 크기 안쪽으로 설정)
        spk_x = round(random.uniform(1.0, width - 1.0), 2)
        spk_y = round(random.uniform(1.0, length - 1.0), 2)
        spk_z = round(random.uniform(1.0, height - 1.0), 2)
        speaker_pos = [spk_x, spk_y, spk_z]

        print(f"[{room_id}/{num_rooms}] 시뮬레이션 중... 방 크기: {width}x{length}x{height}m, RT60: {rt60}s")

        # ---------------------------------------------------------
        # 2. 물리 엔진 객체 생성 및 실행
        # ---------------------------------------------------------
        e_absorption, max_order = pra.inverse_sabine(rt60=rt60, room_dim=room_dim)
        room = pra.ShoeBox(room_dim, fs=FS, materials=pra.Material(e_absorption), max_order=max_order)
        room.add_source(speaker_pos, signal=np.random.randn(FS))
        
        # 마이크 그리드 생성 (방 크기에 맞춰 유동적으로 생성됨)
        x_coords = np.arange(GRID_STEP, width, GRID_STEP)
        y_coords = np.arange(GRID_STEP, length, GRID_STEP)
        xx, yy = np.meshgrid(x_coords, y_coords)
        zz = np.full_like(xx, MIC_HEIGHT)
        
        mic_locs = np.vstack((xx.flatten(), yy.flatten(), zz.flatten()))
        room.add_microphone_array(mic_locs)
        
        room.compute_rir() # 연산 실행

        # ---------------------------------------------------------
        # 3. 데이터 추출 및 누적
        # ---------------------------------------------------------
        for i in range(mic_locs.shape[1]):
            ir = room.rir[i][0]
            sound_energy = np.sum(ir**2)
            db_level = 10 * np.log10(sound_energy + 1e-10)
            
            # 의도적인 에러(전처리용 결측치/이상치) 3% 확률로 삽입
            if random.random() < 0.02:
                db_level = np.nan
            elif random.random() < 0.01:
                db_level = 300.0
                
            all_results.append({
                'room_id': f"ROOM_{room_id:04d}",  # 어떤 방의 데이터인지 식별하는 고유 ID 추가!
                'room_width': width,
                'room_length': length,
                'room_height': height,
                'rt60': rt60,
                'speaker_x': spk_x,
                'speaker_y': spk_y,
                'speaker_z': spk_z,
                'mic_x': mic_locs[0, i],
                'mic_y': mic_locs[1, i],
                'mic_z': mic_locs[2, i],
                'sound_energy_dB': db_level
            })

    # ---------------------------------------------------------
    # 4. 전체 데이터 CSV 적재
    # ---------------------------------------------------------
    df = pd.DataFrame(all_results)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    elapsed_time = round(time.time() - start_time, 2)
    print(f"\n=== 생성 완료! ===")
    print(f"총 {num_rooms}개 방에서 무려 {len(df):,}건의 데이터가 추출되었습니다!")
    print(f"소요 시간: {elapsed_time}초")
    print(f"저장 경로: {save_path}")

if __name__ == "__main__":
    # 일단 10개만 돌려보고, 컴퓨터 성능에 따라 100개, 1000개로 늘려보세요!
    run_mass_simulation(num_rooms=10)