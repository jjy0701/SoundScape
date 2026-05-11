import numpy as np
import pandas as pd
import pyroomacoustics as pra
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------
# 1. 시뮬레이션 설정값 (Configuration)
# -----------------------------------------
NUM_SAMPLES = 100000  # 10만 개 데이터 생성
ROOM_MIN, ROOM_MAX = 5.0, 20.0  
MARGIN = 0.5  
MAX_ORDER = 10  

# ---------------------------------------------------------
# 2. 단일 방 시뮬레이션 (흡음률 변수 추가!)
# ---------------------------------------------------------
def simulate_single_room(sample_idx):
    try:
        # 방 크기 무작위 생성
        room_width = np.random.uniform(ROOM_MIN, ROOM_MAX)
        room_length = np.random.uniform(ROOM_MIN, ROOM_MAX)
        
        # 💡 [Level 1 업그레이드] 흡음률(Absorption) 무작위 생성 (0.1 ~ 0.9)
        # 0.1: 소리가 엄청 울리는 방 (목욕탕, 콘크리트)
        # 0.9: 소리가 쫙쫙 흡수되는 방 (녹음실, 두꺼운 커튼)
        absorption = np.random.uniform(0.1, 0.9)
        
        # 방 객체 생성 시 흡음률 적용
        room = pra.ShoeBox(
            [room_width, room_length], 
            fs=16000, 
            materials=pra.Material(absorption), 
            max_order=MAX_ORDER
        )
        
        # 스피커 좌표 무작위 생성
        spk_x = np.random.uniform(MARGIN, room_width - MARGIN)
        spk_y = np.random.uniform(MARGIN, room_length - MARGIN)
        room.add_source([spk_x, spk_y])
        
        # 마이크 좌표 무작위 생성
        mic_x = np.random.uniform(MARGIN, room_width - MARGIN)
        mic_y = np.random.uniform(MARGIN, room_length - MARGIN)
        R = np.array([[mic_x], [mic_y]])
        room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
        
        # 물리 엔진 시뮬레이션 실행
        room.compute_rir()
        rir_signal = room.rir[0][0]
        energy = np.sum(rir_signal ** 2)
        dB = 10 * np.log10(energy + 1e-12)
        
        # 직선거리 계산
        distance = np.sqrt((spk_x - mic_x)**2 + (spk_y - mic_y)**2)
        
        # 💡 특징 8개 + 정답 1개 반환
        return {
            'room_width': round(room_width, 2),
            'room_length': round(room_length, 2),
            'speaker_x': round(spk_x, 2),
            'speaker_y': round(spk_y, 2),
            'mic_x': round(mic_x, 2),
            'mic_y': round(mic_y, 2),
            'distance': round(distance, 2),
            'absorption': round(absorption, 2), # 새로운 무기 추가!
            'sound_energy_dB': round(dB, 4)
        }
    except Exception as e:
        return None

# ---------------------------------------------------------
# 3. 메인 실행 블록
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 [레벨업 데이터 생성 시작] 흡음률 변수가 추가되었습니다. (목표: {NUM_SAMPLES:,}개)")
    
    results = []
    with ProcessPoolExecutor() as executor:
        futures = executor.map(simulate_single_room, range(NUM_SAMPLES))
        for res in tqdm(futures, total=NUM_SAMPLES, desc="시뮬레이션 진행 중"):
            if res is not None:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    os.makedirs('./data', exist_ok=True)
    save_path = './data/2d_acoustic_data_raw.csv'
    df.to_csv(save_path, index=False)
    
    print("\n🎉 데이터 생성 완료!")
    print(f" 📊 성공적으로 생성된 데이터: {len(df):,}개")
    print(f" 💾 저장 위치: {save_path}")