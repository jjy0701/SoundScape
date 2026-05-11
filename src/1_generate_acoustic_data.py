import numpy as np
import pandas as pd
import pyroomacoustics as pra
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------
# 1. 시뮬레이션 설정값 (Configuration)
# ---------------------------------------------------------
NUM_SAMPLES = 100000  # 우선 1천 개만 테스트 (최종적으로 50000으로 수정)
ROOM_MIN, ROOM_MAX = 5.0, 20.0  # 방 길이 최소/최대 (5m ~ 20m)
MARGIN = 0.5  # 스피커나 마이크가 벽에 너무 딱 붙지 않도록 0.5m 여백 주기
ABSORPTION = 0.2  # 벽면 흡음률 (일반적인 실내 환경 0.2)
MAX_ORDER = 10  # 거울상 기법(ISM) 반사 계산 횟수 (너무 높으면 느려짐)

# ---------------------------------------------------------
# 2. 단일 방 시뮬레이션 함수 (Core Logic)
# ---------------------------------------------------------
def simulate_single_room(sample_idx):
    try:
        # 1. 방 크기 무작위 생성 (가로, 세로)
        room_width = np.random.uniform(ROOM_MIN, ROOM_MAX)
        room_length = np.random.uniform(ROOM_MIN, ROOM_MAX)
        
        # 2. 2D 방(ShoeBox) 객체 생성
        room = pra.ShoeBox(
            [room_width, room_length], 
            fs=16000, 
            materials=pra.Material(ABSORPTION), 
            max_order=MAX_ORDER
        )
        
        # 3. 스피커 좌표 무작위 생성 (벽에서 MARGIN만큼 띄움)
        spk_x = np.random.uniform(MARGIN, room_width - MARGIN)
        spk_y = np.random.uniform(MARGIN, room_length - MARGIN)
        room.add_source([spk_x, spk_y])
        
        # 4. 마이크(수음점) 좌표 무작위 생성
        mic_x = np.random.uniform(MARGIN, room_width - MARGIN)
        mic_y = np.random.uniform(MARGIN, room_length - MARGIN)
        
        # pyroomacoustics 마이크 배열 추가 형태: shape (ndim, n_mics)
        R = np.array([[mic_x], [mic_y]])
        room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
        
        # 5. 거울상 기법(ISM) 연산 실행! (임펄스 응답 도출)
        room.compute_rir()
        
        # 6. 임펄스 응답(RIR) 파형에서 데시벨(dB) 추출
        # 공식: 10 * log10(파형 에너지의 총합)
        rir_signal = room.rir[0][0]  # 첫 번째 마이크, 첫 번째 소스
        energy = np.sum(rir_signal ** 2)
        dB = 10 * np.log10(energy + 1e-12) # log(0) 에러 방지용 아주 작은 숫자 더하기
        
        # 7. 기획서에 명시된 직선거리(distance) 파생 변수 계산
        distance = np.sqrt((spk_x - mic_x)**2 + (spk_y - mic_y)**2)
        
        # 8. 최종 추출된 피처(Feature) 반환
        return {
            'room_width': round(room_width, 2),
            'room_length': round(room_length, 2),
            'speaker_x': round(spk_x, 2),
            'speaker_y': round(spk_y, 2),
            'mic_x': round(mic_x, 2),
            'mic_y': round(mic_y, 2),
            'distance': round(distance, 2),
            'sound_energy_dB': round(dB, 4)
        }
    except Exception as e:
        # 가끔 기하학적 계산 에러가 발생할 수 있으므로 예외 처리
        return None

# ---------------------------------------------------------
# 3. 메인 실행 블록 (다중 프로세싱 적용)
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 [데이터 생성 시작] 목표 데이터 수: {NUM_SAMPLES:,}개")
    
    results = []
    
    # ProcessPoolExecutor를 사용하여 CPU 코어를 풀가동!
    with ProcessPoolExecutor() as executor:
        # tqdm으로 진행률 바 표시
        futures = executor.map(simulate_single_room, range(NUM_SAMPLES))
        for res in tqdm(futures, total=NUM_SAMPLES, desc="시뮬레이션 진행 중"):
            if res is not None:
                results.append(res)
                
    # 4. 데이터 프레임 변환 및 저장
    df = pd.DataFrame(results)
    
    # 저장할 폴더 생성
    os.makedirs('./data', exist_ok=True)
    save_path = './data/2d_acoustic_data_raw.csv'
    df.to_csv(save_path, index=False)
    
    print("\n🎉 데이터 생성 완료!")
    print(f" 📊 성공적으로 생성된 데이터: {len(df):,}개")
    print(f" 💾 저장 위치: {save_path}")
    
    # 데이터 살짝 구경하기
    print("\n[데이터 미리보기]")
    print(df.head())