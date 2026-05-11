import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def process_and_scale_data():
    print("🚀 [Step 2] 데이터 전처리 및 AI용 스케일링 시작...")
    
    # 1. 1단계에서 만든 데이터 불러오기
    raw_data_path = './data/2d_acoustic_data_raw.csv'
    if not os.path.exists(raw_data_path):
        print(f"❌ 파일을 찾을 수 없습니다: {raw_data_path}")
        return
    
    df = pd.read_csv(raw_data_path)
    print(f"   -> 원본 데이터 로드 완료: {len(df):,}개")

    # 2. 결측치 및 이상치 제거 (물리 엔진 연산 중 튄 에러 데이터 정리)
    initial_len = len(df)
    df = df.dropna()  # 결측치 제거
    df = df[(df['sound_energy_dB'] < 150) & (df['sound_energy_dB'] > -100)] # 비현실적인 데시벨 제거
    
    print(f"   -> 이상치/결측치 정제 완료: {initial_len - len(df):,}개 제거됨 (남은 데이터: {len(df):,}개)")

    # 3. 입력(X)과 정답(y) 분리
    # 기획서에 명시된 7개의 특징(Feature)과 1개의 라벨(Label)
    features = ['room_width', 'room_length', 'speaker_x', 'speaker_y', 'mic_x', 'mic_y', 'distance']
    target = ['sound_energy_dB']

    X = df[features]
    y = df[target]

    # 4. 데이터 스케일링 (Min-Max 정규화)
    # 인공지능은 0~1 사이의 숫자를 가장 잘 이해합니다. 길이(m)와 소리(dB)의 단위를 통일시킵니다.
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler() 

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 💡 [핵심] 스케일러 저장!
    # 나중에 Streamlit 화면에서 AI가 뱉어낸 '0.8'이라는 숫자를 '85dB'로 다시 번역(해독)하려면 
    # 이 스케일러 파일이 반드시 필요합니다.
    os.makedirs('./data/models', exist_ok=True)
    joblib.dump(scaler_X, './data/models/scaler_X.pkl')
    joblib.dump(scaler_y, './data/models/scaler_y.pkl')
    print("   -> 💾 스케일러(번역기) 저장 완료: ./data/models/scaler_y.pkl")

    # 5. PyTorch 데이터셋 구조에 맞게 X와 y를 가로로 이어붙이기 (총 8개 컬럼)
    combined_scaled = np.hstack((X_scaled, y_scaled))

    # 6. Train / Test 분리 (기획서 명시: 85% 학습, 15% 시험)
    train_data, test_data = train_test_split(combined_scaled, test_size=0.15, random_state=42)

    # 7. 파이토치가 가장 빠르게 읽을 수 있는 Parquet 포맷으로 변환 후 저장
    os.makedirs('./data/processed_soundscape', exist_ok=True)

    # {'scaled_features': [{'values': [데이터 배열]}]} 형태로 DataFrame 구성
    train_df = pd.DataFrame({'scaled_features': [{'values': row.tolist()} for row in train_data]})
    train_df.to_parquet('./data/processed_soundscape/train.parquet', engine='pyarrow')

    test_df = pd.DataFrame({'scaled_features': [{'values': row.tolist()} for row in test_data]})
    test_df.to_parquet('./data/processed_soundscape/test.parquet', engine='pyarrow')

    print("\n🎉 모든 전처리 및 변환 완료!")
    print(f" 📊 Train 데이터 (모의고사용): {len(train_data):,}개")
    print(f" 📊 Test 데이터 (수능용): {len(test_data):,}개")
    print(" 💾 저장 위치: ./data/processed_soundscape/")
    print("🚀 이제 파이토치(PyTorch) 모델 학습 코드를 실행할 준비가 완벽히 끝났습니다!")

if __name__ == "__main__":
    process_and_scale_data()