import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def process_and_scale_data():
    print("🚀 [Step 2] 데이터 전처리 및 AI용 스케일링 시작 (흡음률 포함)...")
    
    raw_data_path = './data/2d_acoustic_data_raw.csv'
    df = pd.read_csv(raw_data_path)
    print(f"   -> 원본 데이터 로드 완료: {len(df):,}개")

    initial_len = len(df)
    df = df.dropna()  
    df = df[(df['sound_energy_dB'] < 150) & (df['sound_energy_dB'] > -100)] 
    print(f"   -> 이상치 정제 완료: 남은 데이터 {len(df):,}개")

    # 💡 [핵심 변경] 특징(Feature)이 7개에서 8개로 늘어났습니다!
    features = ['room_width', 'room_length', 'speaker_x', 'speaker_y', 'mic_x', 'mic_y', 'distance', 'absorption']
    target = ['sound_energy_dB']

    X = df[features]
    y = df[target]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler() 

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    os.makedirs('./data/models', exist_ok=True)
    joblib.dump(scaler_X, './data/models/scaler_X.pkl')
    joblib.dump(scaler_y, './data/models/scaler_y.pkl')
    print("   -> 💾 스케일러(번역기) 저장 완료")

    combined_scaled = np.hstack((X_scaled, y_scaled))
    train_data, test_data = train_test_split(combined_scaled, test_size=0.15, random_state=42)

    os.makedirs('./data/processed_soundscape', exist_ok=True)
    train_df = pd.DataFrame({'scaled_features': [{'values': row.tolist()} for row in train_data]})
    train_df.to_parquet('./data/processed_soundscape/train.parquet', engine='pyarrow')

    test_df = pd.DataFrame({'scaled_features': [{'values': row.tolist()} for row in test_data]})
    test_df.to_parquet('./data/processed_soundscape/test.parquet', engine='pyarrow')

    print("\n🎉 모든 전처리(8 features) 완료!")

if __name__ == "__main__":
    process_and_scale_data()