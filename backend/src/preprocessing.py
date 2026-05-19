import pandas as pd
import sqlite3
import numpy as np
import os

def run_preprocessing_and_load():
    print("🧹 [Step 1] 데이터 전처리(Transform) 시작...")
    
    # 1. 원본 데이터 불러오기
    raw_file_path = './data/soundscape_mass_data.csv'
    if not os.path.exists(raw_file_path):
        print(f"❌ 원본 파일을 찾을 수 없습니다: {raw_file_path}")
        return
        
    df = pd.read_csv(raw_file_path)
    print(f"   -> 원본 데이터 로드 완료 (총 {len(df):,}행)")
    
    # 상태 파악: 결측치와 이상치가 얼마나 있는지 확인
    initial_nulls = df['sound_energy_dB'].isna().sum()
    outliers = len(df[df['sound_energy_dB'] > 150])
    print(f"   [상태 보고] 초기 결측치: {initial_nulls}개, 이상치(150dB 초과): {outliers}개")

    # 2. 이상치(Outlier) 처리: 150dB가 넘는 폭음 데이터를 찾아 결측치(NaN)로 깎아버림
    df.loc[df['sound_energy_dB'] > 150, 'sound_energy_dB'] = np.nan
    
    # 3. 결측치(Missing Value) 대체: 단순히 전체 평균이 아니라, '해당 방(room_id)의 평균값'으로 정교하게 채우기
    # 실무에서 아주 많이 쓰이는 그룹별 결측치 대체 (Groupby Transform) 로직입니다!
    df['sound_energy_dB'] = df['sound_energy_dB'].fillna(
        df.groupby('room_id')['sound_energy_dB'].transform('mean')
    )
    
    # 결과 확인
    final_nulls = df['sound_energy_dB'].isna().sum()
    print(f"   -> 전처리 완료! 남은 결측치: {final_nulls}개 (모두 해당 방의 평균값으로 복구 성공)")

    # ---------------------------------------------------------
    print("\n💾 [Step 2] 데이터베이스 적재(Load) 시작...")
    
    # 4. SQLite 데이터베이스 연결 (파일이 없으면 자동으로 생성됨)
    db_path = './data/soundscape.db'
    conn = sqlite3.connect(db_path)
    
    # 5. Pandas 데이터를 DB 테이블('acoustic_data_mart')로 밀어 넣기
    # if_exists='replace': 기존 테이블이 있으면 지우고 새로 덮어쓰기
    df.to_sql('acoustic_data_mart', conn, if_exists='replace', index=False)
    
    # 연결 종료
    conn.close()
    
    print(f"   -> 데이터베이스 적재 완료! (DB 파일 위치: {db_path})")
    print(f"   -> 생성된 테이블 이름: 'acoustic_data_mart'")
    print("🎉 ETL 파이프라인 (Transform -> Load) 성공적으로 종료!")

if __name__ == "__main__":
    run_preprocessing_and_load()