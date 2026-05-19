import pandas as pd
import glob

# 1. 분산된 10개의 파일 목록 가져오기
file_list = glob.glob("acoustic_data_part_*.parquet")
print(f"발견된 파일 개수: {len(file_list)}개")

# 2. 모든 데이터 하나로 합치기
df_list = [pd.read_parquet(f) for f in file_list]
full_df = pd.concat(df_list, ignore_index=True)

# 3. 전체 데이터 건강검진
print("\n--- 전체 데이터 요약 ---")
print(f"총 샘플 수: {len(full_df):,}개")
print(full_df.describe())

# 4. 학습용 통합 파일로 저장 (선택 사항)
# full_df.to_parquet("acoustic_data_full_1M.parquet")