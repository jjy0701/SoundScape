import sqlite3
import pandas as pd

def run_sql_test():
    print("🔍 SQLite 데이터베이스 SQL 쿼리 테스트 시작!\n")
    
    # 1. 창고(DB) 문 열기 (연결)
    db_path = './data/soundscape.db'
    conn = sqlite3.connect(db_path)
    
    # ---------------------------------------------------------
    # 🎯 테스트 1: 데이터가 진짜 잘 들어갔을까? (기본 조회)
    # ---------------------------------------------------------
    print("📝 [쿼리 1] 전체 데이터 중 무작위 5줄만 가져오기 (SELECT * LIMIT)")
    query1 = """
        SELECT * FROM acoustic_data_mart 
        LIMIT 5;
    """
    # SQL 쿼리 결과를 예쁜 Pandas 데이터프레임으로 바로 받아옵니다.
    df_sample = pd.read_sql_query(query1, conn)
    print(df_sample.to_string(index=False))
    print("-" * 60)

    # ---------------------------------------------------------
    # 🎯 테스트 2: 데이터 분석가다운 SQL 활용 (그룹핑 및 집계)
    # ---------------------------------------------------------
    print("📊 [쿼리 2] 각 방(room_id)별 마이크 개수와 평균 소리 크기 비교 (GROUP BY)")
    # CSV였다면 파이썬 코드를 길게 짜야 하지만, DB에 넣었기 때문에 SQL 몇 줄로 끝납니다!
    query2 = """
        SELECT 
            room_id,
            COUNT(sound_energy_dB) AS mic_count,
            ROUND(AVG(sound_energy_dB), 2) AS avg_dB,
            ROUND(MAX(sound_energy_dB), 2) AS max_dB
        FROM acoustic_data_mart
        GROUP BY room_id
        ORDER BY avg_dB DESC;
    """
    df_grouped = pd.read_sql_query(query2, conn)
    print(df_grouped.to_string(index=False))
    print("-" * 60)

    # ---------------------------------------------------------
    # 🎯 테스트 3: 특정 조건 필터링 (WHERE)
    # ---------------------------------------------------------
    print("🚨 [쿼리 3] 1번 방(ROOM_0001)에서 소리가 가장 큰 상위 3개 좌표 찾기")
    query3 = """
        SELECT room_id, mic_x, mic_y, sound_energy_dB 
        FROM acoustic_data_mart
        WHERE room_id = 'ROOM_0001'
        ORDER BY sound_energy_dB DESC
        LIMIT 3;
    """
    df_top3 = pd.read_sql_query(query3, conn)
    print(df_top3.to_string(index=False))
    print("-" * 60)

    # 2. 창고(DB) 문 닫기
    conn.close()
    print("✅ SQL 테스트 완료! 창고 문을 닫았습니다.")

if __name__ == "__main__":
    run_sql_test()