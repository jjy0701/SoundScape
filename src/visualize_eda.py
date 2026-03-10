import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda_visualization(target_room='ROOM_0001'):
    print(f"📊 [{target_room}] 데이터 시각화(EDA) 시작...")

    # 1. DB 연결 및 데이터 추출 (Extract from DB)
    db_path = './data/soundscape.db'
    if not os.path.exists(db_path):
        print("❌ DB 파일을 찾을 수 없습니다. 전처리 코드를 먼저 실행해주세요.")
        return

    conn = sqlite3.connect(db_path)
    
    # 🎯 SQL: 특정 방의 마이크 좌표, 소리 크기, 그리고 스피커 위치 가져오기
    query = f"""
        SELECT mic_x, mic_y, sound_energy_dB, speaker_x, speaker_y, room_width, room_length
        FROM acoustic_data_mart
        WHERE room_id = '{target_room}';
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"❌ {target_room}의 데이터가 없습니다.")
        return

    print(f"   -> 데이터 로드 완료! (총 {len(df)}개 마이크 센서 데이터)")

    # 스피커 위치 및 방 크기 정보 추출 (모든 행이 같은 값을 가지므로 첫 번째 행 사용)
    spk_x = df['speaker_x'].iloc[0]
    spk_y = df['speaker_y'].iloc[0]
    room_w = df['room_width'].iloc[0]
    room_l = df['room_length'].iloc[0]

    # 2. 데이터 구조 변경 (Pivot)
    # 1차원 리스트 형태의 데이터를 2차원 바둑판(Grid) 형태로 변환합니다.
    pivot_df = df.pivot_table(index='mic_y', columns='mic_x', values='sound_energy_dB')
    pivot_df = pivot_df.sort_index(ascending=False) # y축이 아래에서 위로 향하도록 정렬

    # 3. Seaborn을 활용한 히트맵 시각화
    plt.figure(figsize=(10, 8))
    
    # 히트맵 그리기 (소리가 클수록 밝은 노란색, 작을수록 어두운 보라색/검은색)
    sns.heatmap(pivot_df, cmap='magma', cbar_kws={'label': 'Sound Energy (dB)'})
    
    # 차트 꾸미기
    plt.title(f'Acoustic Heatmap: {target_room}\n(Room Size: {room_w}m x {room_l}m)', fontsize=16, pad=20)
    plt.xlabel('X Coordinate (m)', fontsize=12)
    plt.ylabel('Y Coordinate (m)', fontsize=12)
    
    # 스피커 위치를 좌표계에 맞게 변환하여 그리기 (★ 표시)
    # 히트맵은 index/columns 기반이므로 실제 물리적 미터(m)를 인덱스 위치로 변환해야 함
    x_step = 0.5 # 우리가 설정했던 마이크 간격
    spk_plot_x = (spk_x / x_step) - 0.5 
    spk_plot_y = (room_l - spk_y) / x_step - 0.5
    
    plt.scatter([spk_plot_x], [spk_plot_y], color='cyan', marker='*', s=300, edgecolor='black', label='Speaker')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 4. 이미지 저장 및 출력
    save_path = f'./data/{target_room}_heatmap.png'
    plt.savefig(save_path, dpi=300)
    print(f"   -> 시각화 완료! 이미지 저장됨: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # ROOM_0001 방의 히트맵을 그려봅니다. 
    # 원한다면 ROOM_0002, ROOM_0005 등으로 바꿔서 확인해보세요!
    run_eda_visualization('ROOM_0001')