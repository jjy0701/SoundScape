import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# 설정
fs = 44100  # 샘플링 레이트 (표준 고음질)
seconds = 5  # 녹음 시간

print("🎤 [녹음 준비] 마이크에 대고 말씀하세요...")
print("시작하려면 엔터를 누르세요!")
input()

print(f"🔴 녹음 중... ({seconds}초 동안)")
# 녹음 수행 (채널 1: 모노)
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # 녹음이 끝날 때까지 대기

print("✅ 녹음 완료!")

# 파일 저장 (src 폴더가 아니라 프로젝트 루트에 저장됨)
output_path = "my_voice.wav"
write(output_path, fs, recording)

print(f"🎉 '{output_path}' 파일이 생성되었습니다. 이제 Streamlit에 업로드하세요!")