# 🎧 SoundScape: AI-Powered Real-Time Spatial Audio Simulator

![SoundScape Demo](https://via.placeholder.com/800x400?text=SoundScape+UI+Screenshot) *(여기에 나중에 스크린샷을 넣으세요!)*

## 📖 프로젝트 개요 (Overview)
**SoundScape**는 방의 크기, 벽면 흡음률, 그리고 **장애물의 기하학적 위치에 따른 소리의 회절/차폐 현상**을 실시간으로 예측하고 시뮬레이션하는 AI 기반 공간 음향 대시보드입니다. 

기존 음향 물리 엔진(Ray Tracing)이 가진 막대한 연산 비용 문제를 해결하기 위해, 100만 개의 정밀 물리 시뮬레이션 데이터를 선행 학습한 **Residual 기반의 딥러닝 모델(SoundScapeNet-Pro)**을 자체 구축했습니다. 이를 통해 0.01초 이내의 초고속 추론으로 복잡한 공간 음향을 실시간으로 합성합니다.

## ✨ 핵심 엔지니어링 성과 (Key Achievements)

### 1. Hybrid Physics Engine 기반 데이터셋 구축 (1M Samples)
기존 `pyroomacoustics` 라이브러리의 한계(방 중앙에 위치한 복잡한 다각형 장애물 연산 시 발생하는 에러 및 메모리 병목)를 극복하기 위해 **하이브리드(Hybrid) 연산 파이프라인**을 설계했습니다.
* **배경 잔향(Reverberation):** `pyroomacoustics`의 Ray-tracing을 활용하여 방 크기와 재질(흡음률)에 따른 정밀한 L자형/Shoebox 공간의 반사파 에너지 계산.
* **직접음 차폐(Occlusion & Diffraction):** 순수 수학적 선분 교차 알고리즘(CCW)을 도입하여 장애물 차폐 여부를 $O(1)$ 수준으로 판별.
* **최적화 결과:** 객체 생성 오버헤드를 걷어내고 32코어 멀티프로세싱을 극대화하여 **데이터 생성 속도를 초당 21개에서 2,200개로 100배 이상 향상**.

### 2. 음향학 기반 회절 패널티 알고리즘 (Acoustic Algorithms)
장애물 뒤편의 소리 감쇠(Acoustic Shadow)를 현실적으로 구현하기 위해 다음과 같은 물리 기반 수학 연산을 적용했습니다.
* **거리 감쇠 (Inverse Square Law):** $L = -20 \cdot \log_{10}(d)$
* **장애물 회절 패널티:** 스피커와 청취자 사이의 Ray가 장애물(Obstacle) 박스를 관통할 경우, 장애물의 면적(두께)에 비례하는 감쇠율 적용.
  $$\text{Penalty (dB)} = 10.0 + 2.5 \cdot \sqrt{W_{\text{obs}} \times L_{\text{obs}}}$$

### 3. 고성능 딥러닝 아키텍처 (SoundScapeNet-Pro)
* **Input (12 Dims):** `[방 가로, 방 세로, 스피커X, 스피커Y, 청취자X, 청취자Y, 거리, 흡음률, 장애물X, 장애물Y, 장애물W, 장애물L]`
* **Architecture:** 비선형적 물리 현상(로그 감쇠 및 흡음 계수)을 학습하기 위해 **Residual Connection(잔차 연결)**과 `LayerNorm`, `SiLU` 활성화 함수를 결합한 딥러닝 네트워크 설계.
* **Performance:** 100만 개 데이터 학습(Epoch 50) 결과, **검증 오차(Val Loss) $0.0003$ 달성** (오차율 1% 미만 수준의 완벽한 물리 엔진 모사).

## 🛠️ 기술 스택 (Tech Stack)

### **V3 Architecture (Current - PoC & Prototype)**
* **AI & Data:** Python, PyTorch, Scikit-learn, Pandas, NumPy
* **Acoustics:** Pyroomacoustics, Librosa, Soundfile
* **Frontend/UI:** Streamlit, Matplotlib, Seaborn

### **V4 Architecture (Future Plans - Production)**
* 다중 스피커 배열(Multi-speaker Array) 환경에서의 에너지 중첩 원리($L_{\text{total}} = 10 \cdot \log_{10} (\sum 10^{\frac{L_i}{10}})$)를 적용하기 위한 백엔드/프론트엔드 분리 아키텍처 도입 예정.
* **Backend:** FastAPI (AI Inference API)
* **Frontend:** React.js, Web Audio API (실시간 노브 조절 및 GainNode 제어)

## 🚀 설치 및 실행 방법 (How to Run)

```bash
# 1. 저장소 클론 및 폴더 이동
git clone [https://github.com/YourUsername/SoundScape.git](https://github.com/YourUsername/SoundScape.git)
cd SoundScape

# 2. 가상환경 세팅 및 의존성 설치
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 실시간 대시보드 실행
streamlit run backend/src/4_app_v3.py

## 지금까지 한 일 (요약)

- **레포 구조 정리:** `backend/`, `frontend/`, 루트 스크립트(`preprocessing.py`, `room-simulation.py`, 등)를 정리했습니다.
- **데이터 전처리 및 파이프라인:** `backend/src/`와 루트의 전처리 스크립트(`preprocessing.py`, `2_preprocess_and_scale.py` 등)를 통해 원시 음향 데이터를 전처리하고 스케일러를 생성했습니다.
- **모델 학습 및 결과물:** 학습된 모델 파일들을 `backend/data/models/`에 저장했습니다 (`best_forward_model.pth`, `soundscape_model_pro_v3.pth`, `soundscape_model_pro.pth`).
- **처리된 데이터셋:** 전처리된 학습/테스트 데이터가 `backend/data/processed_soundscape/` 및 `backend/data/processed_v3/`에 저장되어 있습니다 (`X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`).
- **백엔드 구현:** `backend/main.py`에 FastAPI 기반 서버를 구현했고, 모델 로드(`soundscape_model_pro_v3.pth`) 및 `/predict`, `/predict_map` 엔드포인트를 제공합니다.
- **프론트엔드 초기화:** `frontend/`는 Vite + React 구조로 구성되어 있으며 개발 서버는 `npm run dev`로 실행합니다. 주요 진입점은 `frontend/src/App.jsx`입니다.

## 로컬에서 빠르게 확인하는 방법

1) 백엔드(가상환경 활성화 후):

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

2) 프론트엔드:

```bash
cd frontend
npm install
npm run dev
```

필요하시면 위 내용을 더 상세히(예: API 스펙, 환경변수, 데이터 생성 스크립트 사용법, 모델 재학습 가이드) 문서화해 드리겠습니다.

## 프로젝트 폴더 구조 (Directory Structure)

프로젝트의 주요 폴더와 파일 및 역할을 간단히 정리합니다.

- 루트 파일들:
  - `preprocessing.py`: 전체 파이프라인용 전처리 스크립트(루트 레벨 복수 스크립트 존재).
  - `room-simulation.py`: 물리 기반 룸 시뮬레이션 테스트 스크립트.
  - `test_sql.py`: DB/테스트 관련 유틸리티(있을 경우).

- `backend/`:
  - `main.py`: FastAPI 서버 진입점 — 모델 로드 및 `/predict`, `/predict_map` 엔드포인트 제공.
  - `requirements.txt`: 백엔드 의존성 목록.
  - `data/`: 모델 및 전처리된 데이터 저장소
    - `models/`: 학습된 모델 파일(`soundscape_model_pro_v3.pth` 등) 및 스케일러(`*.pkl`).
    - `processed_soundscape/`, `processed_v3/`: 전처리된 학습/테스트 데이터(`X_*.npy`, `y_*.npy`).
    - `voice/`: 음성 샘플(있을 경우).
  - `data_pra_v3/`, `data_v1/`, `data_v2/`: 원시/버전별 데이터 폴더.
  - `src/`: 데이터 생성, 전처리, 학습, 앱(대시보드) 관련 스크립트
    - 예: `1_generate_acoustic_data.py`, `2_preprocess_and_scale.py`, `3_train_model.py`, `4_app.py`, `train.py`, `visualize_eda.py` 등.

- `frontend/`:
  - React + Vite 기반 프론트엔드(개발 서버: `npm run dev`).
  - 주요 파일: `index.html`, `package.json`, `vite.config.js`, `eslint.config.js`.
  - `src/`: React 소스 코드
    - `App.jsx`: 주요 진입 컴포넌트(현재 작업 중인 파일).
    - `main.jsx`, 스타일 파일(`App.css`, `index.css`), `assets/` 등.
  - `public/`: 정적 자산(아이콘, 정적 HTML 등).

- 기타/유틸:
  - 프로젝트에 포함된 여러 전처리·실험용 스크립트가 루트 또는 `backend/src/`에 존재합니다. 각 스크립트는 데이터 생성 → 전처리 → 학습 → 배포 파이프라인의 일부입니다.

필요하시면 각 폴더 내부의 주요 파일에 대한 자세한 설명(예: 각 스크립트의 입력/출력, 사용 예제, 환경변수)을 추가해 드리겠습니다.

