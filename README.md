<p align="center">
  <img src="https://img.shields.io/badge/Machina-OCR-brightgreen?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSJ3aGl0ZSI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTg4IDMuNTkgOCA4IDh6bS0yLTEzaDR2NGgtNHptMCA2aDR2NGgtNHptLTYgMGg0djRINHptMC02aDR2NGgtNHoiLz48L3N2Zz4=" alt="Machina OCR">
  
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/AI-Document%20Intelligence-orange?style=for-the-badge" alt="AI Technology">
  <img src="https://img.shields.io/badge/License-Apache%202.0-red?style=for-the-badge" alt="License">
</p>

# 🤖 Machina OCR: 다국어 웹툰 생성을 위한 AI 데이터셋 구축 도구

## 🎯 프로젝트 목표
다국어 웹툰 제작을 위한 혁신적인 AI 데이터셋 구축 및 이미지/텍스트 인식 기술 개발

---

## 📘 프로젝트 배경
웹툰의 글로벌화와 다국어 콘텐츠 수요 증가에 대응하기 위해 AI 기반 이미지/텍스트 인식 솔루션 개발

---

## 🔍 모델 특징

### 1. 🔤 PaddleOCR
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-Advanced-2EA44F?style=flat-square)

**광학 문자 인식(OCR) 전문**
- 다국어 텍스트 인식
- 고정밀 문자 추출
- 복잡한 배경 대응
- 딥러닝 기반 문자 탐지

#### 환경 설정
```bash
# 의존성 설치
pip install paddlepaddle-gpu paddlenlp
pip install opencv-python shapely numpy
```

#### 주요 requirements
```
paddlepaddle>=2.4.2
paddlenlp>=2.5.0
opencv-python
shapely
numpy
tqdm
```

### 2. 🕵️ MMDetection/Groounding DINO
![MMDetection](https://img.shields.io/badge/GroundingDINO%20Detection-FF6F61?style=flat-square)

**객체 탐지 및 세분화**
- 다양한 객체 탐지 알고리즘
- 정밀한 객체 위치 식별
- 복잡한 이미지 분석
- 딥러닝 기반 탐지

#### 환경 설정
```bash
# PyTorch 및 MMDetection 설치
pip install torch torchvision
pip install mmcv mmdetection
```

#### 주요 requirements
```
torch>=1.7.0
torchvision
mmcv-full>=1.6.0
mmengine>=0.7.0
numpy
scipy
```

### 3. 🖌️ IOPaint
![IOPaint](https://img.shields.io/badge/IOPaint-Image%20Restoration-blueviolet?style=flat-square)

**이미지 복원 및 인페인팅**
- AI 기반 이미지 결함 복원
- 지능형 이미지 편집
- 확산 모델 활용
- 고급 이미지 처리

#### 환경 설정
```bash
# 의존성 설치
pip install torch diffusers transformers
pip install opencv-python
```

#### 주요 requirements
```
torch>=2.0.0
diffusers==0.27.2
transformers>=4.39.1
opencv-python
huggingface_hub
```

---

## 🚀 통합 워크플로우

1. **문서 입력**: 원본 이미지/문서 로드
2. **객체 탐지(MMDetection)**: 문서 영역 및 구조 분석
3. **텍스트 인식(PaddleOCR)**: 문자 영역 추출 및 인식
4. **이미지 복원(IOPaint)**: 필요시 이미지 결함 보정

---

## 🚀 빠른 시작

### 사전 요구사항
- Python 3.8+
- Git
- CUDA (권장, GPU 가속)

### 설치 및 실행
```bash
# 저장소 클론
git clone https://github.com/yourusername/machina-ocr.git
cd machina-ocr

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or 
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt

# 모델 한 번에 실행
python result.py
```

## 🤝 기여 방법
1. 프로젝트 Fork
2. Feature 브랜치 생성
3. 변경사항 커밋
4. 브랜치에 Push
5. Pull Request 오픈

---

## 📄 라이센스
Apache License 2.0
