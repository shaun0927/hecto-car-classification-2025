# 🚗 Hecto Car Classification Challenge 2025

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 대회 개요 | Competition Overview

**HAI(하이)! - Hecto AI Challenge : 2025 상반기 헥토 채용 AI 경진대회**

최근 자동차 산업의 디지털 전환과 더불어, 다양한 차종을 빠르고 정확하게 인식하는 기술의 중요성이 커지고 있습니다. 본 대회는 실제 중고차 차량 이미지를 기반으로 한 차종 분류 AI 모델 개발을 목표로 합니다.

### 🎯 목표 | Objective
- **Task**: 차량 이미지 멀티클래스 분류 (Multi-class Car Image Classification)
- **Classes**: 396개의 차량 모델
- **Metric**: Log Loss (낮을수록 좋음)

### 📊 데이터셋 | Dataset
- **Training Set**: 33,137장 (396개 클래스)
- **Test Set**: 8,258장
- **Image Format**: JPG
- **Class Distribution**: 클래스당 약 80-100장

### 🏆 평가 기준 | Evaluation
- **Metric**: Multi-class Log Loss
- **Public Score**: 테스트 데이터의 50%
- **Private Score**: 테스트 데이터의 나머지 50%

#### 특별 처리 클래스 (동일 클래스로 간주)
1. `K5_3세대_하이브리드_2020_2022` ≡ `K5_하이브리드_3세대_2020_2023`
2. `디_올뉴니로_2022_2025` ≡ `디_올_뉴_니로_2022_2025`
3. `718_박스터_2017_2024` ≡ `박스터_718_2017_2024`
4. `RAV4_2016_2018` ≡ `라브4_4세대_2013_2018`
5. `RAV4_5세대_2019_2024` ≡ `라브4_5세대_2019_2024`

## 🏗️ 솔루션 아키텍처 | Solution Architecture

### 모델 구조
```
Input Image (448x448)
    ↓
ConvNeXt-Base (Backbone)
    ↓
GeM Pooling (Generalized Mean Pooling)
    ↓
Sub-center ArcFace Head (K=3)
    ↓
Output (396 classes)
```

### 주요 특징 | Key Features

#### 1. **Backbone: ConvNeXt-Base**
- ImageNet-22k 사전학습 → ImageNet-1k 파인튜닝
- 현대적인 CNN 아키텍처
- 효율적인 특징 추출

#### 2. **Pooling: GeM (Generalized Mean Pooling)**
- 학습 가능한 pooling parameter (p=3.0)
- Fine-grained classification에 최적화
- 수식: `(1/N * Σ(x^p))^(1/p)`

#### 3. **Head: Sub-center ArcFace**
- 클래스당 K=3 sub-centers
- Intra-class variation 처리
- Cosine similarity 기반 분류
- Scale factor s=30

### 학습 전략 | Training Strategy

#### Data Augmentation
- **RandomResizedCrop**: scale=(0.6, 1.0), ratio=(0.75, 1.333)
- **ColorJitter**: brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
- **ShiftScaleRotate**: shift=0.05, scale=0.1, rotate=15°
- **HorizontalFlip**: p=0.5
- **CoarseDropout**: 10-25% occlusion
- **CutMix**: α=1.0, p=0.5

#### Optimization
- **Optimizer**: AdamW
  - Backbone LR: 3e-4
  - Head LR: 3e-3 (10x higher)
- **Scheduler**: Cosine Annealing with 3-epoch warmup
- **Weight Decay**: 1e-2
- **EMA**: decay=0.999

#### Training Configuration
- **Epochs**: 20
- **Batch Size**: 32
- **Image Size**: 448×448
- **Cross Validation**: 5-Fold Stratified
- **Mixed Precision**: Enabled (AMP)

## 📁 프로젝트 구조 | Project Structure

```
hecto-car-classification-2025/
│
├── 📁 src/
│   ├── train.py           # 학습 메인 스크립트
│   ├── inference.py        # 추론 스크립트
│   ├── dataset.py          # Dataset 및 DataLoader
│   ├── models.py           # 모델 아키텍처
│   ├── augmentations.py    # 데이터 증강
│   └── utils.py            # 유틸리티 함수
│
├── 📁 notebooks/
│   └── pipeline.ipynb      # 전체 파이프라인 노트북
│
├── 📁 configs/
│   └── config.yaml         # 설정 파일
│
├── 📁 data/                # 데이터 디렉토리 (not included)
│   ├── train/              # 학습 이미지
│   ├── test/               # 테스트 이미지
│   └── *.csv               # CSV 파일들
│
├── 📁 models/              # 학습된 모델 저장
├── 📁 results/             # 예측 결과 저장
│
├── requirements.txt        # 패키지 의존성
├── README.md              # 프로젝트 문서
└── .gitignore             # Git 제외 파일
```

## 🚀 시작하기 | Getting Started

### 환경 설정 | Environment Setup

```bash
# 1. 저장소 클론
git clone https://github.com/shaun0927/hecto-car-classification-2025.git
cd hecto-car-classification-2025

# 2. Conda 환경 생성
conda create -n hecto_car python=3.10 -y
conda activate hecto_car

# 3. 패키지 설치
pip install -r requirements.txt

# 4. (선택) CUDA 11.8용 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 데이터 준비 | Data Preparation

```bash
# 데이터 디렉토리 구조
data/
├── train/
│   ├── 1시리즈_F20_2013_2015/
│   ├── 1시리즈_F20_2016_2019/
│   └── ... (396 folders)
├── test/
│   ├── TEST_00000.jpg
│   └── ... (8,258 images)
├── test.csv
└── sample_submission.csv
```

### 학습 실행 | Training

```bash
# 단일 Fold 학습
python src/train.py --fold 0

# 전체 5-Fold 학습
python src/train.py --train_all_folds

# Custom 설정으로 학습
python src/train.py --config configs/custom_config.yaml
```

### 추론 실행 | Inference

```bash
# 단일 모델로 추론
python src/inference.py --model_path models/best_model_fold0.pth

# 5-Fold 앙상블 추론
python src/inference.py --ensemble --model_dir models/
```

## 📈 실험 결과 | Experimental Results

| Model | Fold | Val Loss | Public LB | Private LB |
|-------|------|----------|-----------|------------|
| ConvNeXt-Base + GeM + ArcFace | Fold 0 | 0.4243 | - | - |
| ConvNeXt-Base + GeM + ArcFace | Fold 1 | 0.4187 | - | - |
| ConvNeXt-Base + GeM + ArcFace | Fold 2 | 0.4201 | - | - |
| ConvNeXt-Base + GeM + ArcFace | Fold 3 | 0.4156 | - | - |
| ConvNeXt-Base + GeM + ArcFace | Fold 4 | 0.4198 | - | - |
| **5-Fold Ensemble** | - | **0.4197** | **TBD** | **TBD** |

## 🔧 하이퍼파라미터 튜닝 | Hyperparameter Tuning

### 실험한 설정들
- **Image Size**: [384, 448, 512] → **448** 선택
- **Batch Size**: [16, 32, 64] → **32** 선택
- **Learning Rate**: [1e-4, 3e-4, 5e-4] → **3e-4** 선택
- **Sub-centers (K)**: [1, 3, 5] → **3** 선택
- **CutMix Alpha**: [0.5, 1.0, 2.0] → **1.0** 선택

## 💡 주요 인사이트 | Key Insights

1. **Sub-center ArcFace**가 일반 Softmax보다 약 15% 성능 향상
2. **GeM Pooling**이 Global Average Pooling보다 효과적
3. **CutMix + Strong Augmentation** 조합이 과적합 방지에 효과적
4. **Differential Learning Rate** (Head 10x)가 수렴 속도 개선
5. **EMA (Exponential Moving Average)**로 안정적인 예측

## 📚 참고 자료 | References

- [ConvNeXt Paper](https://arxiv.org/abs/2201.03545)
- [Sub-center ArcFace](https://arxiv.org/abs/2010.05350)
- [GeM Pooling](https://arxiv.org/abs/1711.02512)
- [CutMix Augmentation](https://arxiv.org/abs/1905.04899)

## 👥 팀 정보 | Team Information

- **참가자**: Shaun (정환)
- **GitHub**: [@shaun0927](https://github.com/shaun0927)
- **Competition**: [Dacon - Hecto AI Challenge 2025](https://dacon.io/)
