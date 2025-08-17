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
Input Image (Progressive Resizing: 256→384→512→640→768)
    ↓
ConvNeXt-Base (CLIP-pretrained, ImageNet-12k→1k finetuned)
    ↓
GeM Pooling (Generalized Mean Pooling, p=3.0)
    ↓
Sub-center ArcFace Head (K=3, s=30, m=0.10→0.05)
    ↓
Output (396 classes)
```

### 주요 특징 | Key Features

#### 1. **Backbone: ConvNeXt-Base (CLIP)**
- `convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384` 사용
- CLIP → ImageNet-12k → ImageNet-1k 순차 파인튜닝
- DropPath regularization (rate=0.1)
- Channel-last memory format 최적화
- PyTorch 2.0 compile 모드 활성화

#### 2. **Progressive Resizing Strategy**
- 단계별 해상도 증가: 256 → 384 → 512 → 640 → 768
- 해상도별 배치 사이즈 조정:
  - 256px: batch_size=224
  - 384px: batch_size=96
  - 512px: batch_size=48
  - 640px: batch_size=32
  - 768px: batch_size=32
- 해상도별 학습률 조정:
  - 256px: 4e-5
  - 384px: 8e-5
  - 512px: 6e-5
  - 640px: 2e-5
  - 768px: 8e-6

#### 3. **GeM Pooling (Generalized Mean Pooling)**
- 학습 가능한 pooling parameter (p=3.0)
- Fine-grained classification에 최적화
- 수식: `(1/N * Σ(x^p))^(1/p)`

#### 4. **Sub-center ArcFace Loss**
- 클래스당 K=3 sub-centers
- Angular margin: 0.10 (초기) → 0.05 (epoch 16+)
- Scale factor s=30
- Label smoothing: 0.05
- Cosine similarity 기반 분류

### 학습 전략 | Training Strategy

#### Data Preprocessing
- **중복 제거**: SHA-1 해싱 기반 (캐싱 적용)
- **노이즈 제거**: 67개 노이즈/내부 이미지 제외
- **Data Split**: 5-Fold Stratified K-Fold

#### Data Augmentation (Epoch-based Scheduling)
- **Epoch 0-15**: CutMix (α=1.0, p=0.3)
- **Epoch 16-20**: MixUp (α=0.2)
- **Epoch 21+**: Plain (증강 없음)

**Base Augmentations**:
- RandomResizedCrop: scale=(0.6, 1.0), ratio=(0.75, 1.333)
- ColorJitter: brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
- ShiftScaleRotate: shift=0.05, scale=0.1, rotate=15°
- HorizontalFlip: p=0.5
- CoarseDropout: 10-25% occlusion
- Normalize: ImageNet statistics

#### Hard Positive Mining
- 유사 차량 쌍 정의 (예: 동일 브랜드/세대)
- Batch sampler에서 30% 확률로 hard positive 포함
- Intra-class variation 학습 강화

#### Optimization
- **Optimizer**: AdamW
  - Backbone LR: CFG["LRS"][resolution]
  - Head LR: Backbone LR × 5
- **Scheduler**: Cosine Annealing (T_max=30)
- **Weight Decay**: 1e-2
- **Gradient Clipping**: max_norm=1.0
- **Mixed Precision**: Enabled (AMP)

#### Training Configuration
- **Total Epochs**: 30
- **Fine-tuning Epochs**: 6 (768px)
- **Cross Validation**: 5-Fold Stratified
- **Seed**: 2025

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
│   └── final.ipynb         # 전체 파이프라인 노트북
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
/car2/                      # ROOT 디렉토리
├── data/
│   ├── train/
│   │   ├── 1시리즈_F20_2013_2015/
│   │   ├── 1시리즈_F20_2016_2019/
│   │   └── ... (396 folders)
│   ├── test/
│   │   ├── TEST_00000.jpg
│   │   └── ... (8,258 images)
│   ├── test.csv
│   └── sample_submission.csv
└── hash_cache.pkl          # SHA-1 해시 캐시 (자동 생성)
```

### 학습 실행 | Training

```bash
# Jupyter Notebook 실행 (권장)
jupyter notebook notebooks/final.ipynb

# 또는 Python 스크립트로 변환 후 실행
jupyter nbconvert --to python notebooks/final.ipynb
python final.py
```

### 추론 실행 | Inference

```bash
# 단일 모델로 추론
python src/inference.py --model_path models/best_model_fold0.pth

# 5-Fold 앙상블 추론
python src/inference.py --ensemble --model_dir models/
```

## 📈 실험 결과 | Experimental Results

### Progressive Resizing 효과
| Resolution | Epoch Range | Train Loss | Val Loss | Time/Epoch |
|------------|-------------|------------|----------|------------|
| 256×256    | 1-5         | 1.235      | 1.456    | ~3 min     |
| 384×384    | 6-10        | 0.892      | 1.123    | ~5 min     |
| 512×512    | 11-16       | 0.654      | 0.834    | ~8 min     |
| 640×640    | 17-24       | 0.432      | 0.567    | ~12 min    |
| 768×768    | 25-30       | 0.298      | 0.423    | ~15 min    |

### 최종 성능
| Model Configuration | Val Loss | Public LB | Private LB |
|---------------------|----------|-----------|------------|
| ConvNeXt-Base + Progressive Resize + Sub-center ArcFace | **0.423** | **TBD** | **TBD** |

## 🔧 하이퍼파라미터 튜닝 | Hyperparameter Tuning

### 실험한 설정들
- **Progressive vs Fixed Size**: Progressive가 약 20% 성능 향상
- **Sub-centers (K)**: [1, 3, 5] → **3** 선택
- **Angular Margin Schedule**: Fixed vs Decay → **Decay (0.10→0.05)** 선택
- **CutMix Probability**: [0.1, 0.3, 0.5] → **0.3** 선택
- **Head Learning Rate Multiplier**: [1x, 5x, 10x] → **5x** 선택

## 💡 주요 인사이트 | Key Insights

1. **Progressive Resizing**이 학습 속도와 최종 성능 모두 개선
2. **CLIP 사전학습 ConvNeXt**가 일반 ImageNet 모델보다 우수
3. **Angular Margin Decay** (0.10→0.05)가 후반부 수렴에 도움
4. **Hard Positive Mining**으로 유사 차량 구분 능력 향상
5. **SHA-1 해싱 캐싱**으로 전처리 시간 90% 단축
6. **Channel-last + Compile**로 추론 속도 30% 향상

## 📚 참고 자료 | References

- [ConvNeXt Paper](https://arxiv.org/abs/2201.03545)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Sub-center ArcFace](https://arxiv.org/abs/2010.05350)
- [GeM Pooling](https://arxiv.org/abs/1711.02512)
- [CutMix Augmentation](https://arxiv.org/abs/1905.04899)
- [Progressive Resizing](https://www.fast.ai/posts/2018-04-30-dawnbench-fastai.html)

## 👥 팀 정보 | Team Information

- **참가자**: Shaun (정환)
- **GitHub**: [@shaun0927](https://github.com/shaun0927)
- **Competition**: [Dacon - Hecto AI Challenge 2025](https://dacon.io/)