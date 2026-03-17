# 자금세탁 탐지 고도화를 위한 GNN-XGBoost 하이브리드 모델링 비교 연구

> **전통적 머신러닝(XGBoost) 대비 그래프 구조 정보 활용에 따른 탐지 성능 개선 효과 분석**

팀명: **자금의 트레이서** | 작성자: 김규빈, 윤가람

---

## 📌 프로젝트 개요

전통적인 통계 기반 ML 모델은 개별 계좌의 수치적 이상 징후는 잘 포착하지만, 자금세탁의 핵심인 **계좌 간 복잡한 연결망(Topology)** 과 거래의 시간적·수단적 맥락을 통합적으로 이해하는 데 한계가 있습니다.

본 프로젝트는 이를 해결하기 위해 **그래프 신경망(GNN)** 을 도입하여 조직적인 자금세탁 구조와 지능적인 위장 거래를 동시에 탐지하고, 기존 ML 모델 대비 성능 향상 폭을 실증합니다.

---

## 🏗️ 모델 아키텍처

### 전체 파이프라인

```
원본 데이터 (IBM HI-Medium)
    ├── 거래 데이터 (trans.csv)      → 엣지 피처 생성 (17차원)
    └── 계좌 데이터 (accounts.csv)   → 노드 피처 생성

노드 피처 구성 (최대 81차원)
    ├── V2 행동 통계 피처 (38개)
    ├── 그래프 구조 피처 (13개): PageRank, Degree, Reciprocity 등
    ├── 조건부 행동 피처 (10개): 그래프 × 통계 교차항
    ├── Cross-border 정밀화 (4개)
    ├── 거래량 대비 정규화 (4개)
    ├── 시간 윈도우 변화 (2개)
    ├── 입출금 독립 피처 (4개)
    ├── 규칙성 피처 (4개)
    └── 금액 분포 피처 (2개)

엣지 피처 구성 (17차원)
    ├── 시간 맥락 (5d): 정규화 시각, sin/cos(hour·weekday)
    ├── 금액 맥락 (4d): log금액, round number, is_large_tx, 백분위
    └── 결제 포맷 (8d): 7종 원-핫 + AML 위험도

GNN Stage (임베딩 추출)
    └── 최적 레이어 선택 (Multi-config 탐색)
         ├── TransformerConv (attention-weighted SUM)
         ├── GATv2Conv (동적 어텐션 SUM)
         ├── NNConv + MAX (엣지 변환행렬 × MAX 집계) ← AML 최적
         └── GINEConv + SUM (WL 이론적 최강 표현력)

XGBoost Stage (최종 분류)
    └── 3-Stage 최적화
         ├── Stage 1: 전체 피처 학습 → Importance 순위 산출
         ├── Stage 2: Forward Selection → 유효 피처만 채택
         └── Stage 3: HP Grid Search → 최적 하이퍼파라미터 확정
```

### 2단계 하이브리드 설계 이유

| 단계 | 모델 | 역할 |
|------|------|------|
| Stage 1 | GNN | 계좌 간 관계(Topology)를 저차원 임베딩으로 압축 |
| Stage 2 | XGBoost | 임베딩 + 수작업 피처를 결합하여 최종 분류 |

GNN 단독 분류기 대신 XGBoost를 최종 분류기로 선택한 이유는 다음과 같습니다.
- **불균형 보정**: `scale_pos_weight` 파라미터로 0.98% 양성 비율을 효과적으로 처리
- **해석 가능성**: Feature importance / SHAP으로 수사 근거 설명 가능
- **연산 효율**: GNN 재학습 없이 XGBoost만 주기적 업데이트 가능

---

## 📊 실험 결과

### 최종 성능 비교표

| 모델 | AUPRC | Precision | F1 | Recall | Top-5000 |
|------|-------|-----------|-----|--------|----------|
| ML 앙상블 (V2+V3) | 0.5496 | 0.1539 | 0.2600 | **0.8387 ★** | **4,645명 ★** |
| TGAT TransformerConv | 0.5794 | 0.4904 | 0.5314 | 0.5799 | 4,363명 |
| **TGAT NNConv+MAX** | 0.5842 | **0.5199 ★** | **0.5445 ★** | 0.5716 | 4,279명 |
| **GraphSAGE ADAN-Ultimate** | **0.6291 ★** | 0.4748 | **0.5433 ★** | 0.6349 | 4,611명 |

> ★ = 해당 지표 최우수 모델

### 핵심 인사이트

**① ML 대비 GNN의 질적 우위**
- GraphSAGE ADAN-Ultimate: AUPRC **0.6291** (베이스라인 대비 **+14.5%**)
- ECMP(NNConv+MAX): Precision **0.5199** (ML 대비 **3.3배** 향상) → 2건 중 1건이 실제 범죄자

**② ML의 양적 검거력**
- Top-5000 기준 ML 앙상블(4,645명) > GNN(4,611명)
- ML은 거액·고빈도 거래와 같은 통계적 이상치(Heavy Hitter) 포착에 강점

**③ 투트랙 운용 전략**

| 목적 | 권장 모델 | 평가 단위 | 강점 |
|------|-----------|-----------|------|
| 일별 고위험 계좌 대량 선별 | ML 앙상블 | 고유 노드(계좌) 단위 | Top-K 검거량, 통계 이상치 |
| 1시간 단위 정밀 행동 감시 | GNN (ADAN / ECMP) | 계좌 × 1시간 단위 | 관계망 탐지, AUPRC / F1 |

---

## 🔬 GNN 레이어 비교 실험 (v7c.19)

4종 레이어, 10개 config를 탐색하여 AML 도메인에 최적인 집계 방식을 실험적으로 검증했습니다.

| 레이어 | 집계 방식 | AML 적합성 |
|--------|-----------|------------|
| TransformerConv | Attention-weighted SUM | 허브 노드에서 신호 희석 가능 |
| GATv2Conv | 동적 어텐션 SUM | TransformerConv 개선, 집계 방식 동일 |
| **NNConv + MAX** | **엣지 변환행렬 × MAX** | **희소 악성 신호 보존 → AML 최적** |
| GINEConv | WL 이론적 최강 SUM | 구조 구별력 최강, 스케일 정규화 필요 |

**MAX 집계가 AML에 유리한 이유**: 수천 건의 정상 거래 속에 단 한 건의 악성 거래가 섞여 있을 때, MEAN/SUM은 이 신호를 희석시키지만 MAX는 가장 강한 신호를 그대로 통과시킵니다.

---

## 🛠️ ADAN (Error-Driven Architecture)

오탐(FP)·미탐(FN) 데이터를 분석하여 사각지대를 직접 타격하는 수동 설계 레이어입니다.

### Layer 1: ADAN v1 (관계 필터링)

| 모듈 | 목적 | 데이터 근거 |
|------|------|------------|
| **EIA** (Edge-Inverse Attention) | 소액 쪼개기(Structuring) 포착 | FN 평균 송금액 826만원 (TP 대비 1/42.5) |
| **REG** (Relational Entropy Gate) | 기업형 ACH 오탐 구제 | FP의 ACH 입금액 정상인 대비 54배 |
| **FSR** (Flow-Symmetry Residual) | 위장 인맥 속 세탁범 검거 | FN 군집 위험도 오히려 정상인보다 낮음 |

### Layer 2: ADAN-X (행동 저격)

| 모듈 | 목적 | 데이터 근거 |
|------|------|------------|
| **TGA** (Temporal Gating Attention) | 자동화 세탁 봇 검거 | ACH에서 FN 1,023건 집중 |
| **RGF** (Retention Gated Filter) | 우량 기업 오탐 구제 | FP 자금 잔존율 TP 대비 1.5배 |
| **ISO-Skip** (Isolated Node Recovery) | 비트코인 일회성 계좌 추적 | Bitcoin Recall 4.8% (전체 최하위) |

---

## 📁 프로젝트 구조

```
AML_project_bytracer/
│
├── 00_data_pipeline/
│   └── 00_merge_files.ipynb                       # 거래·계좌 데이터 병합 및 Parquet 생성
│
├── 01_ml_baseline/
│   ├── 01_first_baseline_features.ipynb           # V1 피처 설계 (기본 통계)
│   ├── 02_first_model_v1.ipynb                    # V1 XGBoost 베이스라인
│   ├── 03_second_features_v2.ipynb                # V2 피처 설계 (고도화 통계)
│   └── 04_second_model_v2_no_graph.ipynb          # V2 XGBoost 튜닝
│
├── 02_ml_graph_ensemble/
│   ├── 05_graph_feature_v3.ipynb                  # V3 그래프 피처 설계
│   ├── 06_graph_feature_v3_advanced.ipynb         # V3 그래프 피처 고도화
│   ├── 07_graph_model_v3.ipynb                    # V3 XGBoost (그래프 피처 결합)
│   ├── 08_topk_ensemble_v2v3.ipynb                # V2+V3 Max 앙상블
│   └── 09_error_analysis.ipynb                    # 오탐·미탐 분석 및 인사이트
│
├── 03_gnn_graphsage/
│   ├── 10_graphsage_baseline.ipynb                # GraphSAGE MEAN 베이스라인
│   ├── 11_graphsage_aggr_type_comparison.ipynb    # 집계 방식 비교 (MEAN/MAX/LSTM)
│   ├── 12_graphsage_ver2_adan_v1.ipynb            # ADAN Layer 1 적용 (EIA·REG·FSR)
│   └── 13_graphsage_advanced_adan_ultimate.ipynb  # ADAN Layer 2 적용 (TGA·RGF·ISO-Skip)
│
├── 04_gnn_tgat/
│   └── 14_tgat_transformerconv_nnconv_max.ipynb   # TGAT: TransformerConv·NNConv+MAX 비교
│
└── README.md
```

---

## ⚙️ 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| 데이터 처리 | `Polars`, `Pandas`, `NumPy` |
| 그래프 분석 | `PyTorch Geometric`, `NetworKit`, `cuGraph` |
| GNN 레이어 | `TransformerConv`, `GATv2Conv`, `GINEConv`, 커스텀 NNConv+MAX |
| 분류기 | `XGBoost` |
| 하드웨어 | NVIDIA A100 GPU |

---

## 📦 데이터셋

**IBM Transactions for Anti-Money Laundering (HI-Medium)**
- 계좌 수: 2,076,999개 (노드)
- 거래 수: 약 2,400만 건 (엣지)
- 자금세탁 비율: 약 0.98% (극심한 불균형)
- 분할: Train / Val / Test = 6 : 2 : 2 (시계열 OOT Split)

---

## 🚀 실행 방법

### 환경 설치

```bash
pip install polars torch torch_geometric xgboost scikit-learn tqdm psutil networkit
```

### 실행 순서

실험은 아래 순서대로 진행하는 것을 권장합니다.

```bash
# Step 0. 데이터 병합
jupyter notebook 00_data_pipeline/00_merge_files.ipynb

# Step 1. ML 베이스라인
jupyter notebook 01_ml_baseline/01_first_baseline_features.ipynb
jupyter notebook 01_ml_baseline/02_first_model_v1.ipynb
jupyter notebook 01_ml_baseline/03_second_features_v2.ipynb
jupyter notebook 01_ml_baseline/04_second_model_v2_no_graph.ipynb

# Step 2. ML 그래프 앙상블
jupyter notebook 02_ml_graph_ensemble/05_graph_feature_v3.ipynb
jupyter notebook 02_ml_graph_ensemble/06_graph_feature_v3_advanced.ipynb
jupyter notebook 02_ml_graph_ensemble/07_graph_model_v3.ipynb
jupyter notebook 02_ml_graph_ensemble/08_topk_ensemble_v2v3.ipynb
jupyter notebook 02_ml_graph_ensemble/09_error_analysis.ipynb

# Step 3. GNN — GraphSAGE
jupyter notebook 03_gnn_graphsage/10_graphsage_baseline.ipynb
jupyter notebook 03_gnn_graphsage/11_graphsage_aggr_type_comparison.ipynb
jupyter notebook 03_gnn_graphsage/12_graphsage_ver2_adan_v1.ipynb
jupyter notebook 03_gnn_graphsage/13_graphsage_advanced_adan_ultimate.ipynb

# Step 4. GNN — TGAT (TransformerConv / NNConv+MAX 레이어 비교)
jupyter notebook 04_gnn_tgat/14_tgat_transformerconv_nnconv_max.ipynb
```

> **Note**: GNN 학습은 NVIDIA GPU 환경을 권장합니다. (실험 환경: A100 80GB)

---

## 💡 주요 기여

- **MAX 집계의 AML 유효성 실증**: SUM/MEAN 대비 통계적 우월성 입증 (파라미터 61% 절감 병행)
- **에러 기반 아키텍처 설계(ADAN)**: 오탐·미탐 데이터 분석 → 수동 레이어로 사각지대 타격
- **투트랙 운용 전략 제안**: ML(일별 대량 선별) + GNN(1시간 단위 정밀 감시) 이원화
- **Forward Selection 피처 선택**: 상관 피처 간 importance 분산 문제를 해결하는 탐욕적 선택 전략

---

## 📚 References

- Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS 2017. — GraphSAGE
- Xu, D. et al. (2020). *Inductive Representation Learning on Temporal Graphs.* ICLR 2020. — TGAT
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
