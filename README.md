# PrompTune | AI 기반 한국어 프롬프트 개선 서비스

> **KcELECTRA 기반 프롬프트 진단 · HyperCLOVA X 기반 생성 파이프라인 · BGE-M3/pgvector 문서 검색**
>
> 2026.08.12 ~ 2026.09.16 | 5인 팀 프로젝트 | **AI / ML · FastAPI 및 RAG 평가 기여**

사용자의 간단하거나 불완전한 업무 지시문에서 **보완이 필요한 8개 요소**를 분석하고, 필요한 정보를 되물어 더 구체적인 프롬프트 및 결과물을 생성하도록 설계한 서비스입니다. 내부 업무 문서가 필요한 요청에는 검색 기반 근거를 연결합니다.

**이 저장소는 팀 프로젝트 [`bxbae/promptune`](https://github.com/bxbae/promptune)의 개인 Fork입니다.** 아래에서는 팀 전체 기능과 개인 기여를 구분해 소개합니다.

## 핵심 성과

| 구분 | 검증 내용 | 결과 |
| --- | --- | --- |
| **RAG 검색 정확도** | 자체 구성한 내부문서 검색 질문 54개 | **Top-1 96.30% (52/54), Top-3 100% (54/54)** |
| **RAG 인덱싱** | 업무 규정 테스트 문서 18개 → Chunk 36개 | **BGE-M3 Dense Embedding 1,024차원** |
| **검색 경로 분류** | 내부검색 외 요청 6개 | **6/6 PASS** (소규모 규칙 기반 테스트) |
| **문서 접근 격리** | `owner_user_id` 기준 사용자별 검색 범위 | **격리 테스트 PASS** |
| **모델 오류 분석** | KcELECTRA 8요소 분류 실험 | **3개 시드의 FP/FN 423건 분석** |

> **평가 범위:** RAG 수치는 테스트용 문서 18개와 자체 작성한 검색 질문 54개에 대한 **정답 문서 검색 적중률**입니다. 생성 답변의 정확도나 일반적인 검색 성능을 의미하지 않습니다. 검색 경로 분류 6건도 제한된 테스트입니다.

## 1. 프로젝트 개요

PrompTune은 프롬프트의 **Task · Audience · Context · Format · Tone · Length · Constraint · Example**을 분석합니다.

- **진단:** KcELECTRA 다중 레이블 분류로 보완이 필요한 항목 탐지
- **개선:** 사용자 확인 및 HyperCLOVA X를 이용한 문구·답변 생성 흐름
- **근거 검색:** 필요할 때 BGE-M3 임베딩과 PostgreSQL/pgvector로 관련 내부문서 검색
- **검증:** 규칙 기반 안전검사와 최종 결과 검증 단계를 분리

![PrompTune 파이프라인 구조도](docs/flowchart.png)

## 2. 담당 역할 및 기여

**주 담당: AI/ML 및 FastAPI 진단 파이프라인**

| 영역 | 담당 업무 |
| --- | --- |
| **KcELECTRA** | 8요소 다중 레이블 분류 모델 파인튜닝·추론·평가, 라벨별 임계값 적용 |
| **실험 및 오류 분석** | 학습/검증 데이터 기반 평가, 3개 시드의 FP/FN 423건 분석, 희소 라벨 데이터 60건 보강 |
| **서비스 연동** | FastAPI `/api/ai/diagnose`에 실제 모델 추론 연결, Mock/Real 전환 및 입출력 스키마 유지 |
| **RAG 공동 작업** | BGE-M3/pgvector 검색 구현·통합 과정 참여 및 검색 성능 평가·실패 사례 검증 |
| **검증 설계** | 검색 정확도뿐 아니라 검색 경로, 사용자별 접근 격리, API E2E를 분리해 확인 |

> **협업 범위:** BGE-M3/pgvector 검색은 팀 내 Retrieval 담당자와 협업했습니다. RAG 성과는 **팀 전체의 검색 파이프라인 결과**이며, 개인 기여는 검색 구현 참여·통합·평가·오류 분석 중심으로 구분했습니다. 팀별 역할은 [`docs/ROLES.md`](docs/ROLES.md)를 참고하세요.

## 3. KcELECTRA — 8요소 멀티레이블 분류

### 문제와 접근

한 문장에서 여러 정보가 동시에 부족할 수 있으므로 단일 클래스 분류가 아닌 **8개 독립 라벨의 멀티레이블 분류**로 정의했습니다. 사용자가 놓친 정보를 다시 질문할지 결정해야 하므로, **미탐(False Negative)** 을 중요하게 보고 Recall·F2 및 Precision을 함께 검토했습니다.

- **학습 및 검증:** 학습 2,000건 / 검증 400건을 사용한 실험 수행
- **오류 분석:** 3개 시드에서 FP/FN 총 423건을 분류해 혼동 패턴 파악
- **개선 시도:** 라벨별 threshold 조정 및 희소 라벨 데이터 60건 보강
- **서빙:** Sigmoid 확률에 라벨별 threshold 적용 후 0/1 판정, 모델·토크나이저 초기 1회 로딩
- **실제 연결:** `USE_REAL_DIAGNOSIS` 설정으로 Mock/Real 진단 경로 전환

**핵심 트러블슈팅:** 전체 라벨에 동일 임계값을 적용하면 요소별 정밀도·재현율의 균형을 조정하기 어렵습니다. 라벨별 threshold를 분리하고, FP/FN 사례를 분석해 희소 라벨과 복합적인 표현을 우선 개선 대상으로 삼았습니다.

**관련 코드:** [`diagnose_real.py`](ai-service/app/services/diagnose_real.py) · [`model_info.json`](ai-service/models/prompt-diagnosis/model_info.json) · [`모델 검증 기준`](ml/ko_vs_kc/K모델_검증기준.md)

## 4. RAG — BGE-M3 + PostgreSQL/pgvector

### 검색 흐름

1. 내부 문서 **18개**를 문장 경계 중심으로 **36개 Chunk**로 분할
2. 사전학습된 **BGE-M3 Dense Embedding (1,024차원)** 으로 벡터화
3. `documents` / `document_chunks` 테이블에 내용·임베딩 저장
4. 사용자 질문을 임베딩해 **pgvector Cosine Similarity** 기반 Top-K 검색
5. SQL의 `owner_user_id` 조건으로 **다른 사용자의 문서를 검색 대상에서 제외**
6. FastAPI 검색 응답으로 관련 문서 후보 전달

**BGE-M3는 별도 파인튜닝하지 않았으며**, Dense Retrieval만 사용했습니다(Sparse·ColBERT 미적용).

### 검색 평가

| 지표 | 결과 |
| --- | ---: |
| 평가 문서 / Chunk | 18개 / 36개 |
| 검색 질문 | 54개 |
| **Top-1 정답 문서 적중** | **52/54 (96.30%)** |
| **Top-3 정답 문서 포함** | **54/54 (100%)** |
| 검색 라우팅 확인 | 6/6 PASS |
| NumPy ↔ pgvector 비교 | 대표 질문 4개에서 검색 순위·점수 일치 확인 |
| 사용자 격리 / FastAPI E2E | PASS / PASS |

### 실패 사례와 개선 방향

- **Q027:** 개인카드로 구매한 장비의 비용 처리 질문에서 `비품 구매 규정` 대신 `법인카드 지침`이 Top-1에 노출됐습니다.
- **Q053:** 근속연수와 승진 관련 질문에서 `성과평가·승진 기준` 대신 `근태 관리 규정`이 Top-1에 노출됐습니다.

두 경우 모두 **정답 문서는 Top-3 안에 포함**됐습니다. 업무 규정 간 의미가 겹치는 문제로 해석했고, 후속 개선 과제로 **Hard Negative 평가셋 확장, Reranker, Dense+Sparse Hybrid Retrieval**을 정리했습니다. 이러한 개선은 **향후 과제이며 적용 완료 성과가 아닙니다.**

**검증 자료:** [`RAG 구현·평가 보고서`](docs/PrompTune_BGE-M3_RAG.md) · [`평가 결과 JSON`](ai-service/app/data/rag/rag_pgvector_eval_results.json) · [`평가 스크립트`](ai-service/app/services/retrieval/evaluate_internal_rag.py)

## 5. 시스템 구성

| 계층 | 기술 | 역할 |
| --- | --- | --- |
| Frontend | Next.js · React · TypeScript | 프롬프트 입력·인라인 보완 UI |
| Backend | Spring Boot · JPA | 인증·오케스트레이션·업무 데이터 |
| AI Service | **Python · FastAPI · PyTorch · Transformers** | **프롬프트 진단·생성·검증 API** |
| Retrieval | **BGE-M3 · PostgreSQL · pgvector** | **문서 임베딩·유사도 검색** |
| Infra | Docker Compose · GitHub Actions | 서비스 통합 및 빌드·테스트 |

전체 흐름은 [`16단계 파이프라인`](docs/PIPELINE.md)을 참고하세요.

## 6. 실행 및 재현 자료

- 로컬 실행 및 환경 설정: [`프로젝트 공통 실행 가이드`](ai-service/README.md) 및 [`Docker 설정`](docker-compose.yml)
- 진단 API: `POST /api/ai/diagnose`
- 검색 API: `POST /api/ai/retrieve`
- 학습·비교 실험 가이드: [`KoELECTRA vs KcELECTRA`](ml/ko_vs_kc/README.md)

> 실제 모델 추론과 검색 E2E를 재현하려면 **별도 모델 가중치, 환경변수, PostgreSQL 데이터 및 임베딩**이 필요할 수 있습니다. 저장소를 복제하는 것만으로 동일한 평가 결과가 자동 재현된다고 보장하지 않습니다.

## 7. 회고

이번 프로젝트에서 가장 중요하게 배운 점은 **모델 학습 결과만으로 서비스 품질을 증명할 수 없다는 것**입니다. 분류 모델에는 라벨별 FP/FN 분석과 임계값 검증이 필요했고, RAG에는 Top-K 검색 평가 외에도 **유사 문서 혼동, 사용자 데이터 격리, API End-to-End** 검증이 필요했습니다.

향후에는 실제 사용 데이터와 독립적인 평가셋을 확장하고, 검색 실패 사례를 바탕으로 Reranking과 Hybrid Retrieval을 비교하려고 합니다.

---

**Team Project:** [원본 팀 저장소](https://github.com/bxbae/promptune) · **Portfolio Fork:** [Seungdeuk/promptune](https://github.com/Seungdeuk/promptune)
