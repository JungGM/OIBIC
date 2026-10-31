# OIBC 2026 최종 모델 — 2026-10-31

[최종 코드와 실행법](outputs/suwan_project_final/README.md) · [최종 결과](outputs/suwan_project_final/RESULTS.md)

사용자 선택으로 **143특징 XGBoost + 기존 LightGBM 앙상블 + 새 배치 + 작은 배율 보정** 후보를 최종 채택했습니다. 2025년 5월 과거 평가 패널티는 **21,393,220.37원**으로, 같은 로컬 환경의 JungGM 공개 비율 대비 **2.04%**, 원저자 보고값 대비 **0.62%** 감소했습니다. 공식 평가기간의 점수는 아닙니다.

원래 안정성 게이트는 실패했고 사용자가 결과를 확인한 뒤 채택했습니다. 그 기록을 보존했으며 미사용 시험 성능이라고 주장하지 않습니다. 제출용 CSV는 로컬에서 생성·형식 검증했으며 공개 저장소에는 원본 데이터·대용량 모델·제출 CSV를 올리지 않았습니다.

---

## 이전 프로젝트 기록

# suwan_projectV1

OIBC 2026 태양광 발전량 예측 및 두 집합 구성 최적화의 첫 구현이다. **실험 중단 상태이며 제출용 완성본이 아니다.** 개선 성능은 아직 확인하지 않았다. 상세 진행 상태는 [STATUS.md](STATUS.md)를 참조한다.

## 구성

- `pipeline.py`: 동일 학습 표본 기준 모델 2개와 통합 예보 특징 모델 비교, 잔차 기반 구성 탐색, 기간별 평가.
- `audit_data.py`: 원본 중복·결측·관측 범위 점검. 발전량 미관측을 0으로 채우지 않는다.
- `baseline_original.py`: 제공 데이터 ZIP의 원본 기준선 코드. pipeline에서 특징과 태양 위치 계산을 가져온다. 이 폴더에서 단독 실행하도록 수정한 파일은 아니다.
- `test_pipeline.py`: 원본 특징 일치, 대상일 경계와 통계, 그룹 간 상쇄 금지, 작은 최적화 문제 검증.
- `verify_report.py`: 전체 실험 완료 후 저장 결과를 별도 집계로 검사하고 RESULTS.md 생성. 미완료 상태에서는 실행할 수 없다.
- `METHOD.md`: 논문에서 가져온 아이디어와 자체 설계의 구분.

## 방법

기준 방식은 24개 특징으로 10시·17시 예보 모델을 따로 학습한 뒤 평균한다. 개선 후보는 두 시각 예보, 수정량, 공급사 차이, 구름/UV의 같은 대상일 이동 통계 등을 86개 특징으로 통합한다. 발전소 용량 대비 출력을 LightGBM으로 예측한다.

그룹 구성은 `sum(SMP*k*(abs(G1 잔차 합)+abs(G2 잔차 합)))`을 줄이는 이동·교환 탐색이다. 두 그룹 모두 20MW 이상을 유지한다. 1~2월 잔차로 구성을 선택하고 3~5월에서 고정 구성의 성능을 평가한다. 전역 최적해는 보장하지 않는다.

현재 학습은 제곱오차 기반이고 대회 평가는 그룹 가중 절대오차이므로 목표 사이에 차이가 있다. 절대오차 학습, 그룹 합계 보정, 구성의 기간별 안정성이 다음 비교 대상이다.

## 로컬 실행

Python 3.12 환경에서:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

대회에서 제공받은 참가자 데이터 폴더를 직접 `data/`에 배치한다:

```text
data/
  baseline.py
  README.md
  sites.csv
  smp.csv
  train/
    generation.parquet
    forecast_wa.parquet
    forecast_wb.parquet
    smp.csv
  test/
    forecast_wa.parquet
    forecast_wb.parquet
```

그 뒤 이 폴더에서:

```powershell
python test_pipeline.py
python audit_data.py
python pipeline.py --out runs/pilot_v2
```

또는 `./run.ps1`을 실행한다. 학습/예측에 시간이 걸린다. 기본 학습량 10만 주간 행, 모델당 600개 트리, CPU 2스레드다. 설정을 바꾸면 새 `--out` 디렉터리를 사용한다. 코드/설정/입력 지문이 일치하는 예측 캐시만 재사용한다.

전체 실행이 성공한 뒤에만 다음을 실행한다:

```powershell
python verify_report.py
```

원본 데이터, PDF, 모델 가중치, 발전소별 예측/잔차, 개인 PC 경로와 인증 정보는 업로드하지 않았다. `data/`와 `runs/`는 Git 추적에서 제외한다. 자동으로 대회에 제출하는 기능은 없다.