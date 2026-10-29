# JungGM 대회 제출 모델

대회 제공 예보만 사용해 발전량을 예측하고, 고정된 두 VPP 그룹의 `SMP×k×합산 절대오차`를 줄이는 모델이다. 논문에서 가져온 다중 시간 규모 특징·트리 부스팅·잔차 상쇄 원리를 실제 시간순 검증에 연결한다. 최종 채택 모델과 검증 수치는 [RESULTS.md](RESULTS.md)에 기록한다.

## 구성

| 파일 | 역할 |
|---|---|
| `features.py` | A/B 제공사 10·17시 예보, 예보 수정량·시간 통계·태양기하·메타정보의 125개 특징 |
| `pipeline.py` | 후보 학습 → 계절 검증 → 앙상블 선택 → 그룹 탐색·검증 → 최종 재학습·CSV 생성 |
| `grouping.py` | 양쪽 20MW 제약을 지키는 이동·교환 탐색과 후속 기간 채택 판단 |
| `predict.py` | 저장한 최종 모델을 불러와 재학습 없이 예측 재생성 |
| `resume_production.py` | 선택 완료 후 중단된 최종 학습을 원본·코드 해시 확인 후 재개 |
| `validate_submission.py` | 두 제출 CSV의 행·시간·값·용량 제약 검사 |
| `audit_run.py` | 원본 정답으로 정산 독립 재계산, 원본 해시와 최종 합산 일치 검사 |
| `PAPER_METHOD.md`, `RULES.md` | 논문 적용 범위와 규정·시간 경계 |

`_local`에는 특징 캐시·학습 모델·실험 기록을, `제출`에는 실제 제출 기간의 `composition.csv`, `forecast.csv`, `검증결과.json`을 둔다. 원본 데이터·학습 모델·CSV는 공개 저장소에서 제외하며 코드와 설명만 공유한다. 원본 배포 폴더는 이동하거나 수정하지 않는다.

## 실행

Python 3.14에서 확인한 패키지 버전은 `requirements.txt`에 고정했다. 다른 Python 버전에서는 해당 버전의 설치 가능 여부를 확인한다.

```powershell
cd <저장소>/JungGM
py -m pip install -r requirements.txt
py -m unittest discover -v
py -u -X utf8 pipeline.py --data "대회 배포 폴더" --run "_local/새로운실험" --submission "제출"
```

출력 폴더를 새로 지정하여 기존 실행 기록과 제출 CSV를 덮어쓰지 않는다. 기본 캐시 `_local/cache`는 원본·특징 코드가 일치할 때 재사용한다. 전체 학습은 CPU와 메모리를 사용하며 인터넷에 접속하지 않는다.

저장 모델의 예측만 다시 만들거나 독립 검사하려면:

```powershell
py predict.py --data "대회 배포 폴더" --run "_local/완료한실험" --out "_local/재생성"
py validate_submission.py --sites "대회 배포 폴더/sites.csv" --composition "제출/composition.csv" --forecast "제출/forecast.csv"
py audit_run.py --data "대회 배포 폴더" --run "_local/완료한실험" --submission "제출"
py verify_reproduction.py --run "_local/완료한실험" --submission "제출" --reproduction "_local/재생성"
py report_results.py --run "_local/완료한실험" --submission "제출"
```

모델·그룹 선택이 저장된 후 최종 학습이 중단되었다면 다음 명령으로 이어갈 수 있다. 완료된 모델은 설정·특징 순서를 검증해 재사용하고, 저장되지 않은 모델만 다시 학습한다. 선택 비율·그룹을 다시 탐색하지 않으며 완료된 제출 파일은 덮어쓰지 않는다.

```powershell
py -u -X utf8 resume_production.py --data "대회 배포 폴더" --run "_local/중단된실험" --submission "제출"
```

## 선택 기준

첫 순방향 모델은 2024-06-30 17시까지 학습하고 7~12월의 고정 관측 코호트에서 비교한다. 이 부분 관측 대리 점수에 50%, 2024-12-31 17시까지 학습한 모델의 2025년 1~2월 전체 253개 점수에 50%를 둔다. 각 블록 안에서는 월별 기준 모델 대비 벌금 비율을 같은 비중으로 반영한다. 7개 후보와 최대 3개 모델의 비음수 앙상블을 비교한다.

선택한 방식으로 2025년 3월 잔차를 만들고, 용량 제약 아래 그룹 후보 하나를 탐색한다. 4월에서 1% 이상 개선되고 일별 쌍대 부트스트랩 하한이 양수인 경우에만 채택한다. 그렇지 않으면 참조 배정을 유지한다. 5월 결과는 해당 실행의 선택을 고정한 후 보고한다.

2024·2025년은 이전에 확인한 개발 자료다. 이번 결과를 연구 전체의 미사용 최종 시험이나 공식 점수로 부르지 않는다. 마지막에는 **2025-05-31 17:00 KST까지** 재학습하고 **2025-06-01 01:00~2026-06-01 00:00 KST**의 17,520행을 만든다. 실제 평가기간 정답이 없어 그 기간의 정확도는 로컬에서 확인할 수 없다.

자동 서버 제출 기능은 없다. 형식 통과와 모델 성능·공식 승인 여부는 별개다.
