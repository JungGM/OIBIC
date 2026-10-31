# OIBC 최종 실험 패키지

JungGM 기준 모델에 집합 잔차 보정과 예보 변동 특징을 추가하고 시간 순 검증으로 비교하는 프로젝트다. 추가 후보는 원래 안정성 게이트에는 미달했지만, 사용자가 확인된 점수 개선을 근거로 최종 채택했다. 이 결정은 ADOPTION.json, 실제 결과는 RESULTS.md에 기록한다. 실패한 안정성 검사를 통과한 것으로 바꾸지 않는다.

## 실행 환경과 데이터

Python3.12에서 requirements.txt의 버전을 사용한다. 원본 대회 데이터는 공개 저장소에 포함하지 않는다. 저장소 루트의 work/oibc_data 안에 sites.csv, smp.csv, train/, test/, baseline.py, README.md가 위치하도록 제공 ZIP을 풀어 둔다. 기존 work/cache 폴더를 다른 버전 실행에 섞지 않는다.

```powershell
python -m pip install -r outputs/suwan_project_final/requirements.txt
python outputs/suwan_project_final/reproduce_all.py
```

처음 실행하면 bootstrap.py가 고정된 JungGM 커밋의 원본 소스를 내려받아 Git blob 해시를 검증한다. 인터넷은 이 다운로드에만 필요하다. 이후 학습·예측은 로컬 제공 데이터만 사용한다. CPU2스레드, 수시간이 걸릴 수 있으며 모델 체크포인트를 보존한다. 다른 장비에서 비트 단위로 같은 모델을 학습한다는 보장은 없다.

기존 기준 실행이 있는 현재 작업공간에서는 새 bootstrap을 실행하지 않는다. 새 코드/설정으로 다시 연구하려면 기존 학습 기록을 덮어쓰지 말고 별도 작업공간을 사용한다.

## 파일

- run.py: 단계 실행, 시간 외 잔차 구성, 특징 학습, 후보 선택, 배치, 채택 게이트, 최종 파일.
- correction.py: 규제 중앙값 회귀·단순 배율·무보정 대조군과 직렬화된 계수 추론.
- extras.py: 같은 발표분·같은 목표일 내 예보 변동 특징18개.
- PROTOCOL.json: 후보 범위·선택 기간·채택 조건을 사전에 고정한 명세.
- ADOPTION.json, adopt_candidate.py: 실패한 안정성 게이트 기록을 유지하면서 사용자가 선택한 고정 후보를 최종 재학습·출력. 승인된 후보 해시와 다르면 중단한다.
- verify_final.py: 저장 모델/예측 재현, 독립 손실, 날짜 격자·야간·원본 보존 검증.
- test_final.py: 날짜·정답 격리, 결측 간격, 보정 범위와 추론의 실측 독립성 테스트.
- bootstrap.py, baseline_runner.py, UPSTREAM.json: 출처 있는 기준 소스의 설치와 재개 가능한 실행.
- results/: 집계 결과 및 채택/탈락 기록. submission/: 로컬 최종 CSV.

최종 파일은 outputs/suwan_project_final/submission/composition.csv와 forecast.csv다. 공식 서버에 자동 제출하지 않는다. 원본 데이터·대용량 트리 모델·행별 중간 예측은 work/에 보관하며 GitHub에 올리지 않는다. 작은 보정 계수와 집계 실험 기록은 results/에 포함한다. 최종 제출 CSV도 이 공개 코드 패키지에서는 제외하고 로컬에서 제공한다.

## 해석

3월 내부 선택 →4월 후보 하나의 채택 판단 →5월 보고 순서로 실험했다. 이후 사용자가5월 결과도 확인하고 후보 채택을 지시했다. 계수나 배치를 다시 튜닝하지 않았으나 최종 채택 결정은5월 결과를 본 뒤 이루어졌으므로 미사용 외부 시험이라고 부르지 않는다. 자세한 식과 논문 활용 범위는 METHOD.md, 실제 결과는 RESULTS.md를 확인한다.
