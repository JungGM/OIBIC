"""Render the completed fixed-group stage 1 comparison without choosing new weights."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
status = json.loads((HERE/'status.json').read_text())
assert status['complete']
rows = list(csv.DictReader((HERE/'development_comparison.csv').open()))
table = {(r['month'], r['model']): float(r['penalty_krw']) for r in rows}
upstream = {'2025-03':12972597.63, '2025-04':17449032.11, '2025-05':21525804.53}
lines = [
 '# 최종 1단계: 기준 모델 재현·비교 결과', '',
 '상태: 기준 비교 완료. 이 문서는1단계 당시의 비교 기록이다. 이후 개선 실험, 공식 기간 생산·검증 및 최종 사용자 채택 결과는 ../suwan_project_final/RESULTS.md를 참조한다.', '',
 '결론: 로컬 선택 비율은 공개 비율보다 일관되게 우수하지 않다. 4월은 개선했지만3·5월은 악화했다. 독자 개선을 달성했다고 주장할 수 없다. 두 비교 기준을 고정하고 2단계의 개선 효과를 각각 비교한다. 5월을 보고 비율을 다시 선택하지 않았다.', '',
 '## 동일 로컬 환경·고정 배치 비교', '',
 'public_fixed는 JungGM 공개 비율 0.2/0.3/0.5, local_selected는 선택 구간에서 고른 0.2/0.4/0.4다. 순서는 기준 LightGBM/L1 LightGBM/XGBoost다. 같은 저장 모델·253곳·원본 정답·가격·시각·참조 배치를 사용한다. 금액은 정산 패널티(원), 낮을수록 좋다.', '',
 '| 월 | 시간 | 기준 LightGBM | 공개 비율 | 로컬 선택 비율 | 로컬의 공개 비율 대비 감소율 |',
 '|---|---:|---:|---:|---:|---:|'
]
for m in upstream:
    b,u,l = [table[m,n] for n in ['official_lgb','public_fixed','local_selected']]
    hours = next(r['hours'] for r in rows if r['month']==m)
    lines.append(f'| {m} | {hours} | {b:,.2f} | {u:,.2f} | {l:,.2f} | {(1-l/u)*100:+.3f}% |')
lines += ['', '양수는 로컬 개선, 음수는 악화다. 3~5월의 마지막 날은17시까지만 포함한다.', '',
 '## 원저자 보고 수치와의 차이', '',
 '| 월 | JungGM 보고 | 공개 비율의 로컬 재실행 | 로컬 재실행 증가율 |',
 '|---|---:|---:|---:|']
for m,u in upstream.items():
    l=table[m,'public_fixed']
    lines.append(f'| {m} | {u:,.2f} | {l:,.2f} | {(l/u-1)*100:+.3f}% |')
lines += ['',
 '기준 LightGBM의 월별 수치는 원저자 표와 소수 둘째 자리까지 일치하지만 강화 모델의 앙상블은 정확히 일치하지 않는다. 공개 비율 자체를 같은 로컬 모델에 적용해도 차이가 남으므로 가중치 선택만의 문제는 아니다.', '',
 '원본 Python과 비교한 의미 있는 실행 차이는 CPU스레드6→2이며 폴더 재개 허용도 추가했다. NumPy2.5.2→2.5.4, pandas3.0.5→3.0.1 차이도 있다. 어느 차이가 수치 차이의 원인인지 분리 실험으로 확정하지 않았다. 원저자 저장 모델/행별 검증 예측과 직접 대조하지 않았으므로 정확한 수치 재현 완료라고 표현하지 않는다. source_diff.json과 selection_check.json에 근거를 기록했다.', '',
 '따라서 향후 개선은 (1) 같은 로컬 환경에서 공개 비율 대비 개선, (2) 원저자 보고 수치보다 낮은지 여부를 분리해 보고한다. 1% 내외의 작은 차이를 곧바로 팀원 모델보다 우수하다고 주장하지 않는다.', '',
 '## 검증', '',
 '- 기존 계약 테스트18개 통과: 발표시각 경계·변수 단위·입력 격리·집합 절대오차·결측 및 배치 제약·제출 형식.',
 '- 원본10개 파일과 실행 당시 Python 소스 해시 일치.',
 '- 선택14개 및 개발3개 모델 체크포인트 해시/정답 cutoff 확인.',
 '- pandas 집합 합산 기반 독립 채점으로 선택 지표0.9442036857758562 재현.',
 '- 개발 정답과 원본 generation.parquet의 값·행 수 일치.',
 '- 저장 L1/XGBoost 모델을 다시 로드한 예측과 기존 앙상블 일치. 기준 LightGBM은 저장 개발 예측을 공통 사용했다.',
 f"- 앙상블 재조합 최대 절대차: {status['saved_model_prediction_max_abs_difference']:.12g} kWh.",
 '- 개발 월별 점수와 기존 실행 점수도 독립 채점에서 일치.', '',
 '## 기준 배치와 다음 단계', '',
 '재현의 배치 후보는4월0.9218% 개선으로1% 기준에 미달했고 일별 부트스트랩 하한도 음수여서 채택되지 않았다. 이는 기존 배치가 최적이라는 뜻이 아니다. 이번 고정 배치 비교에는 배치 효과가 섞이지 않는다.', '',
 '다음은 작은 집합 잔차 보정이다. 무보정과 단순 배율을 대조군으로 두고 시간 외 잔차만 학습한다. 이번 비교에서4·5월을 다시 확인했으므로 이후 같은 기간을 미사용 시험이라고 부르지 않는다. 조건부 보정은 아직 실행하지 않았다.', '',
 '## 재실행', '',
 '프로젝트 루트에서 로컬 의존성 work/python_deps를 PYTHONPATH로 지정해 compare.py, report.py 순서로 실행한다. compare.py는 학습 없이 저장된 선택 예측과 개발 모델을 사용한다. 데이터/모델은 work에 있고 이 폴더에는 비교 코드와 요약만 저장한다.', '',
 'audit_run.py는 후속 공식 기간 감사용 원본 복사본이며 재개 체크포인트 JSON을 모델 메타데이터 목록에서 제외하는 호환 수정만 적용했다. 학습 Python 파일은 수정하지 않았다.',
]
(HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:18]))
