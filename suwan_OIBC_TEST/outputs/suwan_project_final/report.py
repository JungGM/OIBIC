"""Publish all fixed candidate outcomes, including failures, as aggregate tables."""
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/'results'/name).read_text(encoding='utf-8'))

def main():
    freeze=read('frozen_before_May.json');complete=read('completion.json')
    verified=read('verification.json') if (HERE/'results/verification.json').exists() else {'passed':False,'pending':True}
    assert complete['complete']
    reports=pd.read_csv(HERE/'results/monthly.csv')
    trial=read('March_candidate_comparisons.json')
    candidate=freeze['candidate']
    overridden=(HERE/'ADOPTION.json').exists()
    selected=candidate if overridden else freeze['selected']
    recipe=selected['recipe']
    model_summary={'ensemble_weights':{'official_lgb':.2,'lgb_l1':.4,'xgb_deep_143' if selected['variant']=='feature' else 'xgb_deep_125':.4},
        'group_site_counts':verified.get('group_site_counts'),
        'group_capacity_kw':verified.get('group_capacity_kw'),
        'group_changes':complete['group_changes'],'correction':recipe,
        'original_stability_gate_passed':freeze['accepted'],'user_adoption':overridden,
        'production_fit_cutoff':'2025-05-31 17:00+09:00'}
    meta=HERE.parents[1]/'work/suwan_final/feature_production/metadata.json'
    if selected['variant']=='feature' and meta.exists():
        saved=json.loads(meta.read_text())
        model_summary.update(feature_count=len(saved['features']),training_rows=saved['training_rows'],
                             feature_model_sha256=saved['model_sha256'])
    (HERE/'results/final_model.json').write_text(json.dumps(model_summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if overridden:
        reports=reports.rename(columns={'selected_penalty':'original_gate_selected_penalty'})
        reports['selected_penalty']=reports.candidate_penalty
        reports.to_csv(HERE/'results/adopted_monthly.csv',index=False)
    baseline_months={'2025-03':12972597.63,'2025-04':17449032.11,'2025-05':21525804.53}
    lines=['# 최종 실행 결과','',
        ('최종 채택: 사용자 명시적 선택으로 추가 후보 채택. 원래 안정성 게이트는 실패했으며 그 기록을 유지한다.' if overridden else f"최종 채택: 추가 후보 {'채택' if freeze['accepted'] else '탈락 — 사전에 고정한 기준 모델 유지'}."),
        f"최종 예측: {selected['variant']}, 보정: {selected['recipe']['kind']}, 참조 배치 변경 수: {complete['group_changes']}개.",'',
        ('공식 기간 실제 발전량은 없어 공식 점수·순위는 확인할 수 없다. 후보 구성과 계수는5월 전에 고정했지만, 사용자의 최종 채택 결정은5월 결과를 확인한 뒤 이루어졌다. 따라서5월은 최종 채택 결정의 독립 시험이 아니다.' if overridden else '공식 기간 실제 발전량은 없어 공식 점수·순위는 확인할 수 없다. 아래는 이미 검토한 과거 자료를 사용한 로컬 실험 결과다.5월 결과로 후보나 계수를 다시 고르지 않았다.'),'',
        '## 단계별 완료 내역','',
        '1. 기준 재현·비교: 독립 채점과 저장 모델 검증 완료. 원저자 강화 모델 수치와 차이는 남아 있다.',
        '2. 집합 보정: 무보정·배율·규제 중앙값 잔차 보정 비교 완료.',
        '3. 특징: 기존125개에 예보 변동18개 추가, 동일 행·설정의 XGBoost 한 개 학습 및 비교 완료.',
        f"4. 직접 집합 예측 대안: {'실행' if candidate['direct_alternative_run'] else '조건부 생략 — 앞 후보가3월 내부 선택에서 개선'}.",
        '5. 배치:3월 앞부분에서 단일 제안 생성, 후보 배치별 보정 재학습 후3월 뒷부분에서 선택 완료.',
        ('6. 최종 생성·검증:253개소 구성과17,520행 예측 생성, 재생성 해시·시간축·야간·원본 보존 검사 완료.' if verified['passed'] else '6. 최종 생성:253개소 구성과17,520행 예측 생성 및 형식·용량·원본 보존 검사 완료. 별도 최종 재생성 검사는 진행 중.'),'',
        '## 3월 내부 검증: 기능별 효과','',
        '| 후보 | 무보정 패널티 | 후보 안에서 선택한 패널티 | 선택 보정 |',
        '|---|---:|---:|---|']
    for t in trial:
        none=next(x['validation_penalty'] for x in t['candidates'] if x['kind']=='none')
        best=min(x['validation_penalty'] for x in t['candidates'])
        lines.append(f"| {t['variant']} | {none:,.2f} | {best:,.2f} | {t['selected']['kind']} |")
    lines += ['', '이 표는3월21일~31일17시 내부 선택 결과이며 독립 성능 보장이 아니다. 후보 전체 기록은 results/March_candidate_comparisons.json에 있다.','',
        '## 4월 단일 후보 채택 판단','',
        f"- 패널티 감소율: {freeze['April_gate']['improvement_fraction']*100:.3f}% (필요1% 이상).",
        f"- 일별 부트스트랩 평균 절감액 하한: {freeze['April_gate']['mean_daily_savings_lower90']:,.2f}원/일 (필요0 초과).",
        f"- 3일 블록 부트스트랩 하한: {freeze['April_3day_bootstrap_lower90_krw_per_day']:,.2f}원/일 (필요0 초과).",
        f"- 원래 게이트 판정: {'채택' if freeze['accepted'] else '탈락'}.",
        ('- 이후 사용자 지시로 고정 후보를 채택했다. ADOPTION.json에 결정과 한계를 기록했다.' if overridden else '- 원래 게이트 판정을 최종 선택에 사용했다.'),'',
        '## 월별 결과 — 낮을수록 좋은 정산 패널티(원)','',
        '| 월 | 로컬 고정 기준 | 추가 후보 | 최종 선택 | 원저자 보고 | 최종의 원저자 대비 감소율 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in reports.itertuples(index=False):
        original=baseline_months[f'2025-{row.month:02d}']
        lines.append(f'| {row.month}월 | {row.baseline_penalty:,.2f} | {row.candidate_penalty:,.2f} | {row.selected_penalty:,.2f} | {original:,.2f} | {(1-row.selected_penalty/original)*100:+.3f}% |')
    may=reports.set_index('month').loc[5]
    gain=100*(1-may.selected_penalty/baseline_months['2025-05'])
    candidate_gain=100*(1-may.candidate_penalty/baseline_months['2025-05'])
    local_reference=HERE.parent/'final_stage1/development_comparison.csv'
    if local_reference.exists():
        comparison=pd.read_csv(local_reference)
        public=comparison[comparison.model=='public_fixed'].set_index('month')
        lines += ['', '### 같은 로컬 환경의 JungGM 공개 비율과 비교', '',
                  '| 월 | 공개 비율·참조배치 | 최종 선택 | 최종 패널티 감소율 |',
                  '|---|---:|---:|---:|']
        for row in reports.itertuples(index=False):
            value=float(public.loc[f'2025-{row.month:02d}','penalty_krw'])
            lines.append(f'| {row.month}월 | {value:,.2f} | {row.selected_penalty:,.2f} | {(1-row.selected_penalty/value)*100:+.3f}% |')
    lines += ['',f"추가 실험 후보의5월 패널티는 원저자 보고 대비 {abs(candidate_gain):.3f}% {'감소' if candidate_gain>0 else '증가'}했다. " + ('후보 구성 자체는 바꾸지 않고 사용자 지시로 최종 채택했다.' if overridden else '후보와 최종 제출 선택을 구별해야 한다. 안정성 게이트가 실패한 뒤5월 성능을 근거로 채택 규칙을 바꾸지 않았다.'), '',
              f"5월 원저자 보고 대비 최종 패널티는 {abs(gain):.3f}% {'감소' if gain>0 else '증가'}했다. " +
              ('이 과거 구간의 결과이며 공식 기간 우위를 의미하지 않는다.' if gain>0 else 'JungGM보다 더 나은 결과라는 목표는 이번 최종판에서 달성하지 못했다.'),'',
        '원저자 보고는 원저자 실행 환경에서 생성한 집계값이다. 같은 로컬 환경의 공개 비율 비교는1단계 보고서에 따로 기록했다. 기준 LightGBM 수치는 일치하지만 강화 모델 수치는 완전히 일치하지 않는다. CPU스레드 및NumPy/pandas 버전 차이를 기록했으며 인과 원인은 확정하지 않았다.','',
        '## 검증·산출물','',
        '- 기준 계약 테스트18개, 추가 날짜·정답 격리/보정 경계 테스트2개 통과.',
        '- 원본 데이터 SHA256 보존, 독립 정산 계산, 기준 저장 모델 예측 재현 통과.',
        ('- 최종 CSV 재생성 후 바이트 해시 일치,253개소·각그룹20MW·8,760시간씩2그룹·야간0 확인.' if verified['passed'] else '- 최종 CSV 형식·행 수·그룹용량 검사 통과. 추가 재생성 해시 및 야간 검사는 아직 완료 확인 전이다.'),
        '- 로컬 제출 파일: submission/composition.csv, submission/forecast.csv.',
        '- 원본 데이터/행별 중간 예측/대용량 트리 모델은 공개 GitHub에 포함하지 않는다. 작은 보정 계수는 실험 기록에 포함한다.',
        '- 공식 서버 제출은 실행하지 않았다.','',
        '선택 명세: PROTOCOL.json. 출처와 식: METHOD.md. 재실행: README.md.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"Final user_override={overridden}; original_gate={freeze['accepted']}; May vs original reported={gain:+.3f}%")

if __name__=='__main__':main()
