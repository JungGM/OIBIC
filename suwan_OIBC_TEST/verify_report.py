"""Independent output checks and readable experiment report; no model selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;O=R/'runs/pilot_v2';D=R/'data'
sites=pd.read_csv(D/'sites.csv').sort_values('site_id').reset_index(drop=True)
comp=pd.read_csv(O/'composition_candidate.csv');m=pd.read_csv(O/'metrics.csv');summary=json.loads((O/'summary.json').read_text())
assert len(comp)==253 and comp.site_id.is_unique and set(comp.site_id)==set(sites.site_id)
assert set(comp.group)=={'G1','G2'}
caps=comp.merge(sites[['site_id','capacity_kw']],on='site_id',validate='one_to_one').groupby('group').capacity_kw.sum();assert (caps>=20000).all()
ts=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(O/'prediction_time_index.csv').ts,utc=True)).tz_convert('Asia/Seoul')
smp=pd.read_csv(D/'train/smp.csv');smp['ts']=pd.to_datetime(smp.ts,utc=True);w=smp.set_index('ts').smp.reindex(ts).to_numpy()*np.where((ts.hour>=11)&(ts.hour<=15),3,1)
labels=comp.set_index('site_id').group.reindex(sites.site_id)
reference=sites.current_group.to_numpy()
for model in ['baseline','enhanced']:
 e=np.load(O/f'{model}_residuals.npy').astype(float)
 assert e.shape==(3624,253) and np.isfinite(e).all()
 for name,g in [('reference',reference),('candidate',labels.to_numpy())]:
  grouped=pd.DataFrame(e.T).groupby(g).sum().T.to_numpy()
  for period,mask in [('selection',ts<=pd.Timestamp('2025-03-01',tz='Asia/Seoul')),('holdout',ts>=pd.Timestamp('2025-03-01 01:00',tz='Asia/Seoul'))]:
   computed=(np.abs(grouped[mask])*w[mask,None]).sum()
   expected=m.loc[(m.model==model)&(m.composition==name)&(m.period==period),'weighted_loss_W'].iloc[0]
   np.testing.assert_allclose(computed,expected,rtol=1e-6)
h=m[m.period=='holdout'];sel=m[m.period=='selection']
def value(df,model,comp,key='weighted_loss_W'):return float(df.loc[(df.model==model)&(df.composition==comp),key].iloc[0])
base=value(h,'baseline','reference');enh=value(h,'enhanced','reference');both=value(h,'enhanced','candidate')
monthly=pd.read_csv(O/'monthly_metrics.csv')
lines=['# OIBC 1·2단계 첫 실행 결과','', '2026-10-29 로컬 실행. **학습 10만 행의 동일 예산 비교이며 공식 서버 점수가 아니다.**','', '## 결과', '', '| 모델 | 그룹 | 3~5월 페널티(원) | 기준 대비 페널티 감소율 |', '|---|---|---:|---:|']
for _,r in h.iterrows():
 lines.append(f"| {'기준 특징' if r.model=='baseline' else '예보 통합·다중 시간 특징'} | {'기존 구성' if r.composition=='reference' else '최적화 후보'} | {r.penalty_krw:,.0f} | {r.local_score_vs_matched_baseline:+.3f}% |")
lines+=['',f'- 예측 특징 변경만의 효과: {(1-enh/base)*100:+.3f}%.',f'- 개선 모델에서 그룹 변경의 추가 효과: {(1-both/enh)*100:+.3f}%.',f'- 두 단계 결합 효과: {(1-both/base)*100:+.3f}%.',f"- 후보 그룹 용량: G1 {caps['G1']:,.2f}kW / G2 {caps['G2']:,.2f}kW. 변경 발전소 {summary['changed_sites']}곳.",f"- 날짜 단위 재표집 95% 구간: {summary['paired_day_bootstrap_score_95pct'][0]:+.3f}~{summary['paired_day_bootstrap_score_95pct'][1]:+.3f}%p의 로컬 점수. 날짜 간 날씨 상관을 완전히 반영하지 못해 낙관적일 수 있다.", '', '## 월별 성능', '', '| 월 | 기준+기존 페널티(원) | 개선+기존 | 개선+후보 | 결합 감소율 |','|---|---:|---:|---:|---:|']
for month in sorted(monthly.month.unique()):
 d=monthly[monthly.month==month];b=value(d,'baseline','reference','penalty_krw');e=value(d,'enhanced','reference','penalty_krw');c=value(d,'enhanced','candidate','penalty_krw')
 lines.append(f'| {month} | {b:,.0f} | {e:,.0f} | {c:,.0f} | {(1-c/b)*100:+.3f}% |')
lines+=['', '1~2월은 구성 선택에 썼으므로 독립 평가가 아니다. 3~5월은 이번 최초 실험의 별도 평가다.', '', '## 어떻게 만들었나', '', '예측 모델은 제공 기준선의 24개 특징을 바탕으로 두 발표 시각의 예보, 예보 수정량, 두 공급사 차이, 같은 대상일의 구름·UV 3시각/하루 통계를 합친 86개 특징을 사용했다. 두 발표 시각별 별도 모델 평균과 비교했다. 두 방식 모두 같은 학습 행과 600개 트리 설정을 썼다. 기준 방식은 두 모델, 개선 방식은 하나의 통합 모델이다.', '', '구성은 1~2월 잔차의 SMP×시간가중 절대오차를 직접 최소화하는 교환/이동 탐색이다. 3회×6,000번 탐색을 시도하며 용량 조건을 지키는 후보 중 최선을 유지했다. 전역 최적해를 증명한 것은 아니다.', '', '논문의 다중 시간 규모 특징과 오차 상쇄 아이디어를 대회 목적식에 맞게 구현했다. 논문 네트워크나 DRO·MILP 자체를 재현한 결과가 아니다.', '', '## 검증과 제한', '', '- 학습 2023~2024, 구성 선택 2025년 1~2월, 별도 평가 3~5월. 전체 253곳 실적이 완전한 시간만 평가했다.', '- 제공 특징 일치, 같은 대상일 경계와 rolling 통계, 그룹 간 상쇄 금지, 작은 최적화 문제의 완전탐색 정답 일치 테스트 통과.', '- 저장된 잔차를 별도 pandas 집계로 다시 계산해 8개 비교 결과를 검증했다. 후보 253개 ID의 유일성·완전성, 두 그룹 20MW 이상을 검증했다.', '- 학습 표본 최소 발전소는 8행이다. 전 발전소 모델은 공유하지만 신규 설비의 신뢰도는 낮을 수 있다.', '- 여름/가을 검증, 전체 학습량 비교, 공식 테스트 연도 예측, 제출 파일 검증·서버 제출은 수행하지 않았다.', '- 결과를 보고 반복 수정한다면 같은 3~5월은 이후 개발 검증에 해당한다. 계속 독립 최종 평가라고 부를 수 없다.', '', '## 파일', '', '- [실행법 및 방법 상세](README.md)', '- [전체 비교 수치](runs/pilot_v2/metrics.csv)', '- [월별 비교](runs/pilot_v2/monthly_metrics.csv)', '- [후보 그룹 구성](runs/pilot_v2/composition_candidate.csv)', '- [실행 설정](runs/pilot_v2/config.json)', '- [데이터 감사](runs/audit/audit.json)', '', '다음 실험은 예측 특징의 분리 비교와 학습 표본 확대를 우선한다. 구성 개선이 봄까지 유지되지 않으면 후보를 최종안으로 채택하지 않고, 여러 시기 잔차를 확보해 안정성부터 점검한다.']
(R/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'independent_checks':'passed','prediction_only_reduction_pct':100*(1-enh/base),'group_incremental_reduction_pct':100*(1-both/enh),'combined_reduction_pct':100*(1-both/base),'capacity_kw':caps.to_dict()},ensure_ascii=False))
print(h[['model','composition','penalty_krw','local_score_vs_matched_baseline']].to_string(index=False))
