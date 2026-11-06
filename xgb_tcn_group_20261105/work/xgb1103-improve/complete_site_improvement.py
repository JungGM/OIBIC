"""Finish the local improvement study after independently validated export."""
from pathlib import Path
import argparse,json,hashlib,zipfile,math,shutil
import pandas as pd

ROOT=Path(__file__).resolve().parent
STUDY=Path('outputs/OIBC_발전소개별특성_개선_20261105')
CANDIDATE=Path('outputs/OIBC_XGB_TCN_제출후보_20261105')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def clean(v):
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if not (k in ['JanMar','AprMay'] and isinstance(x,dict) and x.get('hours')==0)}
    if isinstance(v,list):return [clean(x) for x in v]
    return v
def save(p,v):p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g=read(STUDY/'그룹손실_추가학습_검증.json')['folds'];old=read(Path('outputs/OIBC_TCN_첫후보_20261104/metrics.json'))
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    fig,axs=plt.subplots(1,2,figsize=(13,5.8),facecolor='#F6F8FB')
    for ax,fold,title in zip(axs,['spring','summer'],['2025년 1~5월 · 253개소','2024년 6~12월 · 207개소 대리검증']):
        values=[old[fold]['xgb177']['all']['penalty_krw'],g[fold]['old_5050']['all']['penalty_krw'],g[fold]['site_control']['all']['penalty_krw'],g[fold]['group_aligned']['all']['penalty_krw']]
        bars=ax.bar(range(4),[v/1e6 for v in values],color=['#AABAC8','#277DA1','#E7A760','#43AA8B'],width=.62)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.2,f'{v/1e6:.2f}',ha='center',fontsize=10)
        ax.set_xticks(range(4),['XGBoost\n단독','기존\n50:50','추가학습\n대조군','그룹손실\n50:50']);ax.set_ylim(0,115);ax.set_title(title,loc='left',fontsize=13,fontweight='bold',pad=14);ax.set_ylabel('과거 검증 벌금 (백만원)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.set_facecolor('white')
        for s in ['top','right']:ax.spines[s].set_visible(False)
    fig.suptitle('채택 후보 검증 · 추가 학습과 그룹 손실의 효과',x=.06,ha='left',fontsize=20,fontweight='bold',color='#19394F')
    fig.text(.06,.045,'낮을수록 좋음 | 두 구간은 다른 학습 마감·모집단 | 금액 합산이나 공식 연간 순위 예측에 사용하지 않음',fontsize=9,color='#526879')
    fig.tight_layout(rect=(0,.08,1,.90),w_pad=2.7);fig.savefig(STUDY/'채택모델_검증비교.png',dpi=180,facecolor=fig.get_facecolor());fig.savefig(STUDY/'채택모델_검증비교.pdf',facecolor=fig.get_facecolor());plt.close(fig)

def finish():
    valid=read(CANDIDATE/'검증결과.json');decision=read(CANDIDATE/'모델선택.json');assert valid['valid'] and valid['existing_inputs_unchanged']
    g=read(STUDY/'그룹손실_추가학습_검증.json');old=read(Path('outputs/OIBC_TCN_첫후보_20261104/metrics.json'));site=read(STUDY/'모델검증.json');cal=clean(read(STUDY/'과거예측_개별배율_검증.json'))
    save(STUDY/'과거예측_개별배율_검증.json',cal)
    assert g['decisions'][decision['choice']]['promotion_passed'];assert not site['promotion_passed'] and not cal['promotion_passed']
    rows=[];monthly=[];tables=[];labels={'spring':'2025년 1~5월·253개소','summer':'2024년 6~12월·207개소 대리검증'}
    for fold,v in g['folds'].items():
        raw=old[fold]['xgb177']['all']['penalty_krw'];baseline=v['old_5050']['all']['penalty_krw'];control=v['site_control']['all']['penalty_krw'];selected=v[decision['choice']]['all']['penalty_krw']
        row={'period':labels[fold],'xgb_alone':raw,'old_5050':baseline,'site_only_control':control,'selected_5050':selected,'gain_vs_old5050':baseline-selected,'reduction_vs_old5050_pct':100*(baseline-selected)/baseline,'gain_vs_xgb':raw-selected,'reduction_vs_xgb_pct':100*(raw-selected)/raw,'group_loss_extra_gain_vs_control':control-selected};rows.append(row)
        tables.append(f'<tr><td>{labels[fold]}</td><td>{raw:,.0f}원</td><td>{baseline:,.0f}원</td><td>{control:,.0f}원</td><td>{selected:,.0f}원</td><td>{row["reduction_vs_old5050_pct"]:.2f}%</td></tr>')
        for month,z in v[decision['choice']]['monthly'].items():
            b=v['old_5050']['monthly'][month]['penalty_krw'];monthly.append({'fold':fold,'month':month,'baseline_penalty':b,'selected_penalty':z['penalty_krw'],'gain':b-z['penalty_krw'],'reduction_pct':100*(b-z['penalty_krw'])/b})
    pd.DataFrame(rows).to_csv(STUDY/'채택모델_최종비교.csv',index=False,encoding='utf-8-sig');pd.DataFrame(monthly).to_csv(STUDY/'채택모델_월별비교.csv',index=False,encoding='utf-8-sig')
    md=['# 지역·발전소 특성 점검부터 제출 후보까지','','## 최종 선택','','XGBoost177개 특징 모델50% + 그룹 손실로 추가 학습한TCN50%. 기존G1/G2 배정과SMP 가중치는 유지했다. 개별 발전소 배율, 명시적ID 이득 파라미터, 연도별SMP 정규화, 비율 최적화는 반영하지 않았다.','','|과거 검증|XGBoost 단독|기존50:50|추가학습 대조군|선택 후보50:50|기존50:50 대비 벌금 감소|','|---|---:|---:|---:|---:|---:|']
    for r in rows:md.append(f'|{r["period"]}|{r["xgb_alone"]:,.0f}원|{r["old_5050"]:,.0f}원|{r["site_only_control"]:,.0f}원|{r["selected_5050"]:,.0f}원|{r["reduction_vs_old5050_pct"]:.2f}%|')
    md+=['','개선 전부가 그룹 손실의 효과는 아니다. 추가 학습 대조군도 개선됐다. 동일한 추가 학습량에서 그룹 손실 후보의 추가 절감은2025년약14.3만원,2024년약74.0만원이었다. 그룹 후보는 두 기간 모두1% 이상 개선하고 어느 달도기존50:50보다2% 초과 악화되지 않는 사전 기준을 통과했다.','','## 진단과 미채택 실험','','- R2의 제공 관측소ID가 예보에서 빠져 있거나 핵심 변수 결측이 많다는 증거는 없었다.2025년1~5월 핵심 예보 변수의 용량 가중 커버리지는100%. 운량 예보 오차도R2만 유난히 크지 않았다. 관측소 날씨가 발전소 현장의 정답이라는 뜻은 아니다.','- A 실측 기상에 지점·시각이 겹치고 값이 다른2개 키가 있어 진단에서만 제외했다. 예보는 중복 키가 없었다. 이 실측 자료를 최종 예측 입력으로 사용하지 않았다.','- 발전소별 이득을 공동 학습한TCN은2025년 결합 벌금0.26% 악화,2024년0.87% 개선으로 채택 기준 미달.','- 과거OOF 예측의 발전소별 배율은2025년4~5월0.96%,2024년9~12월1.70% 악화되어 미채택. 이 후보의 평가 구간은전체기간 후보와 다르다.','- PV011의XGBoost 유효 주간 표본은 검증 마감2024년 말66시간,최종 학습 범위에서는1845시간이다. TCN은완료된24시간 묶음으로 별도 집계한다. 검증에서의 학습 이력 부족을최종 모델의 동일 오차로 단정하지 않는다.','','## 채택한 알고리즘','','1. XGBoost:177개 기존 특징, 설비이용률 타깃, 기존 가중 제곱오차 모델을 그대로 사용.','2. TCN:하루24시간×31개 예보·태양·위치/용량 변수와31개 결측표시.16채널·커널3·팽창1/2/4/8의 잔차 합성곱.','3. 기존 가중MAE로기본 학습한 뒤학습률0.0001로2회 추가 학습. 손실=0.2×개별 가중절대오차+0.8×관측 가용표본의 그룹 잔차합 절대오차.','4. 같은 날짜의 발전소를 함께 묶고 원래G1/G2별 잔차를 합산한다. 누락 실적을0으로 채우지 않는다. 과거253개소 실적이 완전하지 않아학습 그룹 손실은관측 부분집합 기준이며 공식 전체 그룹 손실과 완전히 동일하지 않다. 타깃clip·야간 마스크도존재한다.','5. 날짜당 고정 가중치 질량으로 정규화해과거 소규모 관측 집합이후기 대규모 집합과동일 비중을 받지 않도록 했다.','6. 발전소별 예측CF를float64로 변환해용량을 곱하고XGBoost와50:50 결합한 뒤G1/G2로 합산. 야간은기존 태양고도 기준0.','','Bi-GRU·Attention을결합한 논문형 단일모델을 구현한 결과가 아니다. 최종에는XGBoost와TCN만 사용한다.','','## 최종 재학습과 제출 파일','','검증 모델은2024년5월 말/12월 말까지 학습한 별도 모델이다. 제출 후보의TCN은2025-05-31 17:00 이전 허용 역사로새로 학습했다. 기본10회 후선택된 그룹 손실2회 추가 학습. 정규화도허용 학습자료에서만 적합했다. 완료된24시간 묶음만 사용하므로마지막 불완전 하루는TCN 학습에서 제외한다.','평가 연도 발전량·실측 기상·실제SMP를 예측 입력으로 사용하지 않았다. 평가 연도SMP 파일에서는시각 열만 읽었다. 평가연도 예보의연간 통계로정규화·그룹 구성하지 않았다. 이미전날17시까지 발행된같은 대상일24시간 예보를처리했다.','예측입력 생성기를과거캐시의504개 발전소·시간×31개 특징과대조했다. 저장 모델 재로딩 동일성,253개소 중복/누락,그룹 용량,8760시간×2개 그룹,음수/빈값/무한대,야간0,독립 합산,50:50 산식,원본XGBoost 합산 일치를검사했다.','제출 후보 폴더: '+str(CANDIDATE),'- composition.csv:253개소 원래G1/G2.','- forecast.csv:17520행.2025-06-01 01:00~2026-06-01 00:00 KST.','- 검증결과.json:실제 파일 해시·검증 수치.','온라인 대회 제출을자동 실행하지 않았다. ZIP은전달용두 CSV만 포함한다.','','## 성능 해석의 한계','','2025년은253개소3624시간,2024년은207개소4905시간 부분집합 대리검증이다. 두 금액을 합쳐공식 연간 정산금으로환산하지 않는다. 과거 데이터를여러 차례 확인한 이력이 있어완전 미사용 외부 테스트가 아니다.','최종 재학습 모델은검증 당시 가중치와 다르므로같은 개선율이공식 평가 연도에재현된다고 보장하지 않는다. 공식 정산금·순위는알 수 없다.2,500만원 격차를메웠다고 주장하지 않는다.','기존 제출파일·기존모델은보존했다. 새 후보의생성은검증된 형식과선택한 설계를완성한 것이며1등 보장이 아니다.']
    (STUDY/'최종_개선보고서.md').write_text('\n'.join(md),encoding='utf-8')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>최종 개선 검증</title><style>body{font-family:Malgun Gothic,sans-serif;background:#f4f6f8;color:#182b3b}main{max-width:1250px;margin:40px auto;padding:32px;background:white;border-radius:16px}p{line-height:1.8}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:14px 7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%}.note{background:#edf5f1;padding:20px;border-radius:10px}small{color:#526271}</style><main><h1>최종 개선 검증과 제출 후보</h1><p class="note"><b>선택: XGBoost50% + 그룹 손실로 추가 학습한TCN50%</b><br>기존그룹·SMP 가중치·특징 구성 유지. 개별배율 보정은미채택.</p><table><tr><th>과거 검증</th><th>XGBoost</th><th>기존50:50</th><th>추가학습 대조군</th><th>선택 후보</th><th>기존50:50 대비 개선</th></tr>'''+''.join(tables)+'''</table><p>개선에는추가 학습의 효과가 포함됨. 그룹 손실의추가 절감은대조군 대비2025년약14.3만원·2024년약74.0만원.</p><img src="채택모델_검증비교.png" alt="검증 비교"><h2>제출 파일</h2><p>composition.csv253개소, forecast.csv17520행. 원래G1/G2·각20MW 이상. 상세 검증은제출 후보 폴더의검증결과.json 참조.</p><h2>데이터 점검과 적용하지 않은 변경</h2><p>R2에서예보 누락·단위·지점 키 오류가핵심 원인이라는증거는찾지 못함. 실측A 중복2개 키는진단에서만 제외.<br>발전소별 이득 공동학습과과거OOF 배율 보정은기준 미달로제외. 그룹 손실을적용한 짧은추가 학습만선택.</p><small>과거 검증기간은이전 실험에서검토한 이력이 있음.2024년은207개소 부분집합 대리검증. 두 구간 금액을더해공식 연간 점수로사용하지 않음. 최종모델은허용 학습기간으로재학습한 별도가중치. 실제 정산금과순위는미확정. 기존파일 보존. 온라인 제출은자동실행하지 않음.</small></main></html>'''
    (STUDY/'최종_개선보고서.html').write_text(html,encoding='utf-8')
    shutil.copyfile(STUDY/'최종_개선보고서.md',CANDIDATE/'모델과검증_설명.md')
    save(STUDY/'최종_실행검사.json',{'submission_validation':valid,'decision':decision,'benchmarks':rows,'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['site_adaptation_diagnostics.py','tcn_site_adaptation.py','tcn_site_calibration.py','tcn_group_finetune.py','export_xgb_tcn_candidate.py','complete_site_improvement.py']},'report_only_cleanup':'Empty Jan-Mar slices in audit-only calibration report removed; nonfinite optional metrics become null. Frozen scales and all penalty values unchanged.'})
    save(CANDIDATE/'TCN_특징31.json',read(ROOT/'production_xgb_tcn_20261105/base/production/training.json')['features']);shutil.copyfile(ROOT/'expanded_features.json',CANDIDATE/'XGBoost_특징177.json')
    (STUDY/'README_실행.txt').write_text('선택 모델은XGBoost+TCN이며그룹 손실로TCN만2회 추가학습합니다.\n제출 ZIP은별도 OIBC_XGB_TCN_제출후보_20261105.zip에두 CSV만포함합니다.\n본 ZIP은코드·집계 결과이며원본대회데이터·시간별발전소예측·학습모델은제외했습니다.\n기존 프로젝트의데이터·특징캐시·원래XGBoost 모델 및검증 예측이필요합니다. 새 PC에서이 압축파일만으로독립 실행되는 패키지는아닙니다.\nPyTorch 경로는이 PC의 C:/Users/swan/Documents/Codex/tcn-libs이며다른환경에서는조정하십시오.\n선택/재학습/예측/검증: export_xgb_tcn_candidate.py train / predict / validate 순서로실행.\n기존완료파일을덮어쓰지 않습니다. 신규실험은RUN/OUT/PROD 경로변경이필요합니다.\n검증 당시 그룹학습 소스는스냅샷으로보존했습니다. 이후생산재사용용 경로·추론 옵션만추가했으며손실·학습 설정은변경하지 않았습니다.\n',encoding='utf-8')
    files=['site_adaptation_diagnostics.py','tcn_site_adaptation.py','tcn_site_calibration.py','tcn_group_finetune.py','export_xgb_tcn_candidate.py','complete_site_improvement.py','tcn_candidate.py','experiment.py','xgb_loss_trial.py','deep_linear_ensemble.py']
    with zipfile.ZipFile(Path('outputs/OIBC_최종개선_코드와검증결과_20261105.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(STUDY.iterdir()):
            if p.is_file():z.write(p,arcname=STUDY.name+'/'+p.name)
        for n in files:z.write(ROOT/n,arcname='work/xgb1103-improve/'+n)
        for n in ['expanded_features.json']:z.write(ROOT/n,arcname='work/xgb1103-improve/'+n)
        for p in (ROOT/'reference').glob('*.py'):z.write(p,arcname='work/xgb1103-improve/reference/'+p.name)
        z.write(ROOT/'reference/feature_sets.json',arcname='work/xgb1103-improve/reference/feature_sets.json')
    print(json.dumps({'benchmarks':rows,'validated':valid['valid'],'choice':decision['choice']},ensure_ascii=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plot','finish']);a=p.parse_args();{'plot':plot,'finish':finish}[a.stage]()
