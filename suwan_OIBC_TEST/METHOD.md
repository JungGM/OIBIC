# 논문 활용과 자체 설계

1. Hybrid multi-scale feature extraction and parameter optimization for day-ahead photovoltaic power forecasting — DOI: 10.1038/s41598-026-67336-w.
   이동 평균·변동성 등 서로 다른 시간 규모의 기상 특징을 만드는 아이디어를 적용했다. 현재는 같은 관측소·발표 회차·대상일 안에서 구름/UV의 주변 3개 예보 평균·표준편차·하루 평균을 계산한다. 다음 시간의 실제 관측은 사용하지 않는다. 결측 시 3개 예보 행이 정확히 3시간 간격이 아닐 수 있다. 논문의 XGBoost·GA 전체 파이프라인을 재현한 것은 아니다.

2. Are better combinations of DERs more profitable?: Combinatorial optimization for aggregation of DERs in wholesale electricity markets — DOI: 10.1016/j.apenergy.2025.126264.
   어떤 자원을 함께 묶는지가 정산 성과에 영향을 준다는 원리를 적용했다. 논문의 역사적 발전량 공분산 대신, 고정한 예측 모델의 시간 외 잔차로 실제 대회 가중 절대손실을 직접 계산하도록 바꿨다. 발전소 교환/이동과 냉각 확률 탐색은 이번 대회용 자체 구현이다. 논문의 GA나 다른 시장 정산식을 그대로 복제하지 않았다.

3. A sample robust optimal bidding model for a virtual power plant — DOI: 10.1016/j.ejor.2024.03.001.
   과거 시나리오에만 유리한 선택을 경계하는 원리를 검증 구간 분리에 반영했다. 현재는 1~2월 잔차로 구성 선택, 3~5월 별도 평가 및 날짜 단위 재표집 코드를 구현했다. SRO의 불확실성 집합이나 최악 상황 최적화를 구현한 단계는 아니다.

자체 설계: 17시−10시 예보 수정량, 공급사 간 기온·습도·풍속 차이, 풍속 단위 통일, 운량×태양 위치 상호작용. 모든 특징의 효과는 남은 비교 실험에서 확인해야 한다. 논문의 개선율을 이 프로젝트의 성능으로 주장하지 않는다.
