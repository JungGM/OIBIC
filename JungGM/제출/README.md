# 대회 제출 파일

학습 완료 후 이 폴더의 `composition.csv`와 `forecast.csv`를 사용한다.

위 두 원본 파일은 2026-11-02 첫 공식 제출본이다. 후속 개선 후보는 각 하위 폴더의 `validation.json`이 PASS이고 독립 감사가 완료된 경우 사용한다.

- `improved_v8/`: 기본 후보, 기존 앙상블 75% + CatBoost 25%.
- `improved_v7/`: 대안, 기존 앙상블 50% + 새 LightGBM 25% + CatBoost 25%.

한 후보 폴더 안의 `composition.csv`와 `forecast.csv`를 함께 선택한다. 생성 상태·검증 결과·성능 비교는 `../improvement_results.json`과 [개선 결과](../IMPROVEMENT_RESULTS.md)를 따른다. 로컬 검증을 통과해도 새 공식 점수와 순위는 서버 채점 전까지 알 수 없다.

- `composition.csv`: 발전소 253개, 고정 G1/G2 배정, 각 그룹 20MW 이상.
- `forecast.csv`: 2025-06-01 01:00~2026-06-01 00:00 KST, 8,760시간×2그룹 = 17,520행.
- `검증결과.json`: 파일 SHA-256, 학습 종료 시각, 행 수와 용량의 로컬 검사 기록.

실제 CSV와 학습 산출물은 `.gitignore`로 공개 Git에서 제외한다. GitHub에서 새로 받은 경우 `../README.md`의 학습 명령으로 로컬에 생성한다. 이 폴더에는 서버 제출을 수행하는 코드가 없다.
