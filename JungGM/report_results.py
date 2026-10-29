"""Generate a public Korean aggregate report from a verified completed local run.

Run after audit_run.py and, optionally, the saved-model reproduction check.
No raw rows, site identifiers, predictions, or local source paths enter the report.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
KST = timezone(timedelta(hours=9))
LABELS = {
    "official_lgb": "적응 기준선: LightGBM 10·17시 평균",
    "lgb_l2": "LightGBM L2",
    "lgb_l2_wide": "LightGBM L2 넓은 트리",
    "lgb_l1": "LightGBM L1",
    "lgb_recent": "LightGBM L2 최근 12개월",
    "xgb_regular": "XGBoost 기본",
    "xgb_deep": "XGBoost 깊은 트리",
}
BLOCKS = {"seasonal_proxy": set(range(7, 13)), "complete_fleet": {1, 2}}
DEV_MODELS = ("official_reference", "ensemble_reference", "selected_system")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name):
    require(not isinstance(value, bool), f"Invalid numeric field: {name}")
    result = float(value)
    require(math.isfinite(result), f"Nonfinite field: {name}")
    return result


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact_time(value, expected):
    result = datetime.fromisoformat(str(value))
    require(result.tzinfo is not None and result.utcoffset() == timedelta(hours=9),
            "Receipt timestamp must use KST")
    require(result == datetime.fromisoformat(expected), "Unexpected receipt time boundary")


def reduction(reference, candidate):
    require(reference > 0, "Cannot report a relative change against zero cost")
    return 100 * (1 - candidate / reference)


def money(value):
    return f"{number(value, 'amount'):,.2f}"


def percent(value):
    return f"{value:+.2f}%"


def selection_tables(selection):
    """Normalize each candidate against the identical monthly baseline scope."""
    cover = {}
    for row in selection["coverage"]:
        key = (row["block"], int(row["month"]))
        require(key not in cover, "Duplicate coverage month")
        cover[key] = row
    expected = {(block, month) for block, months in BLOCKS.items() for month in months}
    require(set(cover) == expected, "Expected all eight selection coverage months")
    seasonal_sizes = {int(cover[("seasonal_proxy", month)]["cohort_sites"]) for month in range(7, 13)}
    require(len(seasonal_sizes) == 1, "Seasonal cohort must be fixed across months")
    require(all(int(cover[("complete_fleet", month)]["cohort_sites"]) == 253 for month in (1, 2)),
            "Winter selection must use all 253 sites")
    for row in cover.values():
        require(int(row["hours"]) > 0 and number(row["capacity_kw"], "cohort capacity") > 0,
                "Invalid selection coverage")

    costs = {}
    for row in selection["monthly_candidates"]:
        key = (row["block"], int(row["month"]), row["model"])
        require(key not in costs, "Duplicate candidate monthly cost")
        value = number(row.get("weighted_error", row.get("penalty_krw")), "candidate cost")
        require(value >= 0, "Negative candidate cost")
        costs[key] = value
    require(set(costs) == {(block, month, model) for block, month in expected for model in LABELS},
            "Expected seven candidates for every selection month")
    table = []
    for model in LABELS:
        block_costs = []
        for block, months in BLOCKS.items():
            ratios = []
            for month in sorted(months):
                base = costs[(block, month, "official_lgb")]
                require(base > 0, "Zero monthly baseline cost")
                ratios.append(costs[(block, month, model)] / base)
            block_costs.append(statistics.mean(ratios))
        table.append((model, *block_costs, statistics.mean(block_costs)))
    return cover, table


def verify_inputs(run, submission):
    completion = load_json(run / "completion.json")
    independent = load_json(run / "independent_audit.json")
    require(independent.get("passed") is True, "Independent audit has not passed")
    for key in ("raw_source_hashes_unchanged", "original_truth_precision", "production_aggregation_equal"):
        require(independent.get(key) is True, "Required independent audit check did not pass")
    require(completion.get("format_validation") == "PASS", "Submission format check did not pass")
    require(completion.get("no_server_submission") is True, "Unexpected server-submission state")
    require(completion.get("composition_rows") == 253 and completion.get("forecast_rows") == 17520,
            "Unexpected submission row counts")
    require(completion.get("site_prediction_rows") == 253 * 8760,
            "Unexpected production site-grid count")
    require(independent.get("production_site_rows") == 253 * 8760
            and independent.get("forecast_rows") == 17520,
            "Unexpected independent production row counts")
    exact_time(completion["fit_cutoff"], "2025-05-31 17:00:00+09:00")
    exact_time(completion["target_start"], "2025-06-01 01:00:00+09:00")
    exact_time(completion["target_end"], "2026-06-01 00:00:00+09:00")
    for filename in ("composition.csv", "forecast.csv"):
        actual = digest(submission / filename)
        require(actual == completion["files"][filename], "Submission differs from completion receipt")
        require(actual == independent["submission_sha256"][filename],
                "Submission differs from independent audit")
    with (submission / "composition.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["site_id", "group"], "Unexpected composition schema")
        rows = list(reader)
    require(len(rows) == 253 and len({row["site_id"] for row in rows}) == 253,
            "Invalid composition count or duplicate identifiers")
    require(all(row["group"] in {"G1", "G2"} for row in rows), "Invalid submitted group")
    counts = {group: sum(row["group"] == group for row in rows) for group in ("G1", "G2")}
    capacities = completion["group_capacity_kw"]
    require(set(capacities) == {"G1", "G2"}, "Unexpected capacity groups")
    require(all(number(capacities[group], "group capacity") >= 20000 for group in counts),
            "Submitted group below 20 MW")
    reproduction_path = run / "reproduction_check.json"
    reproduction = load_json(reproduction_path) if reproduction_path.exists() else None
    if reproduction is not None:
        require(reproduction.get("passed") is True, "Saved-model reproduction check did not pass")
    return completion, independent, reproduction, counts


def build_report(selection, monthly, gate_report, search, frozen, completion,
                 independent, reproduction, counts):
    cover, candidates = selection_tables(selection)
    weights = selection["selected"]["weights"]
    require(set(weights) == set(LABELS), "Unexpected selected candidate names")
    weights = {name: number(value, "ensemble weight") for name, value in weights.items()}
    require(all(value >= 0 for value in weights.values()) and math.isclose(sum(weights.values()), 1),
            "Invalid selected convex ensemble")
    require(sum(value > 0 for value in weights.values()) <= 3, "More than three ensemble components")
    require(all(math.isclose(value * 10, round(value * 10), abs_tol=1e-8) for value in weights.values()),
            "Ensemble weights are not on the 0.1 grid")
    require(set(frozen["weights"]) == set(weights), "Unexpected frozen weight names")
    require(all(math.isclose(weights[name], number(frozen["weights"][name], "frozen weight"))
                for name in weights), "Selected and frozen weights differ")
    accepted = gate_report.get("accepted")
    require(isinstance(accepted, bool) and accepted == frozen["group_candidate_accepted"],
            "Group gate and frozen choice differ")
    gate = gate_report["gate"]
    require(gate.get("accepted") is accepted, "Inconsistent gate acceptance")
    required_gain = number(gate["required_improvement_fraction"], "gate threshold")
    require(math.isclose(required_gain, .01), "Unexpected group improvement threshold")
    gain = number(gate["improvement_fraction"], "group gain")
    lower_krw = number(gate["mean_daily_savings_lower90"], "bootstrap lower bound") * .1
    days = int(gate["days"])
    require(accepted == (days >= 14 and gain >= required_gain and lower_krw > 0),
            "Group acceptance does not match its stated gate")
    metrics = {(int(row["month"]), row["model"]): row for row in monthly}
    require(len(metrics) == len(monthly) and set(metrics) == {
        (month, model) for month in (3, 4, 5) for model in DEV_MODELS},
        "Expected nine development score records")
    independent_rows = independent["independent_group_scores"]
    independent_metrics = {(int(row["month"]), row["model"]): row for row in independent_rows}
    require(len(independent_metrics) == len(independent_rows) and set(independent_metrics) == set(metrics),
            "Independent audit score coverage differs from the report")
    require(all(math.isclose(number(metrics[key]["penalty_krw"], "recorded penalty"),
                             number(independent_metrics[key]["penalty_krw"], "independent penalty"),
                             rel_tol=1e-12, abs_tol=1e-5) for key in metrics),
            "Reported development scores differ from independent audit")
    for month in (3, 4, 5):
        require(len({int(metrics[(month, model)]["hours"]) for model in DEV_MODELS}) == 1,
                "Development models use different scoring hours")
    may_base = number(metrics[(5, "official_reference")]["penalty_krw"], "May baseline")
    may_system = number(metrics[(5, "selected_system")]["penalty_krw"], "May selected")
    may_gain = reduction(may_base, may_system)
    may_comparison = f"{abs(may_gain):.2f}% " + ("낮았다" if may_gain >= 0 else "높았다")
    selected_loss = number(selection["selected"]["mean_monthly_relative_loss"], "selection objective")
    searches = int(selection["search_count"])
    require(searches > 0, "Empty ensemble search")
    lines = [
        "# 논문 기반 제출 모델: 실행 결과", "",
        f"5월의 실행 내부 고정 감사에서 선택 시스템의 오차 벌금은 적응 기준선보다 **{may_comparison}**. "
        f"앙상블은 {searches:,}개 조합을 비교해 선택했으며, 공식 기간용 제출 파일의 독립 로컬 검증을 통과했다.", "",
        "**해석 범위:** 2024~2025년 자료는 이전에도 검토했다. 5월은 이번 실행에서 선택을 고정한 후 확인한 감사 구간이며 미사용 외부 홀드아웃이 아니다. "
        "공식 평가연도 발전실적은 없으므로 연간 정확도·공식 점수·순위를 확정할 수 없다.", "",
        "## 모델과 앙상블 선택", "",
        "2024년 7월 예측은 6월 30일 17시, 2025년 1월 예측은 2024년 12월 31일 17시까지의 정답으로 학습했다. "
        "기본 학습창은 직전 24개월이며 최근 후보만 12개월이다. 기준선은 공식 구조에 시간 경계·B 풍속 단위를 맞춘 변형이다. "
        "강화 후보는 용량 가중 학습을 사용한다.", "",
        "아래 값은 각 월의 같은 정답·시각·그룹을 사용한 기준선 비용으로 정규화한 값이다. 낮을수록 좋다. "
        "계절 부분 코호트 블록과 겨울 전체 관측 블록을 각각 50% 반영하고 블록 안의 월은 같은 비중으로 둔다.", "",
        "| 후보 | 7~12월 부분 대리목적 | 1~2월 전체 253개소 | 50:50 선택값 |",
        "|---|---:|---:|---:|",
    ]
    for name, seasonal, winter, balanced in candidates:
        lines.append(f"| {LABELS[name]} | {seasonal:.4f} | {winter:.4f} | {balanced:.4f} |")
    components = " + ".join(f"{value:.1f} × {LABELS[name]}" for name, value in weights.items() if value > 0)
    lines += ["", f"**채택 앙상블:** {components}. 선택값 **{selected_loss:.4f}** "
              f"(기준선 1 대비 {percent(100 * (1-selected_loss))} 감소). 최대 3개 모델, 0.1 간격의 비음수·합계 1 가중치를 탐색했다.", "",
              "### 월별 채점 범위", "",
              "7~12월은 전체 구간 실적 보유율 95% 이상인 하나의 고정 코호트다. 그 코호트 전원의 실적이 있는 시간만 썼다. "
              "이는 전체 VPP 점수가 아닌 부분 대리목적이며, 코호트 밖 설비에 같은 개선이 유지된다는 보장은 없다. "
              "1~2월은 전체 253개소의 동시 실적 시간이다. 각 선택 구간의 마지막 날은 17시까지다.", "",
              "| 월 | 설비 수 | 용량 합계(kW) | 동시 실적 시간 | 점수 범위 |",
              "|---|---:|---:|---:|---|"]
    for block, month in [("seasonal_proxy", month) for month in range(7, 13)] + [("complete_fleet", 1), ("complete_fleet", 2)]:
        row = cover[(block, month)]
        year, scope = (2024, "고정 부분 코호트") if block == "seasonal_proxy" else (2025, "전체 253개소")
        lines.append(f"| {year}-{month:02d} | {int(row['cohort_sites'])} | {money(row['capacity_kw'])} | {int(row['hours']):,} | {scope} |")
    lines += ["", "## 그룹 선택과 후속 감사", "",
              "모든 금액은 `0.10 × Σ SMP × k × |그룹 예측합 − 그룹 실적합|`인 오차 벌금(원)이다. "
              "3월은 그룹 탐색, 4월은 채택 판단, 5월은 고정 후 감사다. 3~5월의 마지막 날은 17시까지 평가했다.", "",
              "| 2025년 월 | 동시 실적 시간 | 기준선·참조그룹 | 앙상블·참조그룹 | 선택 시스템 | 앙상블 감소율 | 시스템 감소율 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for month in (3, 4, 5):
        rows = [metrics[(month, model)] for model in DEV_MODELS]
        values = [number(row["penalty_krw"], "development penalty") for row in rows]
        require(all(value >= 0 for value in values), "Negative development penalty")
        lines.append(f"| {month}월 | {int(rows[0]['hours']):,} | {money(values[0])} | {money(values[1])} | {money(values[2])} | "
                     f"{percent(reduction(values[0], values[1]))} | {percent(reduction(values[0], values[2]))} |")
    failures = []
    if gain < required_gain:
        failures.append("4월 감소율 1% 미달")
    if lower_krw <= 0:
        failures.append("날짜별 쌍체 부트스트랩의 90% 구간 하한이 0 이하")
    if days < 14:
        failures.append("판단 가능한 날짜 14일 미만")
    decision = "그룹 후보를 채택했다" if accepted else "제공 참조 구성을 유지했다: " + "; ".join(failures)
    march_gain = number(search["training_improvement_fraction"], "March search gain") * 100
    april_reference = number(gate_report["reference"]["penalty_krw"], "April group reference")
    april_candidate = number(gate_report["candidate"]["penalty_krw"], "April group candidate")
    lines += ["", f"**최종 구성 결정:** {decision}.", "",
              f"3월 잔차의 용량 제약 탐색에서는 벌금이 {percent(march_gain)} 감소했다. "
              f"동결한 후보의 4월 벌금은 {money(april_candidate)}원(참조 구성 {money(april_reference)}원), 감소율은 {percent(gain*100)}였다. "
              f"{days}일·{int(gate['bootstrap_samples']):,}회 부트스트랩에서 평균 일별 절감액의 90% 구간 하한은 {money(lower_krw)}원/일이었다. "
              "채택 조건은 감소율 1% 이상, 하한 양수, 최소 14일을 모두 만족하는 것이다.", "",
              "탐색은 참조 구성을 포함한 용량 제약 휴리스틱이다. 참조 구성 유지나 후보 채택은 그 검증 규칙의 결과이며 전역 최적 구성의 증명이 아니다. "
              "날짜별 부트스트랩은 날짜의 교환가능성을 가정하며 계절 변화에 대한 보장은 없다.", "",
              "| 최종 그룹 | 설비 수 | 용량 합계(kW) | 20MW 제약 |",
              "|---|---:|---:|---|" ]
    for group in ("G1", "G2"):
        lines.append(f"| {group} | {counts[group]} | {money(completion['group_capacity_kw'][group])} | 통과 |")
    lines += ["", "## 제출 파일과 독립 검증", "",
              "최종 생산 모델의 정답 상한은 **2025-05-31 17:00 KST**다. "
              "2025-06-01 01:00~2026-06-01 00:00 KST를 예측했다. "
              "`제출/composition.csv`는 253행, `제출/forecast.csv`는 17,520행(각각 헤더 제외)이다.", "",
              "| 검증 항목 | 결과 |", "|---|---|",
              "| 제출 형식·전체 키·시간축·용량 제약 | PASS |",
              "| 원본 파일 해시 보존 | PASS |",
              "| 개발 정답의 원본 정밀도 일치 | PASS |",
              "| 개발 그룹 비용 독립 재계산 | PASS |",
              "| 전체 발전소 예측의 그룹 합계와 제출 CSV 일치 | PASS |",
              "| 제출 두 파일과 완료·독립감사 기록의 해시 일치 | PASS |"]
    lines.append("| 저장 모델 별도 재현 검사 | " + ("PASS" if reproduction is not None else "별도 기록 없음") + " |")
    lines += ["", "학습 모델·캐시·예측·제출 CSV는 공개 Git에서 제외하며 이 문서에는 집계 결과만 기록했다. "
              "공식 서버 제출은 실행하지 않았다. 로컬 검사 통과가 공식 접수나 평가연도 성능을 보증하지는 않는다.", "",
              "재현 방법은 [README](README.md), 논문 적용·검증 설계는 [PAPER_METHOD](PAPER_METHOD.md), "
              "공식 조건과 정보 경계는 [RULES](RULES.md)를 참조한다.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "RESULTS.md")
    args = parser.parse_args()
    completion, independent, reproduction, counts = verify_inputs(args.run, args.submission)
    report = build_report(
        load_json(args.run / "selection" / "selection.json"),
        load_json(args.run / "development" / "monthly_metrics.json"),
        load_json(args.run / "development" / "group_gate.json"),
        load_json(args.run / "development" / "group_search.json"),
        load_json(args.run / "frozen_choice_before_may_audit.json"),
        completion, independent, reproduction, counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print("PASS: Korean aggregate results report generated")


if __name__ == "__main__":
    main()
