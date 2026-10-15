"""Fast offline checks for the opt-in smoke guard and benchmark accounting."""
import json
import subprocess
import sys
from pathlib import Path

from examples.phase3_benchmark import build_report, markdown_report, run_case


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_counts_attempts_from_trace():
    normal = run_case(3, "layered_dag", False, 2, 0, 0, 0)
    retry = run_case(3, "layered_dag", True, 2, 0, 0, 0)
    assert normal["statuses"] == retry["statuses"] == ["success"] * 3
    assert normal["llm_attempts"] == 6 and normal["retries"] == 0
    assert retry["llm_attempts"] == 9 and retry["retries"] == 3
    assert retry["tool_calls"] == 3


def test_benchmark_report_has_comparable_cases():
    report = build_report([2], 1, 2, 0, 0, 0)
    assert report["kind"] == "offline_mock_benchmark" and len(report["cases"]) == 4
    for case in report["cases"]:
        assert case["serial"]["workers"] == 1
        assert case["parallel"]["workers"] == 2
        assert case["serial"]["success_rate"] == case["parallel"]["success_rate"] == 1
        assert case["speedup"] > 0
    assert "not real model" in markdown_report(report)


def test_live_smoke_is_off_by_default():
    result = subprocess.run([sys.executable, str(ROOT / "examples" / "phase3_live_smoke.py")],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"status": "not_run", "reason": "pass --live to opt in"}
