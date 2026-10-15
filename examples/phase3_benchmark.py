"""Repeatable offline workload; measurements are local Mock timings, not LLM latency."""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xcode_agent import (AgentFactory, LLMRateLimitError, LLMResponse, TaskNode,
                         ToolCall, ToolRegistry, TraceRecorder)


class ToolArgs(BaseModel):
    value: int


class MockLLM:
    def __init__(self, llm_delay_s: float, retry: bool, retry_delay_s: float):
        self.llm_delay_s = llm_delay_s
        self.retry = retry
        self.retry_delay_s = retry_delay_s
        self.failed_once = False

    def chat(self, messages, tools):
        time.sleep(self.llm_delay_s)
        if self.retry and not self.failed_once:
            self.failed_once = True
            raise LLMRateLimitError("fixed mock rate limit", retry_after_s=self.retry_delay_s)
        if messages[-1]["role"] == "tool":
            return LLMResponse(content="mock complete")
        return LLMResponse(tool_calls=[ToolCall(id="mock-call", name="mock_work",
                                                arguments='{"value":1}')])


def make_tasks(count: int, topology: str) -> list[TaskNode]:
    if count < 1 or topology not in {"independent", "layered_dag"}:
        raise ValueError("Invalid task count or topology")
    roots = (count + 1) // 2
    return [TaskNode(task_id=f"task-{i}", description=f"fixed mock task {i}",
                     depends_on=[f"task-{i - roots}"] if topology == "layered_dag" and i >= roots else [])
            for i in range(count)]


def run_case(count: int, topology: str, retry: bool, workers: int,
             llm_delay_s: float, tool_delay_s: float, retry_delay_s: float) -> dict:
    if workers < 1 or min(llm_delay_s, tool_delay_s, retry_delay_s) < 0:
        raise ValueError("Workers must be positive and delays nonnegative")
    trace = TraceRecorder()
    tools = ToolRegistry()

    def mock_work(value: int) -> int:
        time.sleep(tool_delay_s)
        return value * 2

    tools.register("mock_work", "Fixed local mock operation", ToolArgs, mock_work)
    factory = AgentFactory(lambda: MockLLM(llm_delay_s, retry, retry_delay_s), tools,
                           synchronize_tools=False, trace=trace)
    root = factory.create()
    started = time.perf_counter()
    results = root.dispatch(make_tasks(count, topology), max_workers=workers)
    wall_ms = (time.perf_counter() - started) * 1000
    events = trace.events(root.last_run_id)
    return {
        "wall_ms": wall_ms,
        "success_count": sum(result.status == "success" for result in results),
        "task_count": count,
        "llm_attempts": sum(event.event_type == "llm_start" for event in events),
        "retries": sum(event.event_type == "llm_retry" for event in events),
        "tool_calls": sum(event.event_type == "tool_start" for event in events),
        "statuses": [result.status for result in results],
    }


def summarize_runs(samples: list[dict], workers: int) -> dict:
    count = len(samples)
    return {
        "workers": workers,
        "wall_ms_median": round(statistics.median(sample["wall_ms"] for sample in samples), 3),
        "wall_ms_samples": [round(sample["wall_ms"], 3) for sample in samples],
        "success_rate": round(sum(sample["success_count"] for sample in samples) /
                              sum(sample["task_count"] for sample in samples), 4),
        "llm_attempts_per_run": sum(sample["llm_attempts"] for sample in samples) / count,
        "retries_per_run": sum(sample["retries"] for sample in samples) / count,
        "tool_calls_per_run": sum(sample["tool_calls"] for sample in samples) / count,
    }


def build_report(task_counts: list[int], repeats: int, max_workers: int,
                 llm_delay_ms: float, tool_delay_ms: float, retry_delay_ms: float) -> dict:
    if not task_counts or any(count < 1 for count in task_counts) or repeats < 1 or max_workers < 1:
        raise ValueError("Task counts, repeats and workers must be positive")
    if min(llm_delay_ms, tool_delay_ms, retry_delay_ms) < 0:
        raise ValueError("Delays must be nonnegative")
    cases = []
    for count in task_counts:
        for topology in ("independent", "layered_dag"):
            for retry in (False, True):
                parallel_workers = min(count, max_workers)
                by_mode = {}
                for label, workers in (("serial", 1), ("parallel", parallel_workers)):
                    samples = [run_case(count, topology, retry, workers, llm_delay_ms / 1000,
                                        tool_delay_ms / 1000, retry_delay_ms / 1000)
                               for _ in range(repeats)]
                    by_mode[label] = summarize_runs(samples, workers)
                serial_ms = by_mode["serial"]["wall_ms_median"]
                parallel_ms = by_mode["parallel"]["wall_ms_median"]
                cases.append({
                    "task_count": count,
                    "topology": topology,
                    "failure_mode": "one_mock_429_per_agent" if retry else "normal",
                    **by_mode,
                    "speedup": round(serial_ms / parallel_ms, 3) if parallel_ms else None,
                })
    return {
        "kind": "offline_mock_benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {"task_counts": task_counts, "repeats": repeats,
                          "max_workers": max_workers, "mock_llm_delay_ms": llm_delay_ms,
                          "mock_tool_delay_ms": tool_delay_ms,
                          "mock_retry_after_ms": retry_delay_ms,
                          "tool_handlers_synchronized": False},
        "cases": cases,
    }


def markdown_report(report: dict) -> str:
    lines = ["# Phase 3 offline Mock benchmark", "",
             "These are local simulated delays, not real model or network measurements.", "",
             f"Configuration: `{json.dumps(report['configuration'], ensure_ascii=False)}`", "",
             "| Tasks | Topology | Failure mode | Serial median ms | Parallel workers | Parallel median ms | Speedup | Success rate | LLM attempts/run | Retries/run |",
             "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case in report["cases"]:
        serial, parallel = case["serial"], case["parallel"]
        lines.append(f"| {case['task_count']} | {case['topology']} | {case['failure_mode']} | "
                     f"{serial['wall_ms_median']:.3f} | {parallel['workers']} | "
                     f"{parallel['wall_ms_median']:.3f} | {case['speedup']:.3f} | "
                     f"{parallel['success_rate']:.2%} | {parallel['llm_attempts_per_run']:g} | "
                     f"{parallel['retries_per_run']:g} |")
    lines += ["", "Speedup = serial median wall time / parallel median wall time.",
              "Success rate = successful tasks / scheduled tasks over all repeats.",
              "Each task makes two successful Mock LLM calls and one Mock Tool call; retry mode adds one failed Mock LLM attempt.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-counts", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--llm-delay-ms", type=float, default=15)
    parser.add_argument("--tool-delay-ms", type=float, default=5)
    parser.add_argument("--retry-delay-ms", type=float, default=5)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "reports" / "phase3")
    args = parser.parse_args()
    report = build_report(args.task_counts, args.repeats, args.max_workers,
                          args.llm_delay_ms, args.tool_delay_ms, args.retry_delay_ms)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark.json"
    md_path = args.output_dir / "benchmark.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"Offline Mock benchmark: {len(report['cases'])} cases; JSON: {json_path}; Markdown: {md_path}")


if __name__ == "__main__":
    main()
